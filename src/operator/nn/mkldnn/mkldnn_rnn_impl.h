/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#ifndef MXNET_OPERATOR_NN_MKLDNN_MKLDNN_RNN_IMPL_H_
#define MXNET_OPERATOR_NN_MKLDNN_MKLDNN_RNN_IMPL_H_
#if MXNET_USE_MKLDNN == 1
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <mxnet/storage.h>
#include <algorithm>
#include <map>
#include <vector>
#include <utility>
#include <string>
#include "../../math.h"
#include "../../math_functions-inl.h"
#include "../../operator_common.h"
#include "../../rnn_impl.h"
#include "../../rnn-inl.h"
#include "mkldnn.hpp"
#include "./mkldnn_base-inl.h"

namespace mxnet {
namespace op {

using namespace mkldnn;

typedef ParamOpSign<RNNParam> MKLDNNRNNSignature;

algorithm GetMKLDNNAlgo(int mode,
                        int* ngates,
                        int* nstates) {
  algorithm algo;
  switch (mode) {
    case rnn_enum::kLstm:
      *ngates = 4;
      *nstates = 2;
      algo = algorithm::vanilla_lstm;
      break;
    default:
      LOG(FATAL) << "unsupported RNN mode" << mode;
      break;
  }
  return algo;
}

void ReorderData(memory src,
                 memory dst) {
  MKLDNNStream::Get()->RegisterPrim(reorder(src, dst));
  MKLDNNStream::Get()->Submit();
}

void ConcatData(memory::format src_format,
                memory::format dst_format,
                std::vector<memory::dims> srcs_cds,
                memory::dims dst_cds,
                memory::data_type mkldnn_dtype,
                int concat_dimension,
                std::vector<void*> srcs_data,
                memory dst) {
  auto cpu_engine = CpuEngine::Get()->get_engine();
  std::vector<memory::primitive_desc> srcs_pd;
  std::vector<memory> srcs;
  for (size_t i = 0; i < srcs_cds.size(); i++) {
    auto desc = memory::desc(srcs_cds[i], mkldnn_dtype, src_format);
    auto mpd = memory::primitive_desc(desc, cpu_engine);
    auto src_memory = memory(mpd, srcs_data[i]);
    srcs_pd.push_back(mpd);
    srcs.push_back(src_memory);
  }
  std::vector<primitive::at> inputs;
  for (size_t i = 0; i < srcs_cds.size(); i++) {
    inputs.push_back(srcs[i]);
  }
  auto dst_desc = memory::desc(dst_cds, mkldnn_dtype, dst_format);
  auto concat_pd = concat::primitive_desc(dst_desc, concat_dimension, srcs_pd);
  MKLDNNStream::Get()->RegisterPrim(concat(concat_pd, inputs, dst));
  MKLDNNStream::Get()->Submit();
}

// since there is different sematics of MKLDNN's Fused RNN and Mxnet FusedRNN,
// bidirectional will be fused layer by layer,
// unidirectional will be done by fused 1 + fused (L - 1) layers or fused L layers(when I = H)
template <typename DType>
void MKLDNNRNNForwardSingleLayerBi(bool state_outputs,
                                   const int T,
                                   const int N,
                                   const int I,
                                   const int H,
                                   DType* x_ptr,
                                   memory user_src_layer_memory,
                                   DType* hx_ptr,
                                   DType* cx_ptr,
                                   DType* w_ptr,
                                   DType* b_ptr,
                                   DType* y_ptr,
                                   DType* hy_ptr,
                                   DType* cy_ptr,
                                   std::vector<memory> *weight_bias_memory,
                                   std::vector<memory> *concat_weight_memory,
                                   std::vector<memory> *concat_iter_memory,
                                   std::vector<memory> *dst_memory,
                                   std::vector<memory> *iter_memory,
                                   int layer_index,
                                   int dtype,
                                   int mode) {
  int ngates = 0, nstates = 0;
  algorithm nalgorithm = GetMKLDNNAlgo(mode, &ngates, &nstates);
  memory::data_type mkldnn_dtype = get_mkldnn_type(dtype);
  const int single_cell_size = N * H;
  const int single_b_size = ngates * H;
  DType* wx = w_ptr;  //  ngates * H, I
  DType* wh = w_ptr + I * H * ngates;  //  ngates * H, H
  DType* back_wx = w_ptr + ngates * H * (I + H);
  DType* back_wh = back_wx + I * H * ngates;
  DType* bx = b_ptr;
  DType* bh = b_ptr + H * ngates;
  DType* back_bx = b_ptr + single_b_size * 2;
  DType* back_bh = back_bx + H * ngates;
  const int omp_threads = mxnet::engine::OpenMP::Get()->GetRecommendedOMPThreadCount();
  auto cpu_engine = CpuEngine::Get()->get_engine();
  auto null_memory_ = null_memory(cpu_engine);
  int offset1 = 0, offset2 = 0;
  memory::dims src_layer_tz = {T, N, I};
  memory::dims dst_layer_tz = {T, N, 2 * H};
  memory::dims weights_layer_tz = {1, 2, I, ngates, H};  //  ldigo
  memory::dims weights_layer_r_tz = {1, 1, I, ngates, H};  //  ldigo for reorder
  memory::dims weights_iter_tz = {1, 2, H, ngates, H};  //  ldigo
  memory::dims weights_iter_r_tz = {1, 1, H, ngates, H};  //  ldigo for reorder
  memory::dims bias_tz = {1, 2, ngates, H};
  memory::dims src_iter_tz = {1, 2, nstates, N, H};  //  ldsnc
  memory::dims dst_iter_tz = {1, 2, nstates, N, H};  //  ldsnc
  auto user_weight_layer_md = memory::desc(
      { weights_layer_tz }, mkldnn_dtype, memory::format::ldigo);
  auto user_weight_iter_md = memory::desc(
      { weights_iter_tz }, mkldnn_dtype, memory::format::ldigo);
  auto src_wx = (*concat_weight_memory)[2 * layer_index];
  auto src_wh = (*concat_weight_memory)[2 * layer_index + 1];
  std::vector<void*> srcs_data1;
  srcs_data1.push_back(wx);
  srcs_data1.push_back(back_wx);
  ConcatData(memory::format::ldgoi, memory::format::ldgoi,
      {weights_layer_r_tz, weights_layer_r_tz}, weights_layer_tz,
      mkldnn_dtype, 1, srcs_data1, src_wx);
  srcs_data1.clear();
  srcs_data1.push_back(wh);
  srcs_data1.push_back(back_wh);
  ConcatData(memory::format::ldgoi, memory::format::ldgoi,
      {weights_iter_r_tz, weights_iter_r_tz}, weights_iter_tz,
       mkldnn_dtype, 1, srcs_data1, src_wh);
  ReorderData(src_wx, (*weight_bias_memory)[3 * layer_index]);
  ReorderData(src_wh, (*weight_bias_memory)[3 * layer_index + 1]);
  DType* user_bias = reinterpret_cast<DType *>
      ((*weight_bias_memory)[3 * layer_index + 2].get_data_handle());
  #pragma omp parallel for num_threads(omp_threads)
  for (int j = 0; j < single_b_size; j++) {
    user_bias[j] = bx[j] + bh[j];
    user_bias[single_b_size + j] = back_bx[j] + back_bh[j];
  }
  auto user_src_layer_md = memory::desc(
      { src_layer_tz }, mkldnn_dtype, memory::format::tnc);
  auto user_bias_md = memory::desc({ bias_tz },
      mkldnn_dtype, memory::format::ldgo);
  auto dst_layer_md = memory::desc(
      { dst_layer_tz }, mkldnn_dtype, memory::format::tnc);
  auto dst_iter_md = memory::desc(
      { dst_iter_tz }, mkldnn_dtype, memory::format::ldsnc);
  if (x_ptr) {
    user_src_layer_memory = memory({ user_src_layer_md, cpu_engine }, x_ptr);
  }
  auto user_src_iter_md = memory::desc(
      { src_iter_tz }, mkldnn_dtype, memory::format::ldsnc);
  auto user_src_iter_memory = (*concat_iter_memory)[2];

  if (mode == rnn_enum::kLstm) {
    std::vector<void*> srcs_data1;
    srcs_data1.push_back(hx_ptr);
    srcs_data1.push_back(cx_ptr);
    auto tmp1_src_iter_memory = (*concat_iter_memory)[0];
    ConcatData(memory::format::ldsnc, memory::format::ldsnc,
        {{1, 1, 1, N, H}, {1, 1, 1, N, H}}, {1, 1, nstates, N, H}, mkldnn_dtype, 2,
        srcs_data1, tmp1_src_iter_memory);
    std::vector<void*> srcs_data2;
    srcs_data2.push_back(hx_ptr + single_cell_size);
    srcs_data2.push_back(cx_ptr + single_cell_size);
    auto tmp2_src_iter_memory = (*concat_iter_memory)[1];
    ConcatData(memory::format::ldsnc, memory::format::ldsnc,
        {{1, 1, 1, N, H}, {1, 1, 1, N, H}}, {1, 1, nstates, N, H}, mkldnn_dtype, 2,
        srcs_data2, tmp2_src_iter_memory);
    std::vector<void*> srcs_data3;
    srcs_data3.push_back(reinterpret_cast<DType *>(tmp1_src_iter_memory.get_data_handle()));
    srcs_data3.push_back(reinterpret_cast<DType *>(tmp2_src_iter_memory.get_data_handle()));
    ConcatData(memory::format::ldsnc, memory::format::ldsnc,
        {{1, 1, nstates, N, H}, {1, 1, nstates, N, H}}, {1, 2, nstates, N, H},
        mkldnn_dtype, 1, srcs_data3, user_src_iter_memory);
  }
  rnn_cell::desc cell(nalgorithm);
  rnn_forward::desc layer_desc(prop_kind::forward_inference, cell,
      rnn_direction::bidirectional_concat, user_src_layer_md,
      user_src_iter_md, user_weight_layer_md, user_weight_iter_md,
      user_bias_md, dst_layer_md, dst_iter_md);
  auto prim_desc = rnn_forward::primitive_desc(layer_desc, cpu_engine);
  auto dst_layer_memory = (*dst_memory)[layer_index];
  auto dst_iter_memory = (*iter_memory)[layer_index];
  dst_layer_memory.set_data_handle(y_ptr);
  MKLDNNStream::Get()->RegisterPrim(
      rnn_forward(prim_desc, user_src_layer_memory, user_src_iter_memory,
                  (*weight_bias_memory)[3 * layer_index],
                  (*weight_bias_memory)[3 * layer_index + 1],
                  (*weight_bias_memory)[3 * layer_index + 2], dst_layer_memory,
                  dst_iter_memory, null_memory_));
  MKLDNNStream::Get()->Submit();
  if (state_outputs) {
    offset1 = nstates * single_cell_size;
    offset2 = (nstates + 1) * single_cell_size;
    DType* dst_hcy = reinterpret_cast<DType *> (dst_iter_memory.get_data_handle());
    if (mode == rnn_enum::kLstm) {
      #pragma omp parallel for num_threads(omp_threads)
      for (int n = 0; n < single_cell_size; n++) {
        hy_ptr[n] = dst_hcy[n];
        hy_ptr[n + single_cell_size] = dst_hcy[n + offset1];
        cy_ptr[n] = dst_hcy[n + single_cell_size];
        cy_ptr[n + single_cell_size] = dst_hcy[n + offset2];
      }
    }
  }
}

template <typename DType>
void MKLDNNRNNForwardUnidi(bool state_outputs,
                           const int L,
                           const int T,
                           const int N,
                           const int I,
                           const int H,
                           DType* x_ptr,
                           memory user_src_layer_memory,
                           DType* hx_ptr,
                           DType* cx_ptr,
                           DType* w_ptr,
                           DType* b_ptr,
                           DType* y_ptr,
                           DType* hy_ptr,
                           DType* cy_ptr,
                           std::vector<memory> *weight_bias_memory,
                           std::vector<memory> *concat_weight_memory,
                           std::vector<memory> *concat_iter_memory,
                           std::vector<memory> *dst_memory,
                           std::vector<memory> *iter_memory,
                           int layer_index,
                           int dtype,
                           int mode) {
  int ngates = 0, nstates = 0;
  algorithm nalgorithm = GetMKLDNNAlgo(mode, &ngates, &nstates);
  memory::data_type mkldnn_dtype = get_mkldnn_type(dtype);
  const int cell_size = N * H;
  const int single_cell_size = N * H;
  const int single_b_size = ngates * H;
  int w_size = (I + H) * H * ngates;
  const int omp_threads = mxnet::engine::OpenMP::Get()->GetRecommendedOMPThreadCount();
  auto cpu_engine = CpuEngine::Get()->get_engine();
  auto null_memory_ = null_memory(cpu_engine);
  int offset1 = 0, offset2 = 0;
  memory::dims src_layer_tz = {T, N, I};
  memory::dims dst_layer_tz = {T, N, H};
  memory::dims weights_layer_r_tz = {1, 1, I, ngates, H};  //  ldigo for reorder
  memory::dims weights_iter_r_tz = {1, 1, H, ngates, H};  //  ldigo for reorder
  auto user_src_layer_md = memory::desc(
      { src_layer_tz }, mkldnn_dtype, memory::format::tnc);
  auto dst_layer_md = memory::desc(
      { dst_layer_tz }, mkldnn_dtype, memory::format::tnc);
  if (x_ptr) {
    user_src_layer_memory = memory({ user_src_layer_md, cpu_engine }, x_ptr);
  }
  memory::dims weights_layer_tz = {L, 1, I, ngates, H};  //  ldigo
  memory::dims weights_iter_tz = {L, 1, H, ngates, H};  //  ldigo
  memory::dims bias_tz = {L, 1, ngates, H};
  memory::dims src_iter_tz = {L, 1, nstates, N, H};  //  ldsnc
  memory::dims dst_iter_tz = {L, 1, nstates, N, H};  //  ldsnc
  auto user_src_iter_md = memory::desc(
      { src_iter_tz }, mkldnn_dtype, memory::format::ldsnc);
  auto user_weight_layer_md = memory::desc(
      { weights_layer_tz }, mkldnn_dtype, memory::format::ldigo);
  auto user_weight_iter_md = memory::desc(
      { weights_iter_tz }, mkldnn_dtype, memory::format::ldigo);
  auto user_bias_md = memory::desc({ bias_tz },
      mkldnn_dtype, memory::format::ldgo);
  auto dst_iter_md = memory::desc(
      { dst_iter_tz }, mkldnn_dtype, memory::format::ldsnc);
  rnn_cell::desc cell(nalgorithm);
  rnn_forward::desc layer_desc(prop_kind::forward_inference, cell,
      rnn_direction::unidirectional, user_src_layer_md, user_src_iter_md,
      user_weight_layer_md, user_weight_iter_md, user_bias_md, dst_layer_md, dst_iter_md);
  auto prim_desc = rnn_forward::primitive_desc(layer_desc, cpu_engine);
  auto dst_layer_memory = (*dst_memory)[layer_index];
  dst_layer_memory.set_data_handle(y_ptr);
  auto dst_iter_memory = (*iter_memory)[layer_index];

  for (int l = 0; l < L; l++) {
    if (mode == rnn_enum::kLstm) {
      std::vector<void*> srcs_data;
      srcs_data.push_back(hx_ptr);
      srcs_data.push_back(cx_ptr);
      auto tmp_src_iter_memory = (*concat_iter_memory)[l + layer_index];
      ConcatData(memory::format::ldsnc, memory::format::ldsnc,
          {{1, 1, 1, N, H}, {1, 1, 1, N, H}}, {1, 1, nstates, N, H}, mkldnn_dtype,
          2, srcs_data, tmp_src_iter_memory);
    }
    hx_ptr += cell_size;
    if (mode == rnn_enum::kLstm) {
      cx_ptr += cell_size;
    }
  }

  auto user_src_iter_memory = null_memory_;
  if (L == 1) {
    user_src_iter_memory = (*concat_iter_memory)[layer_index];
  } else {
    user_src_iter_memory = (*concat_iter_memory)[L + layer_index];
    std::vector<void*> src_l_data;
    std::vector<memory::dims> src_l_dim;
    for (int l = 0; l < L; l++) {
      src_l_data.push_back(reinterpret_cast<DType *>
          ((*concat_iter_memory)[l + layer_index].get_data_handle()));
      src_l_dim.push_back({1, 1, nstates, N, H});
    }
    ConcatData(memory::format::ldsnc, memory::format::ldsnc, src_l_dim,
        {L, 1, nstates, N, H}, mkldnn_dtype, 0, src_l_data, user_src_iter_memory);
  }

  auto src_wx_f = (*concat_weight_memory)[2 * layer_index];
  auto src_wh_f = (*concat_weight_memory)[2 * layer_index + 1];
  std::vector<void*> srcs_data_x;
  std::vector<void*> srcs_data_h;
  std::vector<memory::dims> src_l_dim_x;
  std::vector<memory::dims> src_l_dim_h;
  if (L == 1) {
    DType* wx = w_ptr;
    DType* wh = w_ptr + I * H * ngates;
    src_wx_f.set_data_handle(wx);
    src_wh_f.set_data_handle(wh);
  } else {
    for (int l = 0; l < L; l++) {
      DType* wx = w_ptr;
      DType* wh = w_ptr + I * H * ngates;
      srcs_data_x.push_back(wx);
      srcs_data_h.push_back(wh);
      src_l_dim_x.push_back(weights_layer_r_tz);
      src_l_dim_h.push_back(weights_iter_r_tz);
      w_ptr = w_ptr + w_size;
    }
    ConcatData(memory::format::ldgoi, memory::format::ldgoi,
        src_l_dim_x, weights_layer_tz, mkldnn_dtype, 0, srcs_data_x, src_wx_f);
    ConcatData(memory::format::ldgoi, memory::format::ldgoi,
        src_l_dim_h, weights_iter_tz, mkldnn_dtype, 0, srcs_data_h, src_wh_f);
  }
  ReorderData(src_wx_f, (*weight_bias_memory)[3 * layer_index]);  // reorder L layers wx
  ReorderData(src_wh_f, (*weight_bias_memory)[3 * layer_index + 1]);  // reorder L layers wh
  DType* user_bias_f = reinterpret_cast<DType *>
      ((*weight_bias_memory)[3 * layer_index + 2].get_data_handle());
  #pragma omp parallel for num_threads(omp_threads)
  for (int j = 0; j < L * single_b_size; j++) {
    int k = j / single_b_size;
    user_bias_f[j] = b_ptr[j + k * single_b_size] + b_ptr[j + k * single_b_size + single_b_size];
  }
  MKLDNNStream::Get()->RegisterPrim(
      rnn_forward(prim_desc, user_src_layer_memory,
                  user_src_iter_memory, (*weight_bias_memory)[3 * layer_index],
                  (*weight_bias_memory)[3 * layer_index + 1],
                  (*weight_bias_memory)[3 * layer_index + 2],
                  dst_layer_memory, dst_iter_memory, null_memory_));
  MKLDNNStream::Get()->Submit();
  if (state_outputs) {
    DType* dst_hcy = reinterpret_cast<DType *> (dst_iter_memory.get_data_handle());
    for (int l = 0; l < L; l++) {
      offset1 = l * single_cell_size;
      offset2 = l * nstates * single_cell_size;
      if (mode == rnn_enum::kLstm) {
        #pragma omp parallel for num_threads(omp_threads)
        for (int n = 0; n < single_cell_size; n++) {
          hy_ptr[offset1 + n] = dst_hcy[offset2 + n];
          cy_ptr[offset1 + n] = dst_hcy[offset2 + n + single_cell_size];
        }
      }
    }
  }
}

template <typename DType>
void MKLDNNRNNForwardUnidi_hardcode(bool state_outputs,
                                    const int L,
                                    const int T,
                                    const int N,
                                    const int I,
                                    const int H,
                                    DType* x_ptr,
                                    memory user_src_layer_memory,
                                    DType* hx_ptr,
                                    DType* cx_ptr,
                                    DType* w_ptr,
                                    DType* b_ptr,
                                    DType* y_ptr,
                                    DType* hy_ptr,
                                    DType* cy_ptr,
                                    std::vector<memory> *weight_bias_memory,
                                    std::vector<memory> *concat_weight_memory,
                                    std::vector<memory> *concat_iter_memory,
                                    std::vector<memory> *dst_memory,
                                    std::vector<memory> *iter_memory,
                                    int layer_index,
                                    int dtype,
                                    int mode,
                                    int* hardcode_count) {
  //  hard code for sockeye size testing
  //  L == 1 && D == 1 && T == 1 && N == 640 && I == 1024 && H == 512
  //  L == 1 && D == 1 && T == 1 && N == 640 && I == 768 && H == 512
  //  L == 7 && D == 1 && T == 10 - 70, 101 && N == 64 && I == 512 && H == 512
  int ngates = 0, nstates = 0;
  algorithm nalgorithm = GetMKLDNNAlgo(mode, &ngates, &nstates);
  memory::data_type mkldnn_dtype = get_mkldnn_type(dtype);
  const int cell_size = N * H;
  const int single_cell_size = N * H;
  const int single_b_size = ngates * H;
  int w_size = (I + H) * H * ngates;
  const int omp_threads = mxnet::engine::OpenMP::Get()->GetRecommendedOMPThreadCount();
  auto cpu_engine = CpuEngine::Get()->get_engine();
  auto null_memory_ = null_memory(cpu_engine);
  int offset1 = 0, offset2 = 0;
  memory::dims src_layer_tz = {T, N, I};
  memory::dims dst_layer_tz = {T, N, H};
  memory::dims weights_layer_r_tz = {1, 1, I, ngates, H};  //  ldigo for reorder
  memory::dims weights_iter_r_tz = {1, 1, H, ngates, H};  //  ldigo for reorder
  auto user_src_layer_md = memory::desc(
      { src_layer_tz }, mkldnn_dtype, memory::format::tnc);
  auto dst_layer_md = memory::desc(
      { dst_layer_tz }, mkldnn_dtype, memory::format::tnc);
  if (x_ptr) {
    user_src_layer_memory = memory({ user_src_layer_md, cpu_engine }, x_ptr);
  }
  memory::dims weights_layer_tz = {L, 1, I, ngates, H};  //  ldigo
  memory::dims weights_iter_tz = {L, 1, H, ngates, H};  //  ldigo
  memory::dims bias_tz = {L, 1, ngates, H};
  memory::dims src_iter_tz = {L, 1, nstates, N, H};  //  ldsnc
  memory::dims dst_iter_tz = {L, 1, nstates, N, H};  //  ldsnc
  auto user_src_iter_md = memory::desc(
      { src_iter_tz }, mkldnn_dtype, memory::format::ldsnc);
  auto user_weight_layer_md = memory::desc(
      { weights_layer_tz }, mkldnn_dtype, memory::format::ldigo);
  auto user_weight_iter_md = memory::desc(
      { weights_iter_tz }, mkldnn_dtype, memory::format::ldigo);
  auto user_bias_md = memory::desc({ bias_tz },
      mkldnn_dtype, memory::format::ldgo);
  auto dst_iter_md = memory::desc(
      { dst_iter_tz }, mkldnn_dtype, memory::format::ldsnc);
  rnn_cell::desc cell(nalgorithm);
  rnn_forward::desc layer_desc(prop_kind::forward_inference, cell,
      rnn_direction::unidirectional, user_src_layer_md, user_src_iter_md,
      user_weight_layer_md, user_weight_iter_md, user_bias_md, dst_layer_md, dst_iter_md);
  auto prim_desc = rnn_forward::primitive_desc(layer_desc, cpu_engine);
  auto dst_layer_memory = (*dst_memory)[layer_index];
  dst_layer_memory.set_data_handle(y_ptr);
  auto dst_iter_memory = (*iter_memory)[layer_index];

  for (int l = 0; l < L; l++) {
    if (mode == rnn_enum::kLstm) {
      std::vector<void*> srcs_data;
      srcs_data.push_back(hx_ptr);
      srcs_data.push_back(cx_ptr);
      auto tmp_src_iter_memory = (*concat_iter_memory)[l + layer_index];
      ConcatData(memory::format::ldsnc, memory::format::ldsnc,
          {{1, 1, 1, N, H}, {1, 1, 1, N, H}}, {1, 1, nstates, N, H}, mkldnn_dtype,
          2, srcs_data, tmp_src_iter_memory);
    }
    hx_ptr += cell_size;
    if (mode == rnn_enum::kLstm) {
      cx_ptr += cell_size;
    }
  }

  auto user_src_iter_memory = null_memory_;
  if (L == 1) {
    user_src_iter_memory = (*concat_iter_memory)[layer_index];
  } else {
    user_src_iter_memory = (*concat_iter_memory)[L + layer_index];
    std::vector<void*> src_l_data;
    std::vector<memory::dims> src_l_dim;
    for (int l = 0; l < L; l++) {
      src_l_data.push_back(reinterpret_cast<DType *>
          ((*concat_iter_memory)[l + layer_index].get_data_handle()));
      src_l_dim.push_back({1, 1, nstates, N, H});
    }
    ConcatData(memory::format::ldsnc, memory::format::ldsnc, src_l_dim,
        {L, 1, nstates, N, H}, mkldnn_dtype, 0, src_l_data, user_src_iter_memory);
  }

  auto src_wx_f = (*concat_weight_memory)[2 * layer_index];
  auto src_wh_f = (*concat_weight_memory)[2 * layer_index + 1];
  std::vector<void*> srcs_data_x;
  std::vector<void*> srcs_data_h;
  std::vector<memory::dims> src_l_dim_x;
  std::vector<memory::dims> src_l_dim_h;
  if (L == 1) {
    if (I == 1024) {
      if ((*hardcode_count) <= 7) {
        DType* wx = w_ptr;
        DType* wh = w_ptr + I * H * ngates;
        src_wx_f.set_data_handle(wx);
        src_wh_f.set_data_handle(wh);
        layer_index = (*hardcode_count) - 1;
        ReorderData(src_wx_f, (*weight_bias_memory)[3 * layer_index]);  // reorder L layers wx
        ReorderData(src_wh_f, (*weight_bias_memory)[3 * layer_index + 1]);  // reorder L layers wh
        DType* user_bias_f = reinterpret_cast<DType *>
            ((*weight_bias_memory)[3 * layer_index + 2].get_data_handle());
        #pragma omp parallel for num_threads(omp_threads)
        for (int j = 0; j < L * single_b_size; j++) {
          int k = j / single_b_size;
          user_bias_f[j] = b_ptr[j + k * single_b_size] +
              b_ptr[j + k * single_b_size + single_b_size];
        }
      } else {
        layer_index = ((*hardcode_count) - 1) % 7;
      }
    } else if (I == 768) {
      if ((*hardcode_count) <= 1) {
        DType* wx = w_ptr;
        DType* wh = w_ptr + I * H * ngates;
        src_wx_f.set_data_handle(wx);
        src_wh_f.set_data_handle(wh);
        layer_index = (*hardcode_count) - 1;
        ReorderData(src_wx_f, (*weight_bias_memory)[3 * layer_index]);  // reorder L layers wx
        ReorderData(src_wh_f, (*weight_bias_memory)[3 * layer_index + 1]);  // reorder L layers wh
        DType* user_bias_f = reinterpret_cast<DType *>
            ((*weight_bias_memory)[3 * layer_index + 2].get_data_handle());
        #pragma omp parallel for num_threads(omp_threads)
        for (int j = 0; j < L * single_b_size; j++) {
          int k = j / single_b_size;
          user_bias_f[j] = b_ptr[j + k * single_b_size] +
              b_ptr[j + k * single_b_size + single_b_size];
        }
      } else {
        layer_index = 0;
      }
    }
  } else {
    if (I == 512) {
      if ((*hardcode_count) <= 1) {
        for (int l = 0; l < L; l++) {
          DType* wx = w_ptr;
          DType* wh = w_ptr + I * H * ngates;
          srcs_data_x.push_back(wx);
          srcs_data_h.push_back(wh);
          src_l_dim_x.push_back(weights_layer_r_tz);
          src_l_dim_h.push_back(weights_iter_r_tz);
          w_ptr = w_ptr + w_size;
        }
        ConcatData(memory::format::ldgoi, memory::format::ldgoi,
            src_l_dim_x, weights_layer_tz, mkldnn_dtype, 0, srcs_data_x, src_wx_f);
        ConcatData(memory::format::ldgoi, memory::format::ldgoi,
            src_l_dim_h, weights_iter_tz, mkldnn_dtype, 0, srcs_data_h, src_wh_f);
        layer_index = (*hardcode_count) - 1;
        ReorderData(src_wx_f, (*weight_bias_memory)[3 * layer_index]);  // reorder L layers wx
        ReorderData(src_wh_f, (*weight_bias_memory)[3 * layer_index + 1]);  // reorder L layers wh
        DType* user_bias_f = reinterpret_cast<DType *>
            ((*weight_bias_memory)[3 * layer_index + 2].get_data_handle());
        #pragma omp parallel for num_threads(omp_threads)
        for (int j = 0; j < L * single_b_size; j++) {
          int k = j / single_b_size;
          user_bias_f[j] = b_ptr[j + k * single_b_size] +
              b_ptr[j + k * single_b_size + single_b_size];
        }
      } else {
        layer_index = 0;
      }
    }
  }
  MKLDNNStream::Get()->RegisterPrim(
      rnn_forward(prim_desc, user_src_layer_memory,
                  user_src_iter_memory, (*weight_bias_memory)[3 * layer_index],
                  (*weight_bias_memory)[3 * layer_index + 1],
                  (*weight_bias_memory)[3 * layer_index + 2],
                  dst_layer_memory, dst_iter_memory, null_memory_));
  MKLDNNStream::Get()->Submit();
  if (state_outputs) {
    DType* dst_hcy = reinterpret_cast<DType *> (dst_iter_memory.get_data_handle());
    for (int l = 0; l < L; l++) {
      offset1 = l * single_cell_size;
      offset2 = l * nstates * single_cell_size;
      if (mode == rnn_enum::kLstm) {
        #pragma omp parallel for num_threads(omp_threads)
        for (int n = 0; n < single_cell_size; n++) {
          hy_ptr[offset1 + n] = dst_hcy[offset2 + n];
          cy_ptr[offset1 + n] = dst_hcy[offset2 + n + single_cell_size];
        }
      }
    }
  }
}

template <typename DType>
void MKLDNNRNNForwardSingleLayerBi_hardcode(bool state_outputs,
                                            const int T,
                                            const int N,
                                            const int I,
                                            const int H,
                                            DType* x_ptr,
                                            memory user_src_layer_memory,
                                            DType* hx_ptr,
                                            DType* cx_ptr,
                                            DType* w_ptr,
                                            DType* b_ptr,
                                            DType* y_ptr,
                                            DType* hy_ptr,
                                            DType* cy_ptr,
                                            std::vector<memory> *weight_bias_memory,
                                            std::vector<memory> *concat_weight_memory,
                                            std::vector<memory> *concat_iter_memory,
                                            std::vector<memory> *dst_memory,
                                            std::vector<memory> *iter_memory,
                                            int layer_index,
                                            int dtype,
                                            int mode,
                                            int* hardcode_count) {
  //  hard code for sockeye size testing
  //  L == 1 && D == 2 && T == 10 - 70, 101 && N == 64 && I == 256 && H == 256
  int ngates = 0, nstates = 0;
  algorithm nalgorithm = GetMKLDNNAlgo(mode, &ngates, &nstates);
  memory::data_type mkldnn_dtype = get_mkldnn_type(dtype);
  const int single_cell_size = N * H;
  const int single_b_size = ngates * H;
  DType* wx = w_ptr;  //  ngates * H, I
  DType* wh = w_ptr + I * H * ngates;  //  ngates * H, H
  DType* back_wx = w_ptr + ngates * H * (I + H);
  DType* back_wh = back_wx + I * H * ngates;
  DType* bx = b_ptr;
  DType* bh = b_ptr + H * ngates;
  DType* back_bx = b_ptr + single_b_size * 2;
  DType* back_bh = back_bx + H * ngates;
  const int omp_threads = mxnet::engine::OpenMP::Get()->GetRecommendedOMPThreadCount();
  auto cpu_engine = CpuEngine::Get()->get_engine();
  auto null_memory_ = null_memory(cpu_engine);
  int offset1 = 0, offset2 = 0;
  memory::dims src_layer_tz = {T, N, I};
  memory::dims dst_layer_tz = {T, N, 2 * H};
  memory::dims weights_layer_tz = {1, 2, I, ngates, H};  //  ldigo
  memory::dims weights_layer_r_tz = {1, 1, I, ngates, H};  //  ldigo for reorder
  memory::dims weights_iter_tz = {1, 2, H, ngates, H};  //  ldigo
  memory::dims weights_iter_r_tz = {1, 1, H, ngates, H};  //  ldigo for reorder
  memory::dims bias_tz = {1, 2, ngates, H};
  memory::dims src_iter_tz = {1, 2, nstates, N, H};  //  ldsnc
  memory::dims dst_iter_tz = {1, 2, nstates, N, H};  //  ldsnc
  auto user_weight_layer_md = memory::desc(
      { weights_layer_tz }, mkldnn_dtype, memory::format::ldigo);
  auto user_weight_iter_md = memory::desc(
      { weights_iter_tz }, mkldnn_dtype, memory::format::ldigo);
  auto src_wx = (*concat_weight_memory)[2 * layer_index];
  auto src_wh = (*concat_weight_memory)[2 * layer_index + 1];
  if (I == 256) {
    if ((*hardcode_count) <= 1) {
      std::vector<void*> srcs_data1;
      srcs_data1.push_back(wx);
      srcs_data1.push_back(back_wx);
      ConcatData(memory::format::ldgoi, memory::format::ldgoi,
          {weights_layer_r_tz, weights_layer_r_tz}, weights_layer_tz,
          mkldnn_dtype, 1, srcs_data1, src_wx);
      srcs_data1.clear();
      srcs_data1.push_back(wh);
      srcs_data1.push_back(back_wh);
      ConcatData(memory::format::ldgoi, memory::format::ldgoi,
          {weights_iter_r_tz, weights_iter_r_tz}, weights_iter_tz,
           mkldnn_dtype, 1, srcs_data1, src_wh);
      layer_index = (*hardcode_count) - 1;
      ReorderData(src_wx, (*weight_bias_memory)[3 * layer_index]);  // reorder L layers wx
      ReorderData(src_wh, (*weight_bias_memory)[3 * layer_index + 1]);  // reorder L layers wh
      DType* user_bias = reinterpret_cast<DType *>
          ((*weight_bias_memory)[3 * layer_index + 2].get_data_handle());
      #pragma omp parallel for num_threads(omp_threads)
      for (int j = 0; j < single_b_size; j++) {
        user_bias[j] = bx[j] + bh[j];
        user_bias[single_b_size + j] = back_bx[j] + back_bh[j];
      }
    } else {
      layer_index = 0;
    }
  }
  auto user_src_layer_md = memory::desc(
      { src_layer_tz }, mkldnn_dtype, memory::format::tnc);
  auto user_bias_md = memory::desc({ bias_tz },
      mkldnn_dtype, memory::format::ldgo);
  auto dst_layer_md = memory::desc(
      { dst_layer_tz }, mkldnn_dtype, memory::format::tnc);
  auto dst_iter_md = memory::desc(
      { dst_iter_tz }, mkldnn_dtype, memory::format::ldsnc);
  if (x_ptr) {
    user_src_layer_memory = memory({ user_src_layer_md, cpu_engine }, x_ptr);
  }
  auto user_src_iter_md = memory::desc(
      { src_iter_tz }, mkldnn_dtype, memory::format::ldsnc);
  auto user_src_iter_memory = (*concat_iter_memory)[2];

  if (mode == rnn_enum::kLstm) {
    std::vector<void*> srcs_data1;
    srcs_data1.push_back(hx_ptr);
    srcs_data1.push_back(cx_ptr);
    auto tmp1_src_iter_memory = (*concat_iter_memory)[0];
    ConcatData(memory::format::ldsnc, memory::format::ldsnc,
        {{1, 1, 1, N, H}, {1, 1, 1, N, H}}, {1, 1, nstates, N, H}, mkldnn_dtype, 2,
        srcs_data1, tmp1_src_iter_memory);
    std::vector<void*> srcs_data2;
    srcs_data2.push_back(hx_ptr + single_cell_size);
    srcs_data2.push_back(cx_ptr + single_cell_size);
    auto tmp2_src_iter_memory = (*concat_iter_memory)[1];
    ConcatData(memory::format::ldsnc, memory::format::ldsnc,
        {{1, 1, 1, N, H}, {1, 1, 1, N, H}}, {1, 1, nstates, N, H}, mkldnn_dtype, 2,
        srcs_data2, tmp2_src_iter_memory);
    std::vector<void*> srcs_data3;
    srcs_data3.push_back(reinterpret_cast<DType *>(tmp1_src_iter_memory.get_data_handle()));
    srcs_data3.push_back(reinterpret_cast<DType *>(tmp2_src_iter_memory.get_data_handle()));
    ConcatData(memory::format::ldsnc, memory::format::ldsnc,
        {{1, 1, nstates, N, H}, {1, 1, nstates, N, H}}, {1, 2, nstates, N, H},
        mkldnn_dtype, 1, srcs_data3, user_src_iter_memory);
  }
  rnn_cell::desc cell(nalgorithm);
  rnn_forward::desc layer_desc(prop_kind::forward_inference, cell,
      rnn_direction::bidirectional_concat, user_src_layer_md,
      user_src_iter_md, user_weight_layer_md, user_weight_iter_md,
      user_bias_md, dst_layer_md, dst_iter_md);
  auto prim_desc = rnn_forward::primitive_desc(layer_desc, cpu_engine);
  auto dst_layer_memory = (*dst_memory)[layer_index];
  auto dst_iter_memory = (*iter_memory)[layer_index];
  dst_layer_memory.set_data_handle(y_ptr);
  MKLDNNStream::Get()->RegisterPrim(
      rnn_forward(prim_desc, user_src_layer_memory, user_src_iter_memory,
                  (*weight_bias_memory)[3 * layer_index],
                  (*weight_bias_memory)[3 * layer_index + 1],
                  (*weight_bias_memory)[3 * layer_index + 2], dst_layer_memory,
                  dst_iter_memory, null_memory_));
  MKLDNNStream::Get()->Submit();
  if (state_outputs) {
    offset1 = nstates * single_cell_size;
    offset2 = (nstates + 1) * single_cell_size;
    DType* dst_hcy = reinterpret_cast<DType *> (dst_iter_memory.get_data_handle());
    if (mode == rnn_enum::kLstm) {
      #pragma omp parallel for num_threads(omp_threads)
      for (int n = 0; n < single_cell_size; n++) {
        hy_ptr[n] = dst_hcy[n];
        hy_ptr[n + single_cell_size] = dst_hcy[n + offset1];
        cy_ptr[n] = dst_hcy[n + single_cell_size];
        cy_ptr[n + single_cell_size] = dst_hcy[n + offset2];
      }
    }
  }
}

template <typename DType>
void MKLDNNRNNForward(bool state_outputs,
                      const int L,
                      const int D,
                      const int T,
                      const int N,
                      const int I,
                      const int H,
                      DType* x_ptr,
                      DType* hx_ptr,
                      DType* cx_ptr,
                      DType* w_ptr,
                      DType* b_ptr,
                      DType* y_ptr,
                      DType* hy_ptr,
                      DType* cy_ptr,
                      std::vector<memory> *weight_bias_memory,
                      std::vector<memory> *concat_weight_memory,
                      std::vector<memory> *concat_iter_memory,
                      std::vector<memory> *dst_memory,
                      std::vector<memory> *iter_memory,
                      int dtype,
                      int mode,
                      int* hardcode_count) {
  int ngates = 0, nstates = 0;
  GetMKLDNNAlgo(mode, &ngates, &nstates);
  const int b_size = 2 * H * ngates * D;
  const int cell_size = N * H * D;
  //  First layer
  int w_size = (I + H) * H * ngates * D;
  auto cpu_engine = CpuEngine::Get()->get_engine();
  auto null_memory_ = null_memory(cpu_engine);
  DType* tmpNull = NULL;
  if (L == 1 && D == 1 && T == 1 && N == 640 && I == 1024 && H == 512) {
    (*hardcode_count)++;
    MKLDNNRNNForwardUnidi_hardcode(state_outputs, L, T, N, I, H, x_ptr, null_memory_,
        hx_ptr, cx_ptr, w_ptr, b_ptr, y_ptr, hy_ptr, cy_ptr, weight_bias_memory,
        concat_weight_memory, concat_iter_memory, dst_memory, iter_memory, 0,
        dtype, mode, hardcode_count);
    return;
  }
  if (L == 1 && D == 1 && T == 1 && N == 640 && I == 768 && H == 512) {
    (*hardcode_count)++;
    MKLDNNRNNForwardUnidi_hardcode(state_outputs, L, T, N, I, H, x_ptr, null_memory_,
        hx_ptr, cx_ptr, w_ptr, b_ptr, y_ptr, hy_ptr, cy_ptr, weight_bias_memory,
        concat_weight_memory, concat_iter_memory, dst_memory, iter_memory, 0,
        dtype, mode, hardcode_count);
    return;
  }

  if (L == 7 && D == 1 && N == 64 && I == 512 && H == 512) {
    (*hardcode_count)++;
    MKLDNNRNNForwardUnidi_hardcode(state_outputs, L, T, N, I, H, x_ptr, null_memory_,
        hx_ptr, cx_ptr, w_ptr, b_ptr, y_ptr, hy_ptr, cy_ptr, weight_bias_memory,
        concat_weight_memory, concat_iter_memory, dst_memory, iter_memory, 0,
        dtype, mode, hardcode_count);
    return;
  }

  if (L == 1 && D == 2 && N == 64 && I == 256 && H == 256) {
    (*hardcode_count)++;
    MKLDNNRNNForwardSingleLayerBi_hardcode(state_outputs, T, N, I, H, x_ptr, null_memory_,
        hx_ptr, cx_ptr, w_ptr, b_ptr, y_ptr, hy_ptr, cy_ptr, weight_bias_memory,
        concat_weight_memory, concat_iter_memory, dst_memory, iter_memory, 0,
        dtype, mode, hardcode_count);
    return;
  }

  // when D = 1 and I == H, L layers can be fused together
  if (D == 1 && I == H && L > 1) {
    MKLDNNRNNForwardUnidi(state_outputs, L, T, N, I, H, x_ptr, null_memory_,
        hx_ptr, cx_ptr, w_ptr, b_ptr, y_ptr, hy_ptr, cy_ptr, weight_bias_memory,
        concat_weight_memory, concat_iter_memory, dst_memory, iter_memory, 0, dtype, mode);
  } else {  //  common case first layer
    auto user_src_layer_memory_l = null_memory_;
    if (D == 2) {
      MKLDNNRNNForwardSingleLayerBi(state_outputs, T, N, I, H, x_ptr, user_src_layer_memory_l,
          hx_ptr, cx_ptr, w_ptr, b_ptr, y_ptr, hy_ptr, cy_ptr, weight_bias_memory,
          concat_weight_memory, concat_iter_memory, dst_memory, iter_memory, 0, dtype, mode);
    } else {
      MKLDNNRNNForwardUnidi(state_outputs, 1, T, N, I, H, x_ptr, user_src_layer_memory_l,
          hx_ptr, cx_ptr, w_ptr, b_ptr, y_ptr, hy_ptr, cy_ptr, weight_bias_memory,
          concat_weight_memory, concat_iter_memory, dst_memory, iter_memory, 0, dtype, mode);
    }
    user_src_layer_memory_l = (*dst_memory)[0];
    //  go to next L - 1 layers.
    //  If D = 2, do it layer by layer. If D = 1, fused L - 1 layers
    w_ptr += w_size;
    b_ptr += b_size;
    if (L > 1 && D == 2) {
      w_size = (H * D + H) * H * ngates * D;
      for (int l = 0; l < L - 1; l++) {
        if (state_outputs) {
          hy_ptr += cell_size;
          if (mode == rnn_enum::kLstm) {
            cy_ptr += cell_size;
          }
        }
        hx_ptr += cell_size;
        if (mode == rnn_enum::kLstm) {
          cx_ptr += cell_size;
        }
        MKLDNNRNNForwardSingleLayerBi(state_outputs, T, N, D * H, H, tmpNull,
            user_src_layer_memory_l, hx_ptr, cx_ptr, w_ptr, b_ptr, y_ptr, hy_ptr, cy_ptr,
            weight_bias_memory, concat_weight_memory, concat_iter_memory,
            dst_memory, iter_memory, l + 1, dtype, mode);
        user_src_layer_memory_l = (*dst_memory)[l + 1];
        w_ptr += w_size;
        b_ptr += b_size;
     }
    }
    if (L > 1 && D == 1) {
      w_size = (H + H) * H * ngates;
      MKLDNNRNNForwardUnidi(state_outputs, L - 1, T, N, H, H, tmpNull, user_src_layer_memory_l,
          hx_ptr, cx_ptr, w_ptr, b_ptr, y_ptr, hy_ptr, cy_ptr, weight_bias_memory,
          concat_weight_memory, concat_iter_memory, dst_memory, iter_memory, 1, dtype, mode);
    }
  }
}

template <typename DType>
void MKLDNNRNNForwardTraining(bool state_outputs,
                              const int num_layers,
                              const int direction,
                              const int seq_length,
                              const int batch_size,
                              const int input_size,
                              const int state_size,
                              DType* x_ptr,
                              DType* hx_ptr,
                              DType* cx_ptr,
                              DType* w_ptr,
                              DType* b_ptr,
                              DType* y_ptr,
                              DType* hy_ptr,
                              DType* cy_ptr,
                              const float dropout,
                              int mode) {
  switch (mode) {
    case rnn_enum::kLstm:
    case rnn_enum::kGru:
    case rnn_enum::kRnnTanh:
    case rnn_enum::kRnnRelu:
    default:
      LOG(FATAL) << "unknown RNN mode " << mode;
      break;
  }
}

template <typename DType>
void MKLDNNRNNForwardInference(bool state_outputs,
                               const int num_layers,
                               const int direction,
                               const int seq_length,
                               const int batch_size,
                               const int input_size,
                               const int state_size,
                               DType* x_ptr,
                               DType* hx_ptr,
                               DType* cx_ptr,
                               DType* w_ptr,
                               DType* b_ptr,
                               DType* y_ptr,
                               DType* hy_ptr,
                               DType* cy_ptr,
                               std::vector<memory>* weight_bias_memory,
                               std::vector<memory>* concat_weight_memory,
                               std::vector<memory>* concat_iter_memory,
                               std::vector<memory>* dst_memory,
                               std::vector<memory>* iter_memory,
                               int dtype,
                               int mode,
                               int* hardcode_count) {
  switch (mode) {
    case rnn_enum::kLstm:
      MKLDNNRNNForward<DType>(state_outputs, num_layers, direction, seq_length,
                              batch_size, input_size, state_size, x_ptr, hx_ptr,
                              cx_ptr, w_ptr, b_ptr, y_ptr, hy_ptr, cy_ptr,
                              weight_bias_memory, concat_weight_memory,
                              concat_iter_memory, dst_memory, iter_memory, dtype,
                              mode, hardcode_count);
      break;
    case rnn_enum::kGru:
    case rnn_enum::kRnnTanh:
    case rnn_enum::kRnnRelu:
    default:
      LOG(FATAL) << "unknown RNN mode" << mode;
      break;
  }
}

template <typename DType>
void MKLDNNRNNBackward(const int num_layers,
                       const int direction,
                       const int seq_length,
                       const int batch_size,
                       const int input_size,
                       const int state_size,
                       DType* x_ptr,
                       DType* hx_ptr,
                       DType* cx_ptr,
                       DType* w_ptr,
                       DType* y_ptr,
                       DType* dy_ptr,
                       DType* dhy_ptr,
                       DType* dcy_ptr,
                       DType* dx_ptr,
                       DType* dhx_ptr,
                       DType* dcx_ptr,
                       DType* dw_ptr,
                       DType* db_ptr,
                       int req_data,
                       int req_params,
                       int req_state,
                       int req_statecell,
                       const float dropout,
                       int mode) {
  switch (mode) {
    case rnn_enum::kLstm:
    case rnn_enum::kGru:
    case rnn_enum::kRnnTanh:
    case rnn_enum::kRnnRelu:
    default:
      LOG(FATAL) << "unknown RNN mode" << mode;
      break;
  }
}

template<typename DType>
class MKLDNNRNNOp {
 public:
  explicit MKLDNNRNNOp(RNNParam p) {
    param_ = p;
    init_mem_ = false;
    reserve_mem_size_ = 0;
  }

  ~MKLDNNRNNOp() {
    if (init_mem_) {
      Storage::Get()->Free(mem_space_);
      init_mem_ = false;
    }
  }

  void Forward(const OpContext &ctx,
               const std::vector<TBlob> &in_data,
               const std::vector<OpReqType> &req,
               const std::vector<TBlob> &out_data) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK(param_.p >= 0.0f && param_.p < 1.0f)
        << "unsupported dropout value, should be 0 <= dropout < 1";

    size_t in_expected = (param_.mode == rnn_enum::kLstm) ? 4 : 3;
    size_t out_expected = (param_.mode == rnn_enum::kLstm) ? 3 : 2;
    if (!param_.state_outputs) {
      out_expected = 1;
    }
    CHECK_EQ(in_data.size(), in_expected);
    CHECK_EQ(out_data.size(), out_expected);
    Stream<cpu> *s = ctx.get_stream<cpu>();
    // get input + output tensor
    Tensor<cpu, 3, DType> x = in_data[rnn_enum::kData].get<cpu, 3, DType>(s);
    Tensor<cpu, 1, DType> w = in_data[rnn_enum::kParams].get<cpu, 1, DType>(s);
    Tensor<cpu, 3, DType> hx = in_data[rnn_enum::kState].get<cpu, 3, DType>(s);
    Tensor<cpu, 3, DType> y = out_data[rnn_enum::kOut].get<cpu, 3, DType>(s);
    CHECK(x.CheckContiguous());
    CHECK(w.CheckContiguous());
    CHECK(hx.CheckContiguous());
    CHECK(y.CheckContiguous());
    param_.seq_length_ = x.shape_[0];
    param_.batch_size_ = x.shape_[1];
    param_.input_size_ = x.shape_[2];
    int dtype = in_data[rnn_enum::kData].type_flag_;

    const int direction = param_.bidirectional ? 2 : 1;
    const int bsize = GetRnnBiasSize(param_.num_layers, param_.state_size, direction, param_.mode);
    DType* b_ptr = w.dptr_ + w.shape_[0] - bsize;

    DType* hy_ptr = NULL;
    if (param_.state_outputs) {
      hy_ptr = out_data[rnn_enum::kStateOut].dptr<DType>();
    }
    DType* cx_ptr = NULL;
    DType* cy_ptr = NULL;

    if (param_.mode == rnn_enum::kLstm) {
      cx_ptr = in_data[rnn_enum::kStateCell].dptr<DType>();
      if (param_.state_outputs) {
        cy_ptr = out_data[rnn_enum::kStateCellOut].dptr<DType>();
      }
    }

    if (ctx.is_train) {
      MKLDNNRNNForwardTraining<DType>(param_.state_outputs,
                                      param_.num_layers,
                                      direction,
                                      param_.seq_length_,
                                      param_.batch_size_,
                                      param_.input_size_,
                                      param_.state_size,
                                      x.dptr_,
                                      hx.dptr_,
                                      cx_ptr,
                                      w.dptr_,
                                      b_ptr,
                                      y.dptr_,
                                      hy_ptr,
                                      cy_ptr,
                                      param_.p,
                                      param_.mode);
    } else {
      MKLDNNRNNForwardInference<DType>(param_.state_outputs,
                                       param_.num_layers,
                                       direction,
                                       param_.seq_length_,
                                       param_.batch_size_,
                                       param_.input_size_,
                                       param_.state_size,
                                       x.dptr_,
                                       hx.dptr_,
                                       cx_ptr,
                                       w.dptr_,
                                       b_ptr,
                                       y.dptr_,
                                       hy_ptr,
                                       cy_ptr,
                                       &weight_bias_memory,
                                       &concat_weight_memory,
                                       &concat_iter_memory,
                                       &dst_memory,
                                       &iter_memory,
                                       dtype,
                                       param_.mode,
                                       &hardcode_count);
    }
  }

  void Backward(const OpContext &ctx,
                const std::vector<TBlob> &out_grad,
                const std::vector<TBlob> &in_data,
                const std::vector<TBlob> &out_data,
                const std::vector<OpReqType> &req,
                const std::vector<TBlob> &in_grad) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK(param_.p >= 0.0f && param_.p < 1.0f)
        << "unsupported dropout value, should be 0 <= dropout < 1";

    size_t in_expected = (param_.mode == rnn_enum::kLstm) ? 4 : 3;
    size_t out_expected = (param_.mode == rnn_enum::kLstm) ? 3 : 2;
    if (!param_.state_outputs) {
      out_expected = 1;
    }
    CHECK_EQ(in_data.size(), in_expected);
    CHECK_EQ(out_data.size(), out_expected);
    CHECK_EQ(in_grad.size(), in_expected);
    CHECK_EQ(out_grad.size(), out_expected);
    CHECK_EQ(req.size(), in_expected);
    CHECK_NE(req[rnn_enum::kData], kAddTo) << "AddTo is not supported for data";
    CHECK_NE(req[rnn_enum::kState], kAddTo) << "AddTo is not supported for state";
    mshadow::Stream<cpu> *s = ctx.get_stream<cpu>();
    // get input + output tensors
    Tensor<cpu, 3, DType> x = in_data[rnn_enum::kData].get<cpu, 3, DType>(s);
    Tensor<cpu, 1, DType> w = in_data[rnn_enum::kParams].get<cpu, 1, DType>(s);
    Tensor<cpu, 3, DType> hx = in_data[rnn_enum::kState].get<cpu, 3, DType>(s);
    Tensor<cpu, 3, DType> y = out_data[rnn_enum::kOut].get<cpu, 3, DType>(s);
    Tensor<cpu, 3, DType> dx = in_grad[rnn_enum::kData].get<cpu, 3, DType>(s);
    Tensor<cpu, 1, DType> dw = in_grad[rnn_enum::kParams].get<cpu, 1, DType>(s);
    Tensor<cpu, 3, DType> dhx = in_grad[rnn_enum::kState].get<cpu, 3, DType>(s);
    Tensor<cpu, 3, DType> dy = out_grad[rnn_enum::kOut].get<cpu, 3, DType>(s);
    CHECK(x.CheckContiguous());
    CHECK(w.CheckContiguous());
    CHECK(hx.CheckContiguous());
    CHECK(y.CheckContiguous());
    CHECK(dx.CheckContiguous());
    CHECK(dw.CheckContiguous());
    CHECK(dhx.CheckContiguous());
    CHECK(dy.CheckContiguous());
    param_.seq_length_ = x.shape_[0];
    param_.batch_size_ = x.shape_[1];
    param_.input_size_ = x.shape_[2];

    const int direction = param_.bidirectional ? 2 : 1;
    const int bsize = GetRnnBiasSize(param_.num_layers, param_.state_size, direction, param_.mode);
    DType* db_ptr = dw.dptr_ + w.shape_[0] - bsize;

    DType * dhy_ptr = NULL;
    if (param_.state_outputs) {
      dhy_ptr = out_grad[rnn_enum::kStateOut].dptr<DType>();
    }

    DType * cx_ptr = NULL;
    DType * dcx_ptr = NULL;
    DType * dcy_ptr = NULL;

    if (param_.mode == rnn_enum::kLstm) {
      CHECK_NE(req[rnn_enum::kStateCell], kAddTo) << "AddTo is not supported for state cell";
      cx_ptr = in_data[rnn_enum::kStateCell].dptr<DType>();
      dcx_ptr = in_grad[rnn_enum::kStateCell].dptr<DType>();
      if (param_.state_outputs) {
        dcy_ptr = out_grad[rnn_enum::kStateCellOut].dptr<DType>();
      }
    }
    MKLDNNRNNBackward<DType>(param_.num_layers,
                             direction,
                             param_.seq_length_,
                             param_.batch_size_,
                             param_.input_size_,
                             param_.state_size,
                             x.dptr_,
                             hx.dptr_,
                             cx_ptr,
                             w.dptr_,
                             y.dptr_,
                             dy.dptr_,
                             dhy_ptr,
                             dcy_ptr,
                             dx.dptr_,
                             dhx.dptr_,
                             dcx_ptr,
                             dw.dptr_,
                             db_ptr,
                             req[rnn_enum::kData],
                             req[rnn_enum::kParams],
                             req[rnn_enum::kState],
                             param_.mode == rnn_enum::kLstm ? req[rnn_enum::kStateCell] : kNullOp,
                             param_.p,
                             param_.mode);
  }

  RNNParam param_;
  std::vector<memory> weight_bias_memory;
  std::vector<memory> concat_weight_memory;
  std::vector<memory> concat_iter_memory;
  std::vector<memory> dst_memory;
  std::vector<memory> iter_memory;
  int hardcode_count;
  bool init_mem_;
  size_t reserve_mem_size_;
  Storage::Handle mem_space_;
};  // class MKLDNNRNNOp

template<typename DType>
static MKLDNNRNNOp<DType> &GetMKLDNNRNNOp(const RNNParam &param,
                                          const NDArray &data,
                                          const NDArray &weight,
                                          const NDArray &out,
                                          const Context& ctx
                                          ) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local std::unordered_map<MKLDNNRNNSignature,
      std::shared_ptr<MKLDNNRNNOp<DType> >, OpHash> ops;
#else
  static MX_THREAD_LOCAL std::unordered_map<MKLDNNRNNSignature,
      std::shared_ptr<MKLDNNRNNOp<DType> >, OpHash> ops;
#endif
  MKLDNNRNNSignature key(param);
  key.AddSign(data);
  key.AddSign(weight);
  key.AddSign(out);
  key.AddSign(ctx.dev_id);

  auto it = ops.find(key);
  if (it == ops.end()) {
    std::shared_ptr<MKLDNNRNNOp<DType>> op(new MKLDNNRNNOp<DType>(param));
    auto ins_ret = ops.insert(std::pair<MKLDNNRNNSignature,
        std::shared_ptr<MKLDNNRNNOp<DType> > >(key, op));
    CHECK(ins_ret.second);
    it = ins_ret.first;
  }
  return *it->second;
}

inline size_t GetMKLDNNRNNCacheMemorySize(int L,
                                           int D,
                                           int T,
                                           int N,
                                           int I,
                                           int H,
                                           int mode) {
  size_t size = 0;
  switch (mode) {
    case rnn_enum::kLstm:
      size = 2 * (D * (I + H) * 4 * H + (L - 1) * D * (D * H + H) * 4 * H + L * D * 2 * N * H) +
          T * N * D * H + L * 2 * D * 4 * H + (L + 2) * D * 2 * N * H + 6 * D * (I + H + 2) * 4 * H;
      break;
    case rnn_enum::kGru:
    case rnn_enum::kRnnRelu:
    case rnn_enum::kRnnTanh:
    default:
      LOG(FATAL) << "unknown RNN mode " << mode;
      break;
  }
  return size;
}

void RNNComputeExCPU(const nnvm::NodeAttrs& attrs,
                     const OpContext &ctx,
                     const std::vector<NDArray> &inputs,
                     const std::vector<OpReqType> &req,
                     const std::vector<NDArray> &outputs) {
  RNNParam& param = (RNNParam&)nnvm::get<RNNParam>(attrs.parsed);
  std::vector<TBlob> in_blobs;
  std::vector<TBlob> out_blobs;
  std::vector<NDArray> temp_ndarrays_i;
  std::vector<NDArray> temp_ndarrays_o;
  for (const NDArray& in : inputs) {
    if (in.storage_type() == kDefaultStorage) {
      temp_ndarrays_i.push_back(in.Reorder2Default());
      in_blobs.emplace_back(temp_ndarrays_i.back().data());
    } else {
      in_blobs.emplace_back(in.data());
    }
  }

  for (const NDArray& out : outputs) {
    if (out.storage_type() == kDefaultStorage) {
      temp_ndarrays_o.push_back(out.Reorder2Default());
      out_blobs.emplace_back(temp_ndarrays_o.back().data());
    } else {
      out_blobs.emplace_back(out.data());
    }
  }
  int dtype = in_blobs[rnn_enum::kData].type_flag_;
  memory::data_type mkldnn_dtype = get_mkldnn_type(dtype);
  Stream<cpu> *s = ctx.get_stream<cpu>();
  auto cpu_engine = CpuEngine::Get()->get_engine();
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    MKLDNNRNNOp<DType> &RNNFwd =
        GetMKLDNNRNNOp<DType>(param, inputs[rnn_enum::kData], inputs[rnn_enum::kParams],
        outputs[rnn_enum::kOut], ctx.run_ctx.ctx);

    int ngates = 0, nstates = 0;
    GetMKLDNNAlgo(RNNFwd.param_.mode, &ngates, &nstates);
    int D = RNNFwd.param_.bidirectional ? 2 : 1;
    Tensor<cpu, 3, DType> x = in_blobs[rnn_enum::kData].get<cpu, 3, DType>(s);
    int T = x.shape_[0];
    int N = x.shape_[1];
    int I = x.shape_[2];
    int H = RNNFwd.param_.state_size;
    int L = RNNFwd.param_.num_layers;
    const size_t r_size = GetMKLDNNRNNCacheMemorySize(L, D, T, N, I, H, RNNFwd.param_.mode);
    if (RNNFwd.init_mem_ && RNNFwd.reserve_mem_size_ < r_size) {
      Storage::Get()->Free(RNNFwd.mem_space_);
      RNNFwd.init_mem_ = false;
    }

    if (!RNNFwd.init_mem_) {
      RNNFwd.mem_space_ = Storage::Get()->Alloc(r_size * sizeof(DType), Context::CPU());
      RNNFwd.reserve_mem_size_ = r_size;
      RNNFwd.init_mem_ = true;
    }
    DType* workptr = static_cast<DType*>(RNNFwd.mem_space_.dptr);
    memory::dims dst_layer_tz = {T, N, D * H};
    auto dst_layer_md = memory::desc(
      { dst_layer_tz }, mkldnn_dtype, memory::format::tnc);
    if (RNNFwd.weight_bias_memory.size() == 0) {
      RNNFwd.hardcode_count = 0;
      if (D == 1 && I == H && L > 1) {
        memory::dims weights_layer_tz = {L, 1, I, ngates, H};  //  ldigo
        memory::dims weights_iter_tz = {L, 1, H, ngates, H};  //  ldigo
        memory::dims bias_tz = {L, 1, ngates, H};
        auto user_weight_layer_md = memory::desc(
            { weights_layer_tz }, mkldnn_dtype, memory::format::ldigo);
        auto user_weight_iter_md = memory::desc(
            { weights_iter_tz }, mkldnn_dtype, memory::format::ldigo);
        auto user_bias_md = memory::desc({ bias_tz },
            mkldnn_dtype, memory::format::ldgo);
        DType* weight_layer_n = workptr;  //  L * I * ngates * H
        auto user_weight_layer_memory_n
            = memory({ user_weight_layer_md, cpu_engine }, weight_layer_n);
        RNNFwd.weight_bias_memory.push_back(user_weight_layer_memory_n);

        DType* weight_iter_n = weight_layer_n + L * I * ngates * H;  //  L * H * ngates * H
        auto user_weight_iter_memory_n
            = memory({ user_weight_iter_md, cpu_engine }, weight_iter_n);
        RNNFwd.weight_bias_memory.push_back(user_weight_iter_memory_n);

        DType* bias_n = weight_iter_n + L * H * ngates * H;  //  L * ngates * H
        auto user_bias_memory_n =
            memory({ user_bias_md, cpu_engine }, bias_n);
        RNNFwd.weight_bias_memory.push_back(user_bias_memory_n);

        auto wx_md_n = memory::desc(
            { weights_layer_tz }, mkldnn_dtype, memory::format::ldgoi);
        DType* wx_n = bias_n + L * ngates * H;  //   L * ngates * I * H
        auto wx_memory_n =
            memory({ wx_md_n, cpu_engine }, wx_n);
        DType* wh_n = wx_n + L * ngates * I * H;  //  L * ngates * H * H
        auto wh_md_n = memory::desc(
            { weights_iter_tz }, mkldnn_dtype, memory::format::ldgoi);
        auto wh_memory_n =
            memory({ wh_md_n, cpu_engine }, wh_n);

        RNNFwd.concat_weight_memory.push_back(wx_memory_n);
        RNNFwd.concat_weight_memory.push_back(wh_memory_n);
        workptr = wh_n + L * ngates * H * H;

        memory::dims src_iter_tz_n1 = {1, 1, nstates, N, H};  //  ldsnc
        auto src_iter_md_n1 = memory::desc(
            { src_iter_tz_n1 }, mkldnn_dtype, memory::format::ldsnc);
        for (int l = 0; l < L; l++) {
          DType* src_iter_n1 = workptr;  //  nstates * N * H
          auto src_iter_memory_n1 =
              memory({ src_iter_md_n1, cpu_engine }, src_iter_n1);
          RNNFwd.concat_iter_memory.push_back(src_iter_memory_n1);
          workptr = src_iter_n1 + nstates * N * H;
        }
        memory::dims src_iter_tz_n = {L, 1, nstates, N, H};  //  ldsnc
        auto src_iter_md_n = memory::desc(
            { src_iter_tz_n }, mkldnn_dtype, memory::format::ldsnc);
        DType* src_iter_n = workptr;  //  L * nstates * N * H
        auto src_iter_memory_n =
            memory({ src_iter_md_n, cpu_engine }, src_iter_n);
        RNNFwd.concat_iter_memory.push_back(src_iter_memory_n);

        DType* dst_layer_n = src_iter_n + L * nstates * N * H;  //  T * N * D * H
        auto dst_layer_memory_n
            = memory({ dst_layer_md, cpu_engine }, dst_layer_n);
        RNNFwd.dst_memory.push_back(dst_layer_memory_n);

        memory::dims dst_iter_tz_n = {L, 1, nstates, N, H};  //  ldsnc
        auto dst_iter_md_n = memory::desc(
            { dst_iter_tz_n }, mkldnn_dtype, memory::format::ldsnc);
        DType* dst_iter_n = dst_layer_n + T * N * D * H;  //  L * nstates * N * H
        auto dst_iter_memory_n =
            memory({ dst_iter_md_n, cpu_engine }, dst_iter_n);
        RNNFwd.iter_memory.push_back(dst_iter_memory_n);
      } else {
        memory::dims weights_layer_tz_0 = {1, D, I, ngates, H};  //  ldigo
        memory::dims weights_iter_tz_0 = {1, D, H, ngates, H};  //  ldigo
        memory::dims bias_tz_0 = {1, D, ngates, H};

        auto user_weight_layer_md_0 = memory::desc(
            { weights_layer_tz_0 }, mkldnn_dtype, memory::format::ldigo);
        auto user_weight_iter_md_0 = memory::desc(
            { weights_iter_tz_0 }, mkldnn_dtype, memory::format::ldigo);
        auto user_bias_md_0 = memory::desc({ bias_tz_0 },
            mkldnn_dtype, memory::format::ldgo);

        DType* weight_layer_0 = workptr;  //  D * I * ngates * H
        auto user_weight_layer_memory_0
            = memory({ user_weight_layer_md_0, cpu_engine }, weight_layer_0);
        RNNFwd.weight_bias_memory.push_back(user_weight_layer_memory_0);

        DType* weight_iter_0 = weight_layer_0 + D * I * ngates * H;  //  D * H * ngates * H
        auto user_weight_iter_memory_0
            = memory({ user_weight_iter_md_0, cpu_engine }, weight_iter_0);
        RNNFwd.weight_bias_memory.push_back(user_weight_iter_memory_0);

        DType* bias_0 = weight_iter_0 + D * H * ngates * H;  //  D * ngates * H
        auto user_bias_memory_0 =
            memory({ user_bias_md_0, cpu_engine }, bias_0);
        RNNFwd.weight_bias_memory.push_back(user_bias_memory_0);
        workptr = bias_0 + D * ngates * H;

        auto wx_md_0 = memory::desc(
            { weights_layer_tz_0 }, mkldnn_dtype, memory::format::ldgoi);
        auto wx_memory_0 =
            memory({ wx_md_0, cpu_engine });
        auto wh_md_0 = memory::desc(
            { weights_iter_tz_0 }, mkldnn_dtype, memory::format::ldgoi);
        auto wh_memory_0 =
            memory({ wh_md_0, cpu_engine });
        if (D == 2) {
          DType* wx_0 = workptr;  //  D * ngates * I * H
          wx_memory_0.set_data_handle(wx_0);
          DType* wh_0 = wx_0 + D * ngates * I * H;  //  D * ngates * H * H
          wh_memory_0.set_data_handle(wh_0);
          workptr = wh_0 + D * ngates * H * H;
        }
        RNNFwd.concat_weight_memory.push_back(wx_memory_0);
        RNNFwd.concat_weight_memory.push_back(wh_memory_0);

        memory::dims src_iter_undi_tz_0 = {1, 1, nstates, N, H};  //  ldsnc
        auto src_iter_undi_md_0 = memory::desc(
            { src_iter_undi_tz_0 }, mkldnn_dtype, memory::format::ldsnc);
        DType* src_iter_undi_0 = workptr;  //  nstates * N * H
        auto src_iter_undi_memory_0 =
            memory({ src_iter_undi_md_0, cpu_engine }, src_iter_undi_0);
        RNNFwd.concat_iter_memory.push_back(src_iter_undi_memory_0);
        workptr = src_iter_undi_0 + nstates * N * H;

        if (D == 2) {
          DType* src_iter_undi2_0 = workptr;  //  nstates * N * H
          auto src_iter_undi2_memory_0 =
              memory({ src_iter_undi_md_0, cpu_engine }, src_iter_undi2_0);
          RNNFwd.concat_iter_memory.push_back(src_iter_undi2_memory_0);

          memory::dims src_iter_tz_0 = {1, D, nstates, N, H};  //  ldsnc
          auto src_iter_md_0 = memory::desc(
              { src_iter_tz_0 }, mkldnn_dtype, memory::format::ldsnc);
          DType* src_iter_0 = src_iter_undi2_0 + nstates * N * H;  //  D * nstates * N * H
          auto src_iter_memory_0 =
              memory({ src_iter_md_0, cpu_engine }, src_iter_0);
          RNNFwd.concat_iter_memory.push_back(src_iter_memory_0);
          workptr = src_iter_0 + D * nstates * N * H;
        }

        DType* dst_layer_0 = workptr;  //  T * N * D * H
        auto dst_layer_memory_0
            = memory({ dst_layer_md, cpu_engine }, dst_layer_0);
        RNNFwd.dst_memory.push_back(dst_layer_memory_0);

        memory::dims dst_iter_tz_0 = {1, D, nstates, N, H};  //  ldsnc
        auto dst_iter_md_0 = memory::desc(
            { dst_iter_tz_0 }, mkldnn_dtype, memory::format::ldsnc);
        DType* dst_iter_0 = dst_layer_0 + T * N * D * H;  //  D * nstates * N * H
        auto dst_iter_memory_0 =
            memory({ dst_iter_md_0, cpu_engine }, dst_iter_0);
        RNNFwd.iter_memory.push_back(dst_iter_memory_0);
        workptr = dst_iter_0 + D * nstates * N * H;
        //  hard code for sockeye size testing
        if (L == 1 && D == 1 && T == 1 && N == 640 && I == 1024 && H ==512) {
          for (int i = 0; i < 6; i++) {
            DType* weight_layer_i = workptr;  //  D * I * ngates * H
            auto user_weight_layer_memory_i
                = memory({ user_weight_layer_md_0, cpu_engine }, weight_layer_i);
            RNNFwd.weight_bias_memory.push_back(user_weight_layer_memory_i);

            DType* weight_iter_i = weight_layer_i + D * I * ngates * H;  //  D * H * ngates * H
            auto user_weight_iter_memory_i
                = memory({ user_weight_iter_md_0, cpu_engine }, weight_iter_i);
            RNNFwd.weight_bias_memory.push_back(user_weight_iter_memory_i);

            DType* bias_i = weight_iter_i + D * H * ngates * H;  //  D * ngates * H
            auto user_bias_memory_i =
                memory({ user_bias_md_0, cpu_engine }, bias_i);
            RNNFwd.weight_bias_memory.push_back(user_bias_memory_i);
            workptr = bias_i + D * ngates * H;
          }
        }
        //  hard code for sockeye size

        //  next L - 1 layers
        if (L > 1 && D == 1) {
          memory::dims weights_layer_tz = {L - 1, 1, H, ngates, H};  //  ldigo
          memory::dims weights_iter_tz = {L - 1, 1, H, ngates, H};  //  ldigo
          memory::dims bias_tz = {L - 1, 1, ngates, H};
          auto user_weight_layer_md = memory::desc(
              { weights_layer_tz }, mkldnn_dtype, memory::format::ldigo);
          auto user_weight_iter_md = memory::desc(
              { weights_iter_tz }, mkldnn_dtype, memory::format::ldigo);
          auto user_bias_md = memory::desc({ bias_tz },
              mkldnn_dtype, memory::format::ldgo);

          DType* weight_layer_n = workptr;  //  (L - 1) * H * ngates * H
          auto user_weight_layer_memory_n
              = memory({ user_weight_layer_md, cpu_engine }, weight_layer_n);
          RNNFwd.weight_bias_memory.push_back(user_weight_layer_memory_n);

          DType* weight_iter_n = weight_layer_n +
              (L - 1) * H * ngates * H;  //  (L - 1) * H * ngates * H
          auto user_weight_iter_memory_n
              = memory({ user_weight_iter_md, cpu_engine }, weight_iter_n);
          RNNFwd.weight_bias_memory.push_back(user_weight_iter_memory_n);

          DType* bias_n = weight_iter_n + (L - 1) * H * ngates * H;  //  (L - 1) * ngates * H
          auto user_bias_memory_n =
              memory({ user_bias_md, cpu_engine }, bias_n);
          RNNFwd.weight_bias_memory.push_back(user_bias_memory_n);

          auto wx_md_n = memory::desc(
              { weights_layer_tz }, mkldnn_dtype, memory::format::ldgoi);
          DType* wx_n = bias_n + (L - 1) * ngates * H;  //  (L - 1) * ngates * H * H
          auto wx_memory_n =
              memory({ wx_md_n, cpu_engine }, wx_n);
          DType* wh_n = wx_n + (L - 1) * ngates * H * H;  //  (L - 1) * ngates * H * H
          auto wh_md_n = memory::desc(
              { weights_iter_tz }, mkldnn_dtype, memory::format::ldgoi);
          auto wh_memory_n =
              memory({ wh_md_n, cpu_engine }, wh_n);

          RNNFwd.concat_weight_memory.push_back(wx_memory_n);
          RNNFwd.concat_weight_memory.push_back(wh_memory_n);
          workptr = wh_n + (L - 1) * ngates * H * H;

          memory::dims src_iter_tz_n1 = {1, 1, nstates, N, H};  //  ldsnc
          auto src_iter_md_n1 = memory::desc(
              { src_iter_tz_n1 }, mkldnn_dtype, memory::format::ldsnc);
          for (int l = 0; l < L - 1; l++) {
            DType* src_iter_n1 = workptr;  //  nstates * N * H
            auto src_iter_memory_n1 =
                memory({ src_iter_md_n1, cpu_engine }, src_iter_n1);
            RNNFwd.concat_iter_memory.push_back(src_iter_memory_n1);
            workptr = src_iter_n1 + nstates * N * H;
          }
          memory::dims src_iter_tz_n = {L - 1, 1, nstates, N, H};  //  ldsnc
          auto src_iter_md_n = memory::desc(
              { src_iter_tz_n }, mkldnn_dtype, memory::format::ldsnc);
          DType* src_iter_n = workptr;  //  (L - 1) * nstates * N * H
          auto src_iter_memory_n =
              memory({ src_iter_md_n, cpu_engine }, src_iter_n);
          RNNFwd.concat_iter_memory.push_back(src_iter_memory_n);

          DType* dst_layer_n = src_iter_n + (L - 1) * nstates * N * H;  //  T * N * D * H
          auto dst_layer_memory_n
              = memory({ dst_layer_md, cpu_engine }, dst_layer_n);
          RNNFwd.dst_memory.push_back(dst_layer_memory_n);

          memory::dims dst_iter_tz_n = {L - 1, 1, nstates, N, H};  //  ldsnc
          auto dst_iter_md_n = memory::desc(
              { dst_iter_tz_n }, mkldnn_dtype, memory::format::ldsnc);
          DType* dst_iter_n = dst_layer_n + T * N * D * H;  //  (L - 1) * nstates * N * H
          auto dst_iter_memory_n =
              memory({ dst_iter_md_n, cpu_engine }, dst_iter_n);
          RNNFwd.iter_memory.push_back(dst_iter_memory_n);
        }

        if (L > 1 && D == 2) {
          memory::dims weights_layer_tz = {1, D, H * D, ngates, H};  //  ldigo
          memory::dims weights_iter_tz = {1, D, H, ngates, H};  //  ldigo
          memory::dims bias_tz = {1, D, ngates, H};
          auto user_weight_layer_md = memory::desc(
              { weights_layer_tz }, mkldnn_dtype, memory::format::ldigo);
          auto user_weight_iter_md = memory::desc(
              { weights_iter_tz }, mkldnn_dtype, memory::format::ldigo);
          auto user_bias_md = memory::desc({ bias_tz },
              mkldnn_dtype, memory::format::ldgo);

          auto wx_md_n = memory::desc(
              { weights_layer_tz }, mkldnn_dtype, memory::format::ldgoi);
          auto wh_md_n = memory::desc(
              { weights_iter_tz }, mkldnn_dtype, memory::format::ldgoi);

          for (int l = 0; l < L - 1; l++) {
            DType* weight_layer_n = workptr;  //  D * (H * D) * ngates * H
            auto user_weight_layer_memory_n
                = memory({ user_weight_layer_md, cpu_engine }, weight_layer_n);
            RNNFwd.weight_bias_memory.push_back(user_weight_layer_memory_n);

            DType* weight_iter_n = weight_layer_n +
                D * (H * D) * ngates * H;  //  D * H * ngates * H
            auto user_weight_iter_memory_n
                = memory({ user_weight_iter_md, cpu_engine }, weight_iter_n);
            RNNFwd.weight_bias_memory.push_back(user_weight_iter_memory_n);

            DType* bias_n = weight_iter_n + D * H * ngates * H;  //  D * ngates * H
            auto user_bias_memory_n =
                memory({ user_bias_md, cpu_engine }, bias_n);
            RNNFwd.weight_bias_memory.push_back(user_bias_memory_n);

            DType* wx_n = bias_n + D * ngates * H;  //  D * ngates * (D * H) * H
            DType* wh_n = wx_n + D * ngates * (D * H) * H;  //  D * ngates * H * H
            auto wx_memory_n =
                memory({ wx_md_n, cpu_engine }, wx_n);
            auto wh_memory_n =
                memory({ wh_md_n, cpu_engine }, wh_n);
            RNNFwd.concat_weight_memory.push_back(wx_memory_n);
            RNNFwd.concat_weight_memory.push_back(wh_memory_n);
            // reuse layer 1's concat_iter_memory, no need to create more

            DType* dst_layer_n = wh_n + D * ngates * H * H;  //  T * N * D * H
            auto dst_layer_memory_n
                = memory({ dst_layer_md, cpu_engine }, dst_layer_n);
            RNNFwd.dst_memory.push_back(dst_layer_memory_n);

            memory::dims dst_iter_tz_n = {1, D, nstates, N, H};  //  ldsnc
            auto dst_iter_md_n = memory::desc(
                { dst_iter_tz_n }, mkldnn_dtype, memory::format::ldsnc);
            DType* dst_iter_n = dst_layer_n + T * N * D * H;  //  D * nstates * N * H
            auto dst_iter_memory_n =
                memory({ dst_iter_md_n, cpu_engine }, dst_iter_n);
            RNNFwd.iter_memory.push_back(dst_iter_memory_n);
            workptr = dst_iter_n + D * nstates * N * H;
          }
        }
      }
    }
    RNNFwd.Forward(ctx, in_blobs, req, out_blobs);
  });
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_USE_MKLDNN == 1
#endif  // MXNET_OPERATOR_NN_MKLDNN_MKLDNN_RNN_IMPL_H_
