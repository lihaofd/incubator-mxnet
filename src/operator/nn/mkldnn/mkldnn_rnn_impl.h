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
#include <vector>
#include "../../rnn_impl.h"
#include "mkldnn.hpp"
#include "./mkldnn_base-inl.h"

namespace mxnet {
namespace op {

namespace mkldnn_rnn_enum {
  enum RNNModeType {kRnnRelu, kRnnTanh, kLstm, kGru};
}

template <typename DType>
mkldnn::memory::data_type GetMKLDNNDataType() {
  mkldnn::memory::data_type datatype;
  switch (typeid(DType).name()[0]) {
    case 'f':
    case 'd':
      datatype = mkldnn::memory::data_type::f32;
      break;
    case 'i':
      datatype = mkldnn::memory::data_type::s32;
      break;
    case 's':
      datatype = mkldnn::memory::data_type::s16;
      break;
    case 'a':
      datatype = mkldnn::memory::data_type::s8;
      break;
    case 'h':
      datatype = mkldnn::memory::data_type::u8;
      break;
    default:
      datatype = mkldnn::memory::data_type::data_undef;
      break;
  }
  return datatype;
}

mkldnn::algorithm GetMKLDNNAlgo(int mode,
                                int* ngates,
                                int* nstates) {
  mkldnn::algorithm algo;
  switch (mode) {
    case mkldnn_rnn_enum::kLstm:
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

void ReorderForWeight(mkldnn::memory src,
                      mkldnn::memory dst) {
  auto r = mkldnn::reorder(src, dst);
  stream(stream::kind::eager).submit({r}).wait();
}

template <typename DType>
mkldnn::memory ConcatData(mkldnn::memory::format src_format,
                          mkldnn::memory::format dst_format,
                          std::vector<mkldnn::memory::dims> srcs_cds,
                          mkldnn::memory::dims dst_cds,
                          mkldnn::memory::data_type mkldnn_dtype,
                          int concat_dimension,
                          std::vector<DType*> srcs_data) {
  auto cpu_engine = CpuEngine::Get()->get_engine();
  std::vector<mkldnn::memory::primitive_desc> srcs_pd;
  std::vector<mkldnn::memory> srcs;
  for (size_t i = 0; i < srcs_cds.size(); i++) {
    auto desc = memory::desc(srcs_cds[i], mkldnn_dtype, src_format);
    auto mpd = memory::primitive_desc(desc, cpu_engine);
    auto src_memory = mkldnn::memory({desc, cpu_engine}, srcs_data[i]);
    srcs_pd.push_back(mpd);
    srcs.push_back(src_memory);
  }
  std::vector<primitive::at> inputs;
  for (size_t i = 0; i < srcs_cds.size(); i++) {
    inputs.push_back(srcs[i]);
  }

  auto dst_desc = mkldnn::memory::desc(dst_cds, mkldnn_dtype, dst_format);
  auto concat_pd = mkldnn::concat::primitive_desc(dst_desc, concat_dimension, srcs_pd);
  auto dst = mkldnn::memory(concat_pd.dst_primitive_desc());
  
  auto c = mkldnn::concat(concat_pd, inputs, dst);
  stream(stream::kind::eager).submit({c}).wait();
  return dst;
}

template <typename DType>
void MKLDNNRNNForward(bool state_outputs,
                      int L,
                      int D,
                      const int T,
                      const int N,
                      int I,
                      const int H,
                      DType* x_ptr,
                      DType* hx_ptr,
                      DType* cx_ptr,
                      DType* w_ptr,
                      DType* b_ptr,
                      DType* y_ptr,
                      DType* hy_ptr,
                      DType* cy_ptr,
                      std::vector<mkldnn::memory> *weight_bias_memory,
                      int mode) {
  int ngates = 0, nstates = 0;
  mkldnn::algorithm nalgorithm = GetMKLDNNAlgo(mode, &ngates, &nstates);
  mkldnn::memory::data_type mkldnn_dtype = GetMKLDNNDataType<DType>();
  const int b_size = 2 * H * ngates * D;
  const int cell_size = N * H * D;
  const int single_cell_size = N * H;
  const int single_b_size = ngates * H;
  //  First layer
  int w_size = (I + H) * H * ngates * D;
  DType* x_0 = x_ptr;  //  T * N * I
  DType* y = y_ptr;  //  T * N * D * H
  DType* back_w_ptr = w_ptr;
  DType* back_b_ptr = b_ptr;
  DType* wx_0 = w_ptr;  //  ngates * H, I
  DType* wh_0 = w_ptr + I * H * ngates;  //  ngates * H, H
  if (D == 2) {
    back_w_ptr = w_ptr + ngates * H * (I + H);
    back_b_ptr = b_ptr + single_b_size * 2;
  }
  DType* back_wx_0 = back_w_ptr;
  DType* back_wh_0 = back_w_ptr + I * H * ngates;
  DType* bx_0 = b_ptr;
  DType* bh_0 = b_ptr + H * ngates;
  DType* back_bx_0 = back_b_ptr;
  DType* back_bh_0 = back_b_ptr + H * ngates;
  const int omp_threads = mxnet::engine::OpenMP::Get()->GetRecommendedOMPThreadCount();
  using namespace mkldnn;
  auto cpu_engine = CpuEngine::Get()->get_engine();
  auto null_memory_ = null_memory(cpu_engine);
  std::vector<primitive> rnn_net_0;
  memory::dims src_layer_tz_0 = {T, N, I};
  memory::dims weights_layer_tz_0 = {1, D, I, ngates, H};  //  ldigo
  memory::dims weights_layer_r_tz_0 = {1, 1, I, ngates, H};  //  ldigo for reorder
  memory::dims weights_iter_tz_0 = {1, D, H, ngates, H};  //  ldigo
  memory::dims weights_iter_r_tz_0 = {1, 1, H, ngates, H};  //  ldigo for reorder
  memory::dims bias_tz_0 = {1, D, ngates, H};
  memory::dims dst_layer_tz_0 = {T, N, D * H};
  memory::dims src_iter_tz_0 = {1, D, nstates, N, H};  //  ldsnc
  memory::dims dst_iter_tz_0 = {1, D, nstates, N, H};  //  ldsnc
  int offset1 = 0, offset2 = 0;
  auto user_weight_layer_md_0 = mkldnn::memory::desc(
      { weights_layer_tz_0 }, mkldnn_dtype, mkldnn::memory::format::ldigo);
  auto user_weight_iter_md_0 = mkldnn::memory::desc(
      { weights_iter_tz_0 }, mkldnn_dtype, mkldnn::memory::format::ldigo);

  auto src_wx0 = null_memory_, src_wh0 = null_memory_;
  if (D == 1) {
    auto mpd_x = mkldnn::memory::primitive_desc({weights_layer_r_tz_0,
        mkldnn_dtype, mkldnn::memory::format::ldgoi}, cpu_engine);
    src_wx0 = mkldnn::memory(mpd_x, wx_0);
  } else {
    std::vector<DType*> srcs_data1;
    srcs_data1.push_back(wx_0);
    srcs_data1.push_back(back_wx_0);
    src_wx0 =
        ConcatData(mkldnn::memory::format::ldgoi, mkldnn::memory::format::ldgoi,
            {weights_layer_r_tz_0, weights_layer_r_tz_0}, weights_layer_tz_0,
            mkldnn_dtype, 1, srcs_data1);
  }
  if (D == 1) {
    auto mpd_h = mkldnn::memory::primitive_desc({weights_iter_r_tz_0,
      mkldnn_dtype, mkldnn::memory::format::ldgoi}, cpu_engine);
    src_wh0 = mkldnn::memory(mpd_h, wh_0);
  } else {
    std::vector<DType*> srcs_data1;
    srcs_data1.push_back(wh_0);
    srcs_data1.push_back(back_wh_0);
    src_wh0 =
        ConcatData(mkldnn::memory::format::ldgoi, mkldnn::memory::format::ldgoi,
            {weights_iter_r_tz_0, weights_iter_r_tz_0}, weights_iter_tz_0,
             mkldnn_dtype, 1, srcs_data1);
  }

  ReorderForWeight(src_wx0, (*weight_bias_memory)[0]);
  ReorderForWeight(src_wh0, (*weight_bias_memory)[1]);

  DType* user_bias_0 = reinterpret_cast<DType *> ((*weight_bias_memory)[2].get_data_handle());
  #pragma omp parallel for num_threads(omp_threads)
  for (int j = 0; j < single_b_size; j++) {
    user_bias_0[j] = bx_0[j] + bh_0[j];
  }
  if (D == 2) {
    #pragma omp parallel for num_threads(omp_threads)
    for (int j = 0; j < single_b_size; j++) {
      user_bias_0[single_b_size + j] = back_bx_0[j] + back_bh_0[j];
    }
  }

  auto user_src_layer_md_0 = mkldnn::memory::desc(
      { src_layer_tz_0 }, mkldnn_dtype, mkldnn::memory::format::tnc);
  auto user_bias_md_0 = mkldnn::memory::desc({ bias_tz_0 },
      mkldnn_dtype, mkldnn::memory::format::ldgo);
  auto dst_layer_md_0 = mkldnn::memory::desc(
      { dst_layer_tz_0 }, mkldnn_dtype, mkldnn::memory::format::tnc);
  auto dst_iter_md_0 = mkldnn::memory::desc(
      { dst_iter_tz_0 }, mkldnn_dtype, mkldnn::memory::format::ldsnc);

  auto user_src_layer_memory_0 = mkldnn::memory(
      { user_src_layer_md_0, cpu_engine }, x_0);

  auto user_src_iter_md_0 = mkldnn::memory::desc(
      { src_iter_tz_0 }, mkldnn_dtype, mkldnn::memory::format::ldsnc);
  auto user_src_iter_memory_0 = null_memory_;

  if (mode == mkldnn_rnn_enum::kLstm) {
    std::vector<DType*> srcs_data1;
    srcs_data1.push_back(hx_ptr);
    srcs_data1.push_back(cx_ptr);
    auto tmp1_src_iter_memory_0 =
        ConcatData(mkldnn::memory::format::ldsnc, mkldnn::memory::format::ldsnc,
        {{1, 1, 1, N, H}, {1, 1, 1, N, H}}, {1, 1, nstates, N, H}, mkldnn_dtype, 2, srcs_data1);

    if (D == 2) {
      std::vector<DType*> srcs_data2;
      srcs_data2.push_back(hx_ptr + single_cell_size);
      srcs_data2.push_back(cx_ptr + single_cell_size);
      auto tmp2_src_iter_memory_0 =
          ConcatData(mkldnn::memory::format::ldsnc, mkldnn::memory::format::ldsnc,
          {{1, 1, 1, N, H}, {1, 1, 1, N, H}}, {1, 1, nstates, N, H}, mkldnn_dtype, 2,
          srcs_data2);
      std::vector<DType*> srcs_data3;
      srcs_data3.push_back(reinterpret_cast<DType *>(tmp1_src_iter_memory_0.get_data_handle()));
      srcs_data3.push_back(reinterpret_cast<DType *>(tmp2_src_iter_memory_0.get_data_handle()));
      user_src_iter_memory_0 =
          ConcatData(mkldnn::memory::format::ldsnc, mkldnn::memory::format::ldsnc,
          {{1, 1, nstates, N, H}, {1, 1, nstates, N, H}}, {1, D, nstates, N, H},
          mkldnn_dtype, 1, srcs_data3);
    } else {
      user_src_iter_memory_0 = tmp1_src_iter_memory_0;
    }
  }

  rnn_cell::desc cell_0(nalgorithm);
  rnn_forward::desc layer_desc_0(prop_kind::forward_inference, cell_0,
      D == 1 ? rnn_direction::unidirectional : rnn_direction::bidirectional_concat,
      user_src_layer_md_0, user_src_iter_md_0, user_weight_layer_md_0, user_weight_iter_md_0,
      user_bias_md_0, dst_layer_md_0, dst_iter_md_0);

  auto prim_desc_0 = mkldnn::rnn_forward::primitive_desc(layer_desc_0, cpu_engine);
  auto dst_layer_memory_0 = mkldnn::memory(prim_desc_0.dst_layer_primitive_desc());
  auto dst_iter_memory_0 = mkldnn::memory(prim_desc_0.dst_iter_primitive_desc());
  dst_layer_memory_0.set_data_handle(y);

  rnn_net_0.push_back(
      rnn_forward(prim_desc_0, user_src_layer_memory_0,
                  user_src_iter_memory_0, (*weight_bias_memory)[0],
                  (*weight_bias_memory)[1], (*weight_bias_memory)[2],
                  dst_layer_memory_0, dst_iter_memory_0, null_memory_));
  stream(stream::kind::eager).submit(rnn_net_0).wait();

  auto user_src_layer_memory_l = dst_layer_memory_0;
  if (state_outputs) {
    DType* dst_hcy_0 = reinterpret_cast<DType *> (dst_iter_memory_0.get_data_handle());
    #pragma omp parallel for num_threads(omp_threads)
    for (int n = 0; n < single_cell_size; n++) {
      hy_ptr[n] = dst_hcy_0[n];
    }
    if (mode == mkldnn_rnn_enum::kLstm) {
      #pragma omp parallel for num_threads(omp_threads)
      for (int n = 0; n < single_cell_size; n++) {
        cy_ptr[n] = dst_hcy_0[n + single_cell_size];
      }
    }
    if (D == 2) {
      offset2 = nstates * single_cell_size;
      #pragma omp parallel for num_threads(omp_threads)
      for (int n = 0; n < single_cell_size; n++) {
        hy_ptr[n + single_cell_size] = dst_hcy_0[n + offset2];
      }
      if (mode == mkldnn_rnn_enum::kLstm) {
        offset2 = (nstates + 1) * single_cell_size;
        #pragma omp parallel for num_threads(omp_threads)
        for (int n = 0; n < single_cell_size; n++) {
          cy_ptr[n + single_cell_size] = dst_hcy_0[n + offset2];
        }
      }
    }
  }
  //  go to next L - 1 layers.
  //  If D = 2, do it layer by layer. If D = 1, fused L - 1 layers
  memory::dims dst_layer_tz = {T, N, H * D};
  memory::dims src_layer_tz = {T, N, H * D};  //  I has already been reset as H * D
  memory::dims weights_layer_r_tz = {1, 1, H * D, ngates, H};  //  ldigo for reorder
  memory::dims weights_iter_r_tz = {1, 1, H, ngates, H};  //  ldigo for reorder
  auto user_src_layer_md = mkldnn::memory::desc(
      { src_layer_tz }, mkldnn_dtype, mkldnn::memory::format::tnc);
  auto dst_layer_md = mkldnn::memory::desc(
      { dst_layer_tz }, mkldnn_dtype, mkldnn::memory::format::tnc);
  if (L > 1 && D == 2) {
    w_ptr += w_size;
    b_ptr += b_size;
    w_size = (H * D + H) * H * ngates * D;
    memory::dims weights_layer_tz = {1, D, H * D, ngates, H};  //  ldigo
    memory::dims weights_iter_tz = {1, D, H, ngates, H};  //  ldigo
    memory::dims bias_tz = {1, D, ngates, H};
    memory::dims src_iter_tz = {1, D, nstates, N, H};  //  ldsnc
    memory::dims dst_iter_tz = {1, D, nstates, N, H};  //  ldsnc
    auto user_src_iter_md = mkldnn::memory::desc(
        { src_iter_tz }, mkldnn_dtype, mkldnn::memory::format::ldsnc);
    auto user_weight_layer_md = mkldnn::memory::desc(
        { weights_layer_tz }, mkldnn_dtype, mkldnn::memory::format::ldigo);
    auto user_weight_iter_md = mkldnn::memory::desc(
        { weights_iter_tz }, mkldnn_dtype, mkldnn::memory::format::ldigo);
    auto user_bias_md = mkldnn::memory::desc({ bias_tz },
        mkldnn_dtype, mkldnn::memory::format::ldgo);
    auto dst_iter_md = mkldnn::memory::desc(
        { dst_iter_tz }, mkldnn_dtype, mkldnn::memory::format::ldsnc);
    rnn_cell::desc cell(nalgorithm);
    rnn_forward::desc layer_desc(prop_kind::forward_inference, cell,
        rnn_direction::bidirectional_concat,
        user_src_layer_md, user_src_iter_md, user_weight_layer_md, user_weight_iter_md,
        user_bias_md, dst_layer_md, dst_iter_md);
    auto prim_desc = mkldnn::rnn_forward::primitive_desc(layer_desc, cpu_engine);
    auto dst_layer_memory = mkldnn::memory(prim_desc.dst_layer_primitive_desc());
    dst_layer_memory.set_data_handle(y);
    auto dst_iter_memory = mkldnn::memory(prim_desc.dst_iter_primitive_desc());
    for (int l = 0; l < L - 1; l++) {
      std::vector<primitive> rnn_net2;
      if (state_outputs) {
        hy_ptr += cell_size;
        if (mode == mkldnn_rnn_enum::kLstm) {
          cy_ptr += cell_size;
        }
      }
      hx_ptr += cell_size;
      if (mode == mkldnn_rnn_enum::kLstm) {
        cx_ptr += cell_size;
      }
      auto user_src_iter_memory = null_memory_;
      if (mode == mkldnn_rnn_enum::kLstm) {
        std::vector<DType*> srcs_data1;
        srcs_data1.push_back(hx_ptr);
        srcs_data1.push_back(cx_ptr);
        auto tmp1_src_iter_memory =
            ConcatData(mkldnn::memory::format::ldsnc, mkldnn::memory::format::ldsnc,
            {{1, 1, 1, N, H}, {1, 1, 1, N, H}}, {1, 1, nstates, N, H},
            mkldnn_dtype, 2, srcs_data1);

        std::vector<DType*> srcs_data2;
        srcs_data2.push_back(hx_ptr + single_cell_size);
        srcs_data2.push_back(cx_ptr + single_cell_size);
        auto tmp2_src_iter_memory =
            ConcatData(mkldnn::memory::format::ldsnc, mkldnn::memory::format::ldsnc,
            {{1, 1, 1, N, H}, {1, 1, 1, N, H}}, {1, 1, nstates, N, H},
            mkldnn_dtype, 2, srcs_data2);
        std::vector<DType*> srcs_data3;
        srcs_data3.push_back(reinterpret_cast<DType *>(tmp1_src_iter_memory.get_data_handle()));
        srcs_data3.push_back(reinterpret_cast<DType *>(tmp2_src_iter_memory.get_data_handle()));
        user_src_iter_memory =
            ConcatData(mkldnn::memory::format::ldsnc, mkldnn::memory::format::ldsnc,
            {{1, 1, nstates, N, H}, {1, 1, nstates, N, H}}, {1, 2, nstates, N, H},
            mkldnn_dtype, 1, srcs_data3);
      }
      back_w_ptr = w_ptr;
      back_b_ptr = b_ptr;
      DType* wx = w_ptr;  //  ngates * H, H * D
      DType* wh = w_ptr + (H * D) * H * ngates;  //  ngates * H, H
      back_w_ptr = w_ptr + ngates * H * ((H * D) + H);
      back_b_ptr = b_ptr + single_b_size * 2;
      DType* back_wx = back_w_ptr;
      DType* back_wh = back_w_ptr + (H * D) * H * ngates;
      DType* bx = b_ptr;
      DType* bh = b_ptr + H * ngates;
      DType* back_bx = back_b_ptr;
      DType* back_bh = back_b_ptr + H * ngates;

      auto src_wx = null_memory_, src_wh = null_memory_;
      std::vector<DType*> srcs_data1;
      srcs_data1.push_back(wx);
      srcs_data1.push_back(back_wx);
      src_wx =
          ConcatData(mkldnn::memory::format::ldgoi, mkldnn::memory::format::ldgoi,
          {weights_layer_r_tz, weights_layer_r_tz}, weights_layer_tz, mkldnn_dtype, 1, srcs_data1);
      ReorderForWeight(src_wx, (*weight_bias_memory)[(l + 1) * 3]);

      srcs_data1.clear();
      srcs_data1.push_back(wh);
      srcs_data1.push_back(back_wh);
      src_wh =
          ConcatData(mkldnn::memory::format::ldgoi, mkldnn::memory::format::ldgoi,
          {weights_iter_r_tz, weights_iter_r_tz}, weights_iter_tz, mkldnn_dtype, 1, srcs_data1);
      ReorderForWeight(src_wh, (*weight_bias_memory)[(l + 1) * 3 + 1]);

      DType* user_bias = reinterpret_cast<DType *>
          ((*weight_bias_memory)[(l + 1) * 3 + 2].get_data_handle());
      #pragma omp parallel for num_threads(omp_threads)
      for (int j = 0; j < single_b_size; j++) {
        user_bias[j] = bx[j] + bh[j];
      }
      #pragma omp parallel for num_threads(omp_threads)
      for (int j = 0; j < single_b_size; j++) {
        user_bias[single_b_size + j] = back_bx[j] + back_bh[j];
      }
      w_ptr += w_size;
      b_ptr += b_size;

      rnn_net2.push_back(
          rnn_forward(prim_desc, user_src_layer_memory_l, user_src_iter_memory,
                     (*weight_bias_memory)[(l + 1) * 3], (*weight_bias_memory)[(l + 1) * 3 + 1],
                     (*weight_bias_memory)[(l + 1) * 3 + 2], dst_layer_memory,
                     dst_iter_memory, null_memory_));
      stream(stream::kind::eager).submit(rnn_net2).wait();
      user_src_layer_memory_l = dst_layer_memory;
      if (state_outputs) {
        DType* dst_hcy = reinterpret_cast<DType *> (dst_iter_memory.get_data_handle());
        #pragma omp parallel for num_threads(omp_threads)
        for (int n = 0; n < single_cell_size; n++) {
          hy_ptr[n] = dst_hcy[n];
        }
        if (mode == mkldnn_rnn_enum::kLstm) {
          #pragma omp parallel for num_threads(omp_threads)
          for (int n = 0; n < single_cell_size; n++) {
            cy_ptr[n] = dst_hcy[n + single_cell_size];
          }
        }
        offset2 = nstates * single_cell_size;
        #pragma omp parallel for num_threads(omp_threads)
        for (int n = 0; n < single_cell_size; n++) {
          hy_ptr[n + single_cell_size] = dst_hcy[n + offset2];
        }
        if (mode == mkldnn_rnn_enum::kLstm) {
          offset2 = (nstates + 1) * single_cell_size;
          #pragma omp parallel for num_threads(omp_threads)
          for (int n = 0; n < single_cell_size; n++) {
            cy_ptr[n + single_cell_size] = dst_hcy[n + offset2];
          }
        }
      }
    }
  }
  if (L > 1 && D == 1) {
    w_ptr += w_size;
    b_ptr += b_size;
    w_size = (H + H) * H * ngates;
    memory::dims weights_layer_tz = {L - 1, 1, H, ngates, H};  //  ldigo
    memory::dims weights_iter_tz = {L - 1, 1, H, ngates, H};  //  ldigo
    memory::dims bias_tz = {L - 1, 1, ngates, H};
    memory::dims src_iter_tz = {L - 1, 1, nstates, N, H};  //  ldsnc
    memory::dims dst_iter_tz = {L - 1, 1, nstates, N, H};  //  ldsnc
    auto user_src_iter_md = mkldnn::memory::desc(
        { src_iter_tz }, mkldnn_dtype, mkldnn::memory::format::ldsnc);
    auto user_weight_layer_md = mkldnn::memory::desc(
        { weights_layer_tz }, mkldnn_dtype, mkldnn::memory::format::ldigo);
    auto user_weight_iter_md = mkldnn::memory::desc(
        { weights_iter_tz }, mkldnn_dtype, mkldnn::memory::format::ldigo);
    auto user_bias_md = mkldnn::memory::desc({ bias_tz },
        mkldnn_dtype, mkldnn::memory::format::ldgo);
    auto dst_iter_md = mkldnn::memory::desc(
        { dst_iter_tz }, mkldnn_dtype, mkldnn::memory::format::ldsnc);
    rnn_cell::desc cell(nalgorithm);
    rnn_forward::desc layer_desc(prop_kind::forward_inference, cell,
        rnn_direction::unidirectional,
        user_src_layer_md, user_src_iter_md, user_weight_layer_md, user_weight_iter_md,
        user_bias_md, dst_layer_md, dst_iter_md);
    auto prim_desc = mkldnn::rnn_forward::primitive_desc(layer_desc, cpu_engine);
    auto dst_layer_memory = mkldnn::memory(prim_desc.dst_layer_primitive_desc());
    dst_layer_memory.set_data_handle(y);
    auto dst_iter_memory = mkldnn::memory(prim_desc.dst_iter_primitive_desc());
    std::vector<primitive> rnn_net2;
    if (state_outputs) {
      hy_ptr += cell_size;
      if (mode == mkldnn_rnn_enum::kLstm) {
        cy_ptr += cell_size;
      }
    }
    std::vector<mkldnn::memory> f_memory;
    auto user_src_iter_memory = null_memory_;
    for (int l = 0; l < L - 1; l++) {
      hx_ptr += cell_size;
      if (mode == mkldnn_rnn_enum::kLstm) {
        cx_ptr += cell_size;
      }
      if (mode == mkldnn_rnn_enum::kLstm) {
        std::vector<DType*> srcs_data;
        srcs_data.push_back(hx_ptr);
        srcs_data.push_back(cx_ptr);
        auto tmp_src_iter_memory =
            ConcatData(mkldnn::memory::format::ldsnc, mkldnn::memory::format::ldsnc,
            {{1, 1, 1, N, H}, {1, 1, 1, N, H}}, {1, 1, nstates, N, H}, mkldnn_dtype, 2, srcs_data);
        f_memory.push_back(tmp_src_iter_memory);
      }
    }
    if (L == 2) {
      user_src_iter_memory = f_memory[0];
    } else {
      std::vector<DType*> src_l_data;
      std::vector<memory::dims> src_l_dim;
      for (int l = 0; l < L - 1; l++) {
        src_l_data.push_back(reinterpret_cast<DType *>(f_memory[l].get_data_handle()));
        src_l_dim.push_back({1, 1, nstates, N, H});
      }
      user_src_iter_memory =
          ConcatData(mkldnn::memory::format::ldsnc, mkldnn::memory::format::ldsnc,
              src_l_dim, {L - 1, 1, nstates, N, H}, mkldnn_dtype, 0, src_l_data);
    }

    auto src_wx_f = null_memory_, src_wh_f = null_memory_;
    std::vector<DType*> srcs_data_x;
    std::vector<DType*> srcs_data_h;
    std::vector<memory::dims> src_l_dim_x;
    std::vector<memory::dims> src_l_dim_h;
    for (int l = 0; l < L - 1; l++) {
      DType* wx = w_ptr;
      DType* wh = w_ptr + H * H * ngates;
      srcs_data_x.push_back(wx);
      srcs_data_h.push_back(wh);
      src_l_dim_x.push_back(weights_layer_r_tz);
      src_l_dim_h.push_back(weights_iter_r_tz);
      w_ptr = w_ptr + w_size;
    }
    src_wx_f =
        ConcatData(mkldnn::memory::format::ldgoi, mkldnn::memory::format::ldgoi,
        src_l_dim_x, weights_layer_tz, mkldnn_dtype, 0, srcs_data_x);
    ReorderForWeight(src_wx_f, (*weight_bias_memory)[3]);  // reorder L - 1 layers wx
    src_wh_f =
        ConcatData(mkldnn::memory::format::ldgoi, mkldnn::memory::format::ldgoi,
        src_l_dim_h, weights_iter_tz, mkldnn_dtype, 0, srcs_data_h);
    ReorderForWeight(src_wh_f, (*weight_bias_memory)[4]);  // reorder L - 1 layers wh

    DType* user_bias_f = reinterpret_cast<DType *> ((*weight_bias_memory)[5].get_data_handle());
    #pragma omp parallel for num_threads(omp_threads)
    for (int j = 0; j < (L - 1) * single_b_size; j++) {
      int k = j / single_b_size;
      user_bias_f[j] = b_ptr[j + k * single_b_size] + b_ptr[j + k * single_b_size + single_b_size];
    }
    rnn_net2.push_back(
        rnn_forward(prim_desc, user_src_layer_memory_l,
                    user_src_iter_memory, (*weight_bias_memory)[3],
                    (*weight_bias_memory)[4], (*weight_bias_memory)[5],
                    dst_layer_memory, dst_iter_memory, null_memory_));
    stream(stream::kind::eager).submit(rnn_net2).wait();

    if (state_outputs) {
      DType* dst_hcy = reinterpret_cast<DType *> (dst_iter_memory.get_data_handle());
      for (int l = 0; l < L - 1; l++) {
        offset1 = l * single_cell_size;
        offset2 = l * nstates * single_cell_size;
        #pragma omp parallel for num_threads(omp_threads)
        for (int n = 0; n < single_cell_size; n++) {
          hy_ptr[offset1 + n] = dst_hcy[offset2 + n];
        }
        if (mode == mkldnn_rnn_enum::kLstm) {
          #pragma omp parallel for num_threads(omp_threads)
          for (int n = 0; n < single_cell_size; n++) {
            cy_ptr[offset1 + n] = dst_hcy[offset2 + n + single_cell_size];
          }
        }
      }
    }
  }
}
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_USE_MKLDNN == 1
#endif  // MXNET_OPERATOR_NN_MKLDNN_MKLDNN_RNN_IMPL_H_
