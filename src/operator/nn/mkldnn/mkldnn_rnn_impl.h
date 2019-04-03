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

algorithm GetMKLDNNRNNAlgo(int mode,
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

void ReorderData(const mkldnn::memory &src,
                 const mkldnn::memory &dst) {
  MKLDNNStream::Get()->RegisterPrim(reorder(src, dst));
  MKLDNNStream::Get()->Submit();
}

void ConcatData(mkldnn::memory::format src_format,
                mkldnn::memory::format dst_format,
                std::vector<mkldnn::memory::dims> srcs_cds,
                mkldnn::memory::dims dst_cds,
                mkldnn::memory::data_type mkldnn_dtype,
                int concat_dimension,
                std::vector<void*> srcs_data,
                const mkldnn::memory &dst) {
  auto cpu_engine = CpuEngine::Get()->get_engine();
  std::vector<mkldnn::memory::primitive_desc> srcs_pd;
  std::vector<mkldnn::memory> srcs;
  for (size_t i = 0; i < srcs_cds.size(); i++) {
    auto desc = mkldnn::memory::desc(srcs_cds[i], mkldnn_dtype, src_format);
    auto mpd = mkldnn::memory::primitive_desc(desc, cpu_engine);
    auto src_memory = mkldnn::memory(mpd, srcs_data[i]);
    srcs_pd.push_back(mpd);
    srcs.push_back(src_memory);
  }
  std::vector<primitive::at> inputs;
  for (size_t i = 0; i < srcs_cds.size(); i++) {
    inputs.push_back(srcs[i]);
  }
  auto dst_desc = mkldnn::memory::desc(dst_cds, mkldnn_dtype, dst_format);
  auto concat_pd = concat::primitive_desc(dst_desc, concat_dimension, srcs_pd);
  MKLDNNStream::Get()->RegisterPrim(concat(concat_pd, inputs, dst));
  MKLDNNStream::Get()->Submit();
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
      size = 2 * (D * (I + H) * 4 * H + (L - 1) * D * (D * H + H) * 4 * H +
             L * D * 2 * N * H) + T * N * D * H + L * 2 * D * 4 * H + (L + 2) * D * 2 * N * H +
             6 * D * (I + H + 2) * 4 * H + T * N * I * 2 + L * 4 * (H + I + 2) * 7;
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

template<typename DType>
inline DType getmax(DType* x, size_t size) {
  DType ret = 0;
  if (std::is_same<float, DType>::value) {
    CBLAS_INDEX idx = cblas_isamax(size, reinterpret_cast<float*>(x), 1);
    ret = (x[idx] != 0) ? fabs(x[idx]) : 1;
  } else if (std::is_same<double, DType>::value) {
    CBLAS_INDEX idx = cblas_idamax(size, reinterpret_cast<double*>(x), 1);
    ret = (x[idx] != 0) ? fabs(x[idx]) : 1;
  } else {
    LOG(INFO) << "not support";
  }
  return ret;
}

// since there is different sematics of MKLDNN's Fused RNN and Mxnet FusedRNN,
// bidirectional will be fused layer by layer,
// unidirectional will be done by fused 1 + fused (L - 1) layers or fused L layers(when I = H)
template <typename DType>
void MKLDNNRNNForwardUnidi(bool state_outputs,
                           const int L,
                           const int T,
                           const int N,
                           const int I,
                           const int H,
                           DType* x_ptr,
                           mkldnn::memory user_src_layer_memory,
                           DType* hx_ptr,
                           DType* cx_ptr,
                           DType* w_ptr,
                           DType* b_ptr,
                           DType* y_ptr,
                           DType* hy_ptr,
                           DType* cy_ptr,
                           std::vector<mkldnn::memory> *concat_weight_memory,
                           std::vector<mkldnn::memory> *concat_iter_memory,
                           std::vector<mkldnn::memory> *x_memory,
                           std::vector<mkldnn::memory> *x_memory_u8,
                           std::vector<mkldnn::memory> *hcx_memory,
                           std::vector<mkldnn::memory> *wx_memory,
                           std::vector<mkldnn::memory> *wx_memory_s8,
                           std::vector<mkldnn::memory> *wh_memory,
                           std::vector<mkldnn::memory> *wh_memory_s8,
                           std::vector<mkldnn::memory> *bias_memory,
                           std::vector<mkldnn::memory> *y_memory,
                           std::vector<mkldnn::memory> *y_memory_u8,
                           std::vector<mkldnn::memory> *hcy_memory,
                           std::vector<primitive> *rnn_forward_prim,
                           int layer_index,
                           bool has_nextlayer,
                           bool *has_cache,
                           int dtype,
                           int mode) {
  int ngates = 0, nstates = 0;
  //algorithm nalgorithm = GetMKLDNNRNNAlgo(mode, &ngates, &nstates);
  GetMKLDNNRNNAlgo(mode, &ngates, &nstates);
  mkldnn::memory::data_type mkldnn_dtype = get_mkldnn_type(dtype);
  const int cell_size = N * H;
  const int single_cell_size = N * H;
  const int single_b_size = ngates * H;
  int w_size = (I + H) * H * ngates;

  const int omp_threads = mxnet::engine::OpenMP::Get()->GetRecommendedOMPThreadCount();
  auto cpu_engine = CpuEngine::Get()->get_engine();
  auto null_memory_ = null_memory(cpu_engine);
  int offset1 = 0, offset2 = 0;
  bool cached = *has_cache;
 
  const float data_shift = 64.;
  const float data_scale = 63.;

  const int weights_scale_mask = 3; // 11 for last two dimensions of ldigo
  
  mkldnn::memory::dims src_layer_tz = {T, N, I};
  mkldnn::memory::dims dst_layer_tz = {T, N, H};
  mkldnn::memory::dims weights_layer_tz = {L, 1, I, ngates, H};  //  ldigo
  mkldnn::memory::dims weights_iter_tz = {L, 1, H, ngates, H};  //  ldigo
  mkldnn::memory::dims bias_tz = {L, 1, ngates, H};
  mkldnn::memory::dims src_iter_tz = {L, 1, nstates, N, H};  //  ldsnc
  mkldnn::memory::dims dst_iter_tz = {L, 1, nstates, N, H};  //  ldsnc
  mkldnn::memory::dims weights_layer_r_tz = {1, 1, I, ngates, H};  //  ldigo for reorder
  mkldnn::memory::dims weights_iter_r_tz = {1, 1, H, ngates, H};  //  ldigo for reorder

  //auto src_layer_md = mkldnn::memory::desc(
      //{src_layer_tz}, mkldnn_dtype, mkldnn::memory::format::tnc);
  auto dst_layer_md = mkldnn::memory::desc(
      {dst_layer_tz}, mkldnn_dtype, mkldnn::memory::format::tnc);
  auto src_iter_md = mkldnn::memory::desc(
      {src_iter_tz}, mkldnn_dtype, mkldnn::memory::format::ldsnc);
  //auto weight_layer_md = mkldnn::memory::desc(
      //{weights_layer_tz}, mkldnn_dtype, mkldnn::memory::format::ldigo);
  //auto weight_iter_md = mkldnn::memory::desc(
      //{weights_iter_tz}, mkldnn_dtype, mkldnn::memory::format::ldigo);
  auto bias_md = mkldnn::memory::desc({bias_tz},
      mkldnn_dtype, mkldnn::memory::format::ldgo);
  auto dst_iter_md = mkldnn::memory::desc(
      {dst_iter_tz}, mkldnn_dtype, mkldnn::memory::format::ldsnc);

  for (int l = 0; l < L; l++) {
    if (mode == rnn_enum::kLstm) {
      std::vector<void*> srcs_data;
      srcs_data.push_back(hx_ptr);
      srcs_data.push_back(cx_ptr);
      auto tmp_src_iter_memory = (*concat_iter_memory)[l + layer_index];
      ConcatData(mkldnn::memory::format::ldsnc, mkldnn::memory::format::ldsnc,
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
    std::vector<mkldnn::memory::dims> src_l_dim;
    for (int l = 0; l < L; l++) {
      src_l_data.push_back(reinterpret_cast<DType *>
          ((*concat_iter_memory)[l + layer_index].get_data_handle()));
      src_l_dim.push_back({1, 1, nstates, N, H});
    }
    ConcatData(mkldnn::memory::format::ldsnc, mkldnn::memory::format::ldsnc, src_l_dim,
        {L, 1, nstates, N, H}, mkldnn_dtype, 0, src_l_data, user_src_iter_memory);
  }
  (*hcx_memory)[layer_index].set_data_handle(user_src_iter_memory.get_data_handle());

  auto src_wx_f = (*concat_weight_memory)[2 * layer_index];
  auto src_wh_f = (*concat_weight_memory)[2 * layer_index + 1];

  std::vector<void*> srcs_data_x;
  std::vector<void*> srcs_data_h;
  std::vector<mkldnn::memory::dims> src_l_dim_x;
  std::vector<mkldnn::memory::dims> src_l_dim_h;
  std::vector<float> weights_scales(ngates * H);
  if (!cached) {    
    if (L == 1) {
      DType* wx = w_ptr;
      DType* wh = w_ptr + I * H * ngates;
      src_wx_f.set_data_handle(wx);
      src_wh_f.set_data_handle(wh);
      const Tensor<cpu, 2, DType> wx_tensor(wx, Shape2(ngates * H, I));
      const Tensor<cpu, 2, DType> wh_tensor(wh, Shape2(ngates * H, H));
      DType* wx_tensor_data = wx_tensor.dptr_;
      DType* wh_tensor_data = wh_tensor.dptr_;
      #pragma omp parallel for num_threads(omp_threads)
      for (int i = 0; i< ngates * H; i++) {
        DType max_wx = getmax(wx_tensor_data + i * I, I);
        DType max_wh = getmax(wh_tensor_data + i * H, H);
        weights_scales[i] = data_scale/(max_wx > max_wh ? max_wx : max_wh);
      }

    } else {
      DType tmp_max[ngates * H][L];
      for (int l = 0; l < L; l++) {
        DType* wx = w_ptr;
        DType* wh = w_ptr + I * H * ngates;
        srcs_data_x.push_back(wx);
        srcs_data_h.push_back(wh);
        src_l_dim_x.push_back(weights_layer_r_tz);
        src_l_dim_h.push_back(weights_iter_r_tz);
        const Tensor<cpu, 2, DType> wx_tensor(wx, Shape2(ngates * H, I));
        const Tensor<cpu, 2, DType> wh_tensor(wh, Shape2(ngates * H, H));
        DType* wx_tensor_data = wx_tensor.dptr_;
        DType* wh_tensor_data = wh_tensor.dptr_;
        #pragma omp parallel for num_threads(omp_threads)
        for (int i = 0; i< ngates * H; i++ ) {
          DType max_wx = getmax(wx_tensor_data + i * I, I);
          DType max_wh = getmax(wh_tensor_data + i * H, H);
          tmp_max[i][l] = max_wx > max_wh ? max_wx : max_wh;
        }
        w_ptr = w_ptr + w_size;
      }
      #pragma omp parallel for num_threads(omp_threads)
      for (int i = 0; i< ngates * H; i++) {
        weights_scales[i] = data_scale/getmax(tmp_max[i], L);
      }
      ConcatData(mkldnn::memory::format::ldgoi, mkldnn::memory::format::ldgoi,
          src_l_dim_x, weights_layer_tz, mkldnn_dtype, 0, srcs_data_x, src_wx_f);
      ConcatData(mkldnn::memory::format::ldgoi, mkldnn::memory::format::ldgoi,
          src_l_dim_h, weights_iter_tz, mkldnn_dtype, 0, srcs_data_h, src_wh_f);
    }
    ReorderData(src_wx_f, (*wx_memory)[layer_index]);
    ReorderData(src_wh_f, (*wh_memory)[layer_index]);
    
    DType* user_bias_f = reinterpret_cast<DType *> ((*bias_memory)[layer_index].get_data_handle());
    #pragma omp parallel for num_threads(omp_threads)
    for (int j = 0; j < L * single_b_size; j++) {
      int k = j / single_b_size;
      user_bias_f[j] = b_ptr[j + k * single_b_size] + b_ptr[j + k * single_b_size + single_b_size];
    }
    // cached data is ready now
  } 
  auto src_layer_md_u8 = memory::desc({src_layer_tz },
          memory::data_type::u8, memory::format::any);
  auto weight_layer_md_s8 = memory::desc({weights_layer_tz },
          memory::data_type::s8, memory::format::any);
  auto weight_iter_md_s8 = memory::desc({weights_iter_tz },
          memory::data_type::s8, memory::format::any);
  auto dst_layer_md_u8 = memory::desc({dst_layer_tz },
          memory::data_type::u8, memory::format::any);

  rnn_cell::desc lstm_cell(algorithm::vanilla_lstm);

  primitive_attr attr;
  attr.set_int_output_round_mode(round_mode::round_nearest);
  attr.set_rnn_data_qparams(data_scale, data_shift);
  attr.set_rnn_weights_qparams(weights_scale_mask, weights_scales);
  std::vector<primitive> rnn_net;
  std::vector<primitive> src_int8;
  std::vector<primitive> weights_int8;

  auto dst_layer_memory_u8 = null_memory_;
  auto src_layer_memory_u8 = null_memory_;
  auto weight_layer_memory_s8 = null_memory_;
  auto weight_iter_memory_s8 = null_memory_;

  rnn_forward::desc layer_desc(prop_kind::forward_inference, lstm_cell,
      rnn_direction::unidirectional, src_layer_md_u8,
      src_iter_md, weight_layer_md_s8, weight_iter_md_s8,
      bias_md, has_nextlayer ? dst_layer_md_u8 : dst_layer_md, dst_iter_md); 

  auto prim_desc
       = rnn_forward::primitive_desc(layer_desc, attr, cpu_engine);

  if (x_ptr && layer_index == 0) {
    (*x_memory)[layer_index].set_data_handle(x_ptr);
    src_layer_memory_u8 = mkldnn::memory(prim_desc.src_layer_primitive_desc());
    auto src_layer_reorder_pd = reorder::primitive_desc(
        (*x_memory)[layer_index].get_primitive_desc(),
        src_layer_memory_u8.get_primitive_desc(), attr);
    src_int8.push_back(reorder(src_layer_reorder_pd,
        (*x_memory)[layer_index], src_layer_memory_u8));
    MKLDNNStream::Get()->RegisterPrim(src_int8[0]);
    MKLDNNStream::Get()->Submit();
    x_memory_u8->push_back(src_layer_memory_u8);
  }
  if (!cached) {
    weight_layer_memory_s8 = mkldnn::memory(prim_desc.weights_layer_primitive_desc());
    auto weight_layer_reorder_pd = reorder::primitive_desc(
        (*wx_memory)[layer_index].get_primitive_desc(),
        weight_layer_memory_s8.get_primitive_desc(), attr);
    weights_int8.push_back(reorder(weight_layer_reorder_pd,
        (*wx_memory)[layer_index], weight_layer_memory_s8));

    weight_iter_memory_s8 = mkldnn::memory(prim_desc.weights_iter_primitive_desc());
    auto weight_iter_reorder_pd = reorder::primitive_desc(
        (*wh_memory)[layer_index].get_primitive_desc(),
        weight_iter_memory_s8.get_primitive_desc(), attr);
    weights_int8.push_back(reorder(weight_iter_reorder_pd,
      (*wh_memory)[layer_index], weight_iter_memory_s8));

    MKLDNNStream::Get()->RegisterPrim(weights_int8[0]);
    MKLDNNStream::Get()->RegisterPrim(weights_int8[1]);
    MKLDNNStream::Get()->Submit();

    wx_memory_s8->push_back(weight_layer_memory_s8);
    wh_memory_s8->push_back(weight_iter_memory_s8);
  }

  if (!has_nextlayer) {
    (*y_memory)[layer_index].set_data_handle(y_ptr);
  }

  if (rnn_forward_prim->size() <= layer_index) {
    if (has_nextlayer) {    
      dst_layer_memory_u8
          = mkldnn::memory(prim_desc.dst_layer_primitive_desc());
      y_memory_u8->push_back(dst_layer_memory_u8);      
    }
    primitive rnn_prim = rnn_forward(prim_desc, (*x_memory_u8)[layer_index],
          (*hcx_memory)[layer_index], (*wx_memory_s8)[layer_index],
          (*wh_memory_s8)[layer_index], (*bias_memory)[layer_index],
          has_nextlayer ? (*y_memory_u8)[layer_index] : (*y_memory)[layer_index],
         (*hcy_memory)[layer_index], null_memory_);
    rnn_forward_prim->push_back(rnn_prim);
  }
  MKLDNNStream::Get()->RegisterPrim((*rnn_forward_prim)[layer_index]);
  MKLDNNStream::Get()->Submit();
  if (has_nextlayer) {
    x_memory_u8->push_back((*y_memory_u8)[layer_index]);
  }

  if (state_outputs) {
    DType* dst_hcy = reinterpret_cast<DType *> ((*hcy_memory)[layer_index].get_data_handle());
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
void MKLDNNRNNForwardINT8(bool state_outputs,
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
                          std::vector<mkldnn::memory> *concat_weight_memory,
                          std::vector<mkldnn::memory> *concat_iter_memory,
                          std::vector<mkldnn::memory> *x_memory,
                          std::vector<mkldnn::memory> *x_memory_u8,
                          std::vector<mkldnn::memory> *hcx_memory,
                          std::vector<mkldnn::memory> *wx_memory,
                          std::vector<mkldnn::memory> *wx_memory_s8,
                          std::vector<mkldnn::memory> *wh_memory,
                          std::vector<mkldnn::memory> *wh_memory_s8,
                          std::vector<mkldnn::memory> *bias_memory,
                          std::vector<mkldnn::memory> *y_memory,
                          std::vector<mkldnn::memory> *y_memory_u8,
                          std::vector<mkldnn::memory> *hcy_memory,
                          std::vector<primitive> *rnn_forward_prim,
                          bool *has_cache,
                          int dtype,
                          int mode) {
  int ngates = 0, nstates = 0;
  GetMKLDNNRNNAlgo(mode, &ngates, &nstates);
  const int b_size = 2 * H * ngates * D;
  //const int cell_size = N * H * D;
  //  First layer
  int w_size = (I + H) * H * ngates * D;
  auto cpu_engine = CpuEngine::Get()->get_engine();
  auto null_memory_ = null_memory(cpu_engine);
  DType* tmpNull = NULL;
  
  // when D = 1 and I == H, L layers can be fused together
  if (D == 1 && I == H) {
    MKLDNNRNNForwardUnidi(state_outputs, L, T, N, I, H, x_ptr, null_memory_,
        hx_ptr, cx_ptr, w_ptr, b_ptr, y_ptr, hy_ptr, cy_ptr, concat_weight_memory,
        concat_iter_memory, x_memory, x_memory_u8, hcx_memory, wx_memory, wx_memory_s8, wh_memory,
        wh_memory_s8, bias_memory, y_memory, y_memory_u8, hcy_memory, rnn_forward_prim,
        0, false, has_cache, dtype, mode);
  } else {
    auto user_src_layer_memory_l = null_memory_;
    if (D == 2) {
      /*MKLDNNRNNForwardSingleLayerBi(state_outputs, T, N, I, H, x_ptr, user_src_layer_memory_l,
          hx_ptr, cx_ptr, w_ptr, b_ptr, y_ptr, hy_ptr, cy_ptr, concat_weight_memory,
          concat_iter_memory, x_memory, hcx_memory, wx_memory, wh_memory,
          bias_memory, cached_wx_memory, cached_wh_memory, cached_bias_memory,
          y_memory, hcy_memory, rnn_forward_prim, call_num, 0, dtype, mode);*/
    } else {
      MKLDNNRNNForwardUnidi(state_outputs, 1, T, N, I, H, x_ptr, user_src_layer_memory_l,
          hx_ptr, cx_ptr, w_ptr, b_ptr, y_ptr, hy_ptr, cy_ptr, concat_weight_memory,
          concat_iter_memory, x_memory, x_memory_u8, hcx_memory, wx_memory, wx_memory_s8, wh_memory,
          wh_memory_s8, bias_memory, y_memory, y_memory_u8, hcy_memory,rnn_forward_prim,
          0, L > 1 ? true : false, has_cache, dtype, mode);
    }
    if (L > 1) {
      user_src_layer_memory_l = (*y_memory_u8)[0];
      //  go to next L - 1 layers.
      //  If D = 2, do it layer by layer. If D = 1, fused L - 1 layers
      w_ptr += w_size;
      b_ptr += b_size;
      if (D == 2) {
        /*w_size = (H * D + H) * H * ngates * D;
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
              concat_weight_memory, concat_iter_memory, x_memory, hcx_memory, wx_memory,
              wh_memory, bias_memory, cached_wx_memory, cached_wh_memory, cached_bias_memory,
              y_memory, hcy_memory, rnn_forward_prim, call_num, 1, dtype, mode);
          user_src_layer_memory_l = (*y_memory)[1];
          w_ptr += w_size;
          b_ptr += b_size;
        }*/
      }
      if (D == 1) {
        w_size = (H + H) * H * ngates;
        MKLDNNRNNForwardUnidi(state_outputs, L - 1, T, N, H, H, tmpNull, user_src_layer_memory_l,
            hx_ptr, cx_ptr, w_ptr, b_ptr, y_ptr, hy_ptr, cy_ptr, concat_weight_memory,
            concat_iter_memory, x_memory, x_memory_u8, hcx_memory, wx_memory, wx_memory_s8, wh_memory,
            wh_memory_s8, bias_memory, y_memory, y_memory_u8, hcy_memory,rnn_forward_prim,
            1, false, dtype, has_cache, mode);
      }
    }
  }
  *has_cache = true;
}

template <typename DType>
void MKLDNNRNNForwardInferenceINT8(bool state_outputs,
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
                                   std::vector<mkldnn::memory>* concat_weight_memory,
                                   std::vector<mkldnn::memory>* concat_iter_memory,
                                   std::vector<mkldnn::memory> *x_memory,
                                   std::vector<mkldnn::memory> *x_memory_u8,
                                   std::vector<mkldnn::memory> *hcx_memory,
                                   std::vector<mkldnn::memory> *wx_memory,
                                   std::vector<mkldnn::memory> *wx_memory_s8,
                                   std::vector<mkldnn::memory> *wh_memory,
                                   std::vector<mkldnn::memory> *wh_memory_s8,
                                   std::vector<mkldnn::memory> *bias_memory,
                                   std::vector<mkldnn::memory> *y_memory,
                                   std::vector<mkldnn::memory> *y_memory_u8,
                                   std::vector<mkldnn::memory> *hcy_memory,
                                   std::vector<primitive> *rnn_forward_prim,
                                   bool *has_cache;
                                   int dtype,
                                   int mode) {
  switch (mode) {
    case rnn_enum::kLstm:
      MKLDNNRNNForwardINT8<DType>(state_outputs, num_layers, direction, seq_length,
                                  batch_size, input_size, state_size, x_ptr, hx_ptr,
                                  cx_ptr, w_ptr, b_ptr, y_ptr, hy_ptr, cy_ptr,
                                  concat_weight_memory, concat_iter_memory, x_memory, x_memory_u8,
                                  hcx_memory, wx_memory, wx_memory_s8, wh_memory, wh_memory_s8,
                                  bias_memory, y_memory, y_memory_u8, hcy_memory, rnn_forward_prim,
                                  has_cache, dtype, mode);
      break;
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

    if (!ctx.is_train) {
      MKLDNNRNNForwardInferenceINT8<DType>(param_.state_outputs,
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
                                           &concat_weight_memory,
                                           &concat_iter_memory,
                                           &x_memory,
                                           &x_memory_u8,
                                           &hcx_memory,
                                           &wx_memory,
                                           &wx_memory_s8,
                                           &wh_memory,
                                           &wh_memory_s8,
                                           &bias_memory,
                                           &y_memory,
                                           &y_memory_u8,
                                           &hcy_memory,
                                           &rnn_forward_prim,
                                           &has_cache,
                                           dtype,
                                           param_.mode);
    }
  }

  RNNParam param_;
  std::vector<mkldnn::memory> concat_weight_memory;
  std::vector<mkldnn::memory> concat_iter_memory;
  std::vector<primitive> rnn_forward_prim;
  std::vector<mkldnn::memory> x_memory;
  std::vector<mkldnn::memory> x_memory_u8;
  std::vector<mkldnn::memory> hcx_memory;
  std::vector<mkldnn::memory> wx_memory;
  std::vector<mkldnn::memory> wx_memory_s8;
  std::vector<mkldnn::memory> wh_memory;
  std::vector<mkldnn::memory> wh_memory_s8;
  std::vector<mkldnn::memory> bias_memory;
  std::vector<mkldnn::memory> y_memory;
  std::vector<mkldnn::memory> y_memory_u8;
  std::vector<mkldnn::memory> hcy_memory;
  bool has_cache;
  bool init_mem_;
  size_t reserve_mem_size_;
  Storage::Handle mem_space_;
};  // class MKLDNNRNNOp

static OpStatePtr CreateRNNState(const nnvm::NodeAttrs &attrs,
                                 const Context ctx,
                                 const mxnet::ShapeVector &in_shapes,
                                 const std::vector<int> &in_types) {
  const RNNParam& param = nnvm::get<RNNParam>(attrs.parsed);
  OpStatePtr state = OpStatePtr();
  MSHADOW_REAL_TYPE_SWITCH(in_types[rnn_enum::kData], DType, {
    state = OpStatePtr::Create<MKLDNNRNNOp<DType>>(param);
  });
  return state;
}

static void RNNStatefulComputeCPU(const OpStatePtr& state_ptr,
                                  const OpContext& ctx,
                                  const std::vector<NDArray>& inputs,
                                  const std::vector<OpReqType>& req,
                                  const std::vector<NDArray>& outputs) {
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
  mkldnn::memory::data_type mkldnn_dtype = get_mkldnn_type(dtype);
  Stream<cpu> *s = ctx.get_stream<cpu>();
  auto cpu_engine = CpuEngine::Get()->get_engine();
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    MKLDNNRNNOp<DType>& op = state_ptr.get_state<MKLDNNRNNOp<DType>>();
    const RNNParam& param = op.param_;

    int ngates = 0, nstates = 0;
    GetMKLDNNRNNAlgo(param.mode, &ngates, &nstates);
    int D = param.bidirectional ? 2 : 1;
    Tensor<cpu, 3, DType> x = in_blobs[rnn_enum::kData].get<cpu, 3, DType>(s);
    int T = x.shape_[0];
    int N = x.shape_[1];
    int I = x.shape_[2];
    int H = param.state_size;
    int L = param.num_layers;
    const size_t r_size = GetMKLDNNRNNCacheMemorySize(L, D, T, N, I, H, param.mode);

    if (op.init_mem_ && op.reserve_mem_size_ < r_size) {
      Storage::Get()->Free(op.mem_space_);
      op.init_mem_ = false;
    }

    if (!op.init_mem_) {
      op.mem_space_ = Storage::Get()->Alloc(
          r_size * sizeof(DType),
          Context::CPU());
      op.reserve_mem_size_ = r_size;
      op.init_mem_ = true;
    }
    if (op.has_cache && op.x_memory.size() == 0) {
      op.has_cache = false;
    }
    DType* workptr = static_cast<DType*>(op.mem_space_.dptr);
    mkldnn::memory::dims src_layer_tz_0 = {T, N, I};
    mkldnn::memory::dims src_layer_tz = {T, N, D * H};
    mkldnn::memory::dims dst_layer_tz = {T, N, D * H};
    auto dst_layer_md = mkldnn::memory::desc(
      { dst_layer_tz }, mkldnn_dtype, mkldnn::memory::format::tnc);
    if (op.x_memory.size() == 0) {
      if (D == 1 && I == H) {
        auto user_src_layer_md = mkldnn::memory::desc(
            { src_layer_tz }, mkldnn_dtype, mkldnn::memory::format::tnc);
        auto user_src_layer_memory_n = mkldnn::memory({ user_src_layer_md, cpu_engine });
        op.x_memory.push_back(user_src_layer_memory_n);
        
        mkldnn::memory::dims weights_layer_tz = {L, 1, I, ngates, H};  //  ldigo
        mkldnn::memory::dims weights_iter_tz = {L, 1, H, ngates, H};  //  ldigo
        mkldnn::memory::dims bias_tz = {L, 1, ngates, H};
        auto user_weight_layer_md = mkldnn::memory::desc(
            { weights_layer_tz }, mkldnn_dtype, mkldnn::memory::format::ldigo);
        auto user_weight_iter_md = mkldnn::memory::desc(
            { weights_iter_tz }, mkldnn_dtype, mkldnn::memory::format::ldigo);
        auto user_bias_md = mkldnn::memory::desc({ bias_tz },
            mkldnn_dtype, mkldnn::memory::format::ldgo);
        DType* weight_layer_n = workptr;  //  L * I * ngates * H
        auto user_weight_layer_memory_n
            = mkldnn::memory({ user_weight_layer_md, cpu_engine }, weight_layer_n);
        op.wx_memory.push_back(user_weight_layer_memory_n);

        DType* weight_iter_n = weight_layer_n + L * I * ngates * H;  //  L * H * ngates * H
        auto user_weight_iter_memory_n
            = mkldnn::memory({ user_weight_iter_md, cpu_engine }, weight_iter_n);
        op.wh_memory.push_back(user_weight_iter_memory_n);

        DType* bias_n = weight_iter_n + L * H * ngates * H;  //  L * ngates * H
        auto user_bias_memory_n =
            mkldnn::memory({ user_bias_md, cpu_engine }, bias_n);
        op.bias_memory.push_back(user_bias_memory_n);

        auto wx_md_n = mkldnn::memory::desc(
            { weights_layer_tz }, mkldnn_dtype, mkldnn::memory::format::ldgoi);
        DType* wx_n = bias_n + L * ngates * H;  //   L * ngates * I * H
        auto wx_memory_n =
            mkldnn::memory({ wx_md_n, cpu_engine }, wx_n);
        DType* wh_n = wx_n + L * ngates * I * H;  //  L * ngates * H * H
        auto wh_md_n = mkldnn::memory::desc(
            { weights_iter_tz }, mkldnn_dtype, mkldnn::memory::format::ldgoi);
        auto wh_memory_n =
            mkldnn::memory({ wh_md_n, cpu_engine }, wh_n);

        op.concat_weight_memory.push_back(wx_memory_n);
        op.concat_weight_memory.push_back(wh_memory_n);
        workptr = wh_n + L * ngates * H * H;

        mkldnn::memory::dims src_iter_tz_n1 = {1, 1, nstates, N, H};  //  ldsnc
        auto src_iter_md_n1 = mkldnn::memory::desc(
            { src_iter_tz_n1 }, mkldnn_dtype, mkldnn::memory::format::ldsnc);
        for (int l = 0; l < L; l++) {
          DType* src_iter_n1 = workptr;  //  nstates * N * H
          auto src_iter_memory_n1 =
              mkldnn::memory({ src_iter_md_n1, cpu_engine }, src_iter_n1);
          op.concat_iter_memory.push_back(src_iter_memory_n1);
          workptr = src_iter_n1 + nstates * N * H;
        }
        mkldnn::memory::dims src_iter_tz_n = {L, 1, nstates, N, H};  //  ldsnc
        auto src_iter_md_n = mkldnn::memory::desc(
            { src_iter_tz_n }, mkldnn_dtype, mkldnn::memory::format::ldsnc);
        DType* src_iter_n = workptr;  //  L * nstates * N * H
        auto src_iter_memory_n =
            mkldnn::memory({ src_iter_md_n, cpu_engine }, src_iter_n);
        op.concat_iter_memory.push_back(src_iter_memory_n);
        op.hcx_memory.push_back(src_iter_memory_n);
        DType* dst_layer_n = src_iter_n + L * nstates * N * H;  //  T * N * D * H
        auto dst_layer_memory_n
            = mkldnn::memory({ dst_layer_md, cpu_engine }, dst_layer_n);
        op.y_memory.push_back(dst_layer_memory_n);

        mkldnn::memory::dims dst_iter_tz_n = {L, 1, nstates, N, H};  //  ldsnc
        auto dst_iter_md_n = mkldnn::memory::desc(
            { dst_iter_tz_n }, mkldnn_dtype, mkldnn::memory::format::ldsnc);
        DType* dst_iter_n = dst_layer_n + T * N * D * H;  //  L * nstates * N * H
        auto dst_iter_memory_n =
            mkldnn::memory({ dst_iter_md_n, cpu_engine }, dst_iter_n);
        op.hcy_memory.push_back(dst_iter_memory_n);
        workptr = dst_iter_n + L * nstates * N * H;

      } else {
        auto user_src_layer_md_0 = mkldnn::memory::desc(
            { src_layer_tz_0 }, mkldnn_dtype, mkldnn::memory::format::tnc);
        auto user_src_layer_memory_0 = mkldnn::memory({ user_src_layer_md_0, cpu_engine });
        op.x_memory.push_back(user_src_layer_memory_0);

        mkldnn::memory::dims weights_layer_tz_0 = {1, D, I, ngates, H};  //  ldigo
        mkldnn::memory::dims weights_iter_tz_0 = {1, D, H, ngates, H};  //  ldigo
        mkldnn::memory::dims bias_tz_0 = {1, D, ngates, H};
        auto user_weight_layer_md_0 = mkldnn::memory::desc(
            { weights_layer_tz_0 }, mkldnn_dtype, mkldnn::memory::format::ldigo);
        auto user_weight_iter_md_0 = mkldnn::memory::desc(
            { weights_iter_tz_0 }, mkldnn_dtype, mkldnn::memory::format::ldigo);
        auto user_bias_md_0 = mkldnn::memory::desc({ bias_tz_0 },
            mkldnn_dtype, mkldnn::memory::format::ldgo);

        DType* weight_layer_0 = workptr;  //  D * I * ngates * H
        auto user_weight_layer_memory_0
            = mkldnn::memory({ user_weight_layer_md_0, cpu_engine }, weight_layer_0);
        op.wx_memory.push_back(user_weight_layer_memory_0);

        DType* weight_iter_0 = weight_layer_0 + D * I * ngates * H;  //  D * H * ngates * H
        auto user_weight_iter_memory_0
            = mkldnn::memory({ user_weight_iter_md_0, cpu_engine }, weight_iter_0);
        op.wh_memory.push_back(user_weight_iter_memory_0);

        DType* bias_0 = weight_iter_0 + D * H * ngates * H;  //  D * ngates * H
        auto user_bias_memory_0 =
            mkldnn::memory({ user_bias_md_0, cpu_engine }, bias_0);
        op.bias_memory.push_back(user_bias_memory_0);
        workptr = bias_0 + D * ngates * H;

        auto wx_md_0 = mkldnn::memory::desc(
            { weights_layer_tz_0 }, mkldnn_dtype, mkldnn::memory::format::ldgoi);
        auto wx_memory_0 =
            mkldnn::memory({ wx_md_0, cpu_engine });
        auto wh_md_0 = mkldnn::memory::desc(
            { weights_iter_tz_0 }, mkldnn_dtype, mkldnn::memory::format::ldgoi);
        auto wh_memory_0 =
            mkldnn::memory({ wh_md_0, cpu_engine });
        if (D == 2) {
          DType* wx_0 = workptr;  //  D * ngates * I * H
          wx_memory_0.set_data_handle(wx_0);
          DType* wh_0 = wx_0 + D * ngates * I * H;  //  D * ngates * H * H
          wh_memory_0.set_data_handle(wh_0);
          workptr = wh_0 + D * ngates * H * H;
        }
        op.concat_weight_memory.push_back(wx_memory_0);
        op.concat_weight_memory.push_back(wh_memory_0);

        mkldnn::memory::dims src_iter_undi_tz_0 = {1, 1, nstates, N, H};  //  ldsnc
        auto src_iter_undi_md_0 = mkldnn::memory::desc(
            { src_iter_undi_tz_0 }, mkldnn_dtype, mkldnn::memory::format::ldsnc);
        DType* src_iter_undi_0 = workptr;  //  nstates * N * H
        auto src_iter_undi_memory_0 =
            mkldnn::memory({ src_iter_undi_md_0, cpu_engine }, src_iter_undi_0);
        op.concat_iter_memory.push_back(src_iter_undi_memory_0);
        workptr = src_iter_undi_0 + nstates * N * H;
        if (D == 1) {
          op.hcx_memory.push_back(src_iter_undi_memory_0);
        } else {
          DType* src_iter_undi2_0 = workptr;  //  nstates * N * H
          auto src_iter_undi2_memory_0 =
              mkldnn::memory({ src_iter_undi_md_0, cpu_engine }, src_iter_undi2_0);
          op.concat_iter_memory.push_back(src_iter_undi2_memory_0);

          mkldnn::memory::dims src_iter_tz_0 = {1, D, nstates, N, H};  //  ldsnc
          auto src_iter_md_0 = mkldnn::memory::desc(
              { src_iter_tz_0 }, mkldnn_dtype, mkldnn::memory::format::ldsnc);
          DType* src_iter_0 = src_iter_undi2_0 + nstates * N * H;  //  D * nstates * N * H
          auto src_iter_memory_0 =
              mkldnn::memory({ src_iter_md_0, cpu_engine }, src_iter_0);
          op.concat_iter_memory.push_back(src_iter_memory_0);
          op.hcx_memory.push_back(src_iter_memory_0);
          workptr = src_iter_0 + D * nstates * N * H;
        }

        DType* dst_layer_0 = workptr;  //  T * N * D * H
        auto dst_layer_memory_0
            = mkldnn::memory({ dst_layer_md, cpu_engine }, dst_layer_0);
        op.y_memory.push_back(dst_layer_memory_0);

        mkldnn::memory::dims dst_iter_tz_0 = {1, D, nstates, N, H};  //  ldsnc
        auto dst_iter_md_0 = mkldnn::memory::desc(
            { dst_iter_tz_0 }, mkldnn_dtype, mkldnn::memory::format::ldsnc);
        DType* dst_iter_0 = dst_layer_0 + T * N * D * H;  //  D * nstates * N * H
        auto dst_iter_memory_0 =
            mkldnn::memory({ dst_iter_md_0, cpu_engine }, dst_iter_0);
        op.hcy_memory.push_back(dst_iter_memory_0);
        workptr = dst_iter_0 + D * nstates * N * H;
        

        //  next L - 1 layers
        auto user_src_layer_md = mkldnn::memory::desc(
            { src_layer_tz }, mkldnn_dtype, mkldnn::memory::format::tnc);
        auto user_src_layer_memory = mkldnn::memory({ user_src_layer_md, cpu_engine });
        op.x_memory.push_back(user_src_layer_memory);

        if (L > 1 && D == 1) {
          mkldnn::memory::dims weights_layer_tz = {L - 1, 1, H, ngates, H};  //  ldigo
          mkldnn::memory::dims weights_iter_tz = {L - 1, 1, H, ngates, H};  //  ldigo
          mkldnn::memory::dims bias_tz = {L - 1, 1, ngates, H};
          auto user_weight_layer_md = mkldnn::memory::desc(
              { weights_layer_tz }, mkldnn_dtype, mkldnn::memory::format::ldigo);
          auto user_weight_iter_md = mkldnn::memory::desc(
              { weights_iter_tz }, mkldnn_dtype, mkldnn::memory::format::ldigo);
          auto user_bias_md = mkldnn::memory::desc({ bias_tz },
              mkldnn_dtype, mkldnn::memory::format::ldgo);

          DType* weight_layer_n = workptr;  //  (L - 1) * H * ngates * H
          auto user_weight_layer_memory_n
              = mkldnn::memory({ user_weight_layer_md, cpu_engine }, weight_layer_n);
          op.wx_memory.push_back(user_weight_layer_memory_n);

          DType* weight_iter_n = weight_layer_n +
              (L - 1) * H * ngates * H;  //  (L - 1) * H * ngates * H
          auto user_weight_iter_memory_n
              = mkldnn::memory({ user_weight_iter_md, cpu_engine }, weight_iter_n);
          op.wh_memory.push_back(user_weight_iter_memory_n);

          DType* bias_n = weight_iter_n + (L - 1) * H * ngates * H;  //  (L - 1) * ngates * H
          auto user_bias_memory_n =
              mkldnn::memory({ user_bias_md, cpu_engine }, bias_n);
          op.bias_memory.push_back(user_bias_memory_n);

          auto wx_md_n = mkldnn::memory::desc(
              { weights_layer_tz }, mkldnn_dtype, mkldnn::memory::format::ldgoi);
          DType* wx_n = bias_n + (L - 1) * ngates * H;  //  (L - 1) * ngates * H * H
          auto wx_memory_n =
              mkldnn::memory({ wx_md_n, cpu_engine }, wx_n);
          DType* wh_n = wx_n + (L - 1) * ngates * H * H;  //  (L - 1) * ngates * H * H
          auto wh_md_n = mkldnn::memory::desc(
              { weights_iter_tz }, mkldnn_dtype, mkldnn::memory::format::ldgoi);
          auto wh_memory_n =
              mkldnn::memory({ wh_md_n, cpu_engine }, wh_n);

          op.concat_weight_memory.push_back(wx_memory_n);
          op.concat_weight_memory.push_back(wh_memory_n);
          workptr = wh_n + (L - 1) * ngates * H * H;

          mkldnn::memory::dims src_iter_tz_n1 = {1, 1, nstates, N, H};  //  ldsnc
          auto src_iter_md_n1 = mkldnn::memory::desc(
              { src_iter_tz_n1 }, mkldnn_dtype, mkldnn::memory::format::ldsnc);
          for (int l = 0; l < L - 1; l++) {
            DType* src_iter_n1 = workptr;  //  nstates * N * H
            auto src_iter_memory_n1 =
                mkldnn::memory({ src_iter_md_n1, cpu_engine }, src_iter_n1);
            op.concat_iter_memory.push_back(src_iter_memory_n1);
            workptr = src_iter_n1 + nstates * N * H;
          }
          mkldnn::memory::dims src_iter_tz_n = {L - 1, 1, nstates, N, H};  //  ldsnc
          auto src_iter_md_n = mkldnn::memory::desc(
              { src_iter_tz_n }, mkldnn_dtype, mkldnn::memory::format::ldsnc);
          DType* src_iter_n = workptr;  //  (L - 1) * nstates * N * H
          auto src_iter_memory_n =
              mkldnn::memory({ src_iter_md_n, cpu_engine }, src_iter_n);
          op.concat_iter_memory.push_back(src_iter_memory_n);
          op.hcx_memory.push_back(src_iter_memory_n);

          DType* dst_layer_n = src_iter_n + (L - 1) * nstates * N * H;  //  T * N * D * H
          auto dst_layer_memory_n
              = mkldnn::memory({ dst_layer_md, cpu_engine }, dst_layer_n);
          op.y_memory.push_back(dst_layer_memory_n);

          mkldnn::memory::dims dst_iter_tz_n = {L - 1, 1, nstates, N, H};  //  ldsnc
          auto dst_iter_md_n = mkldnn::memory::desc(
              { dst_iter_tz_n }, mkldnn_dtype, mkldnn::memory::format::ldsnc);
          DType* dst_iter_n = dst_layer_n + T * N * D * H;  //  (L - 1) * nstates * N * H
          auto dst_iter_memory_n =
              mkldnn::memory({ dst_iter_md_n, cpu_engine }, dst_iter_n);
          op.hcy_memory.push_back(dst_iter_memory_n);
        }

        if (L > 1 && D == 2) {
          mkldnn::memory::dims weights_layer_tz = {1, D, H * D, ngates, H};  //  ldigo
          mkldnn::memory::dims weights_iter_tz = {1, D, H, ngates, H};  //  ldigo
          mkldnn::memory::dims bias_tz = {1, D, ngates, H};
          auto user_weight_layer_md = mkldnn::memory::desc(
              { weights_layer_tz }, mkldnn_dtype, mkldnn::memory::format::ldigo);
          auto user_weight_iter_md = mkldnn::memory::desc(
              { weights_iter_tz }, mkldnn_dtype, mkldnn::memory::format::ldigo);
          auto user_bias_md = mkldnn::memory::desc({ bias_tz },
              mkldnn_dtype, mkldnn::memory::format::ldgo);

          auto wx_md_n = mkldnn::memory::desc(
              { weights_layer_tz }, mkldnn_dtype, mkldnn::memory::format::ldgoi);
          auto wh_md_n = mkldnn::memory::desc(
              { weights_iter_tz }, mkldnn_dtype, mkldnn::memory::format::ldgoi);

          DType* weight_layer_n = workptr;  //  D * (H * D) * ngates * H
          auto user_weight_layer_memory_n
              = mkldnn::memory({ user_weight_layer_md, cpu_engine }, weight_layer_n);
          op.wx_memory.push_back(user_weight_layer_memory_n);

          DType* weight_iter_n = weight_layer_n +
              D * (H * D) * ngates * H;  //  D * H * ngates * H
          auto user_weight_iter_memory_n
              = mkldnn::memory({ user_weight_iter_md, cpu_engine }, weight_iter_n);
          op.wh_memory.push_back(user_weight_iter_memory_n);

          DType* bias_n = weight_iter_n + D * H * ngates * H;  //  D * ngates * H
          auto user_bias_memory_n =
              mkldnn::memory({ user_bias_md, cpu_engine }, bias_n);
          op.bias_memory.push_back(user_bias_memory_n);

          DType* wx_n = bias_n + D * ngates * H;  //  D * ngates * (D * H) * H
          DType* wh_n = wx_n + D * ngates * (D * H) * H;  //  D * ngates * H * H
          auto wx_memory_n =
              mkldnn::memory({ wx_md_n, cpu_engine }, wx_n);
          auto wh_memory_n =
              mkldnn::memory({ wh_md_n, cpu_engine }, wh_n);
          op.concat_weight_memory.push_back(wx_memory_n);
          op.concat_weight_memory.push_back(wh_memory_n);

          mkldnn::memory::dims src_iter_undi_tz = {1, 1, nstates, N, H};  //  ldsnc
          auto src_iter_undi_md = mkldnn::memory::desc(
              { src_iter_undi_tz }, mkldnn_dtype, mkldnn::memory::format::ldsnc);
          DType* src_iter_undi = wh_n + D * ngates * H * H;  //  nstates * N * H
          auto src_iter_undi_memory =
              mkldnn::memory({ src_iter_undi_md, cpu_engine }, src_iter_undi);
          op.concat_iter_memory.push_back(src_iter_undi_memory_0);

          DType* src_iter_undi2 = src_iter_undi + nstates * N * H;  //  nstates * N * H
          auto src_iter_undi2_memory =
              mkldnn::memory({ src_iter_undi_md, cpu_engine }, src_iter_undi2);
          op.concat_iter_memory.push_back(src_iter_undi2_memory);

          mkldnn::memory::dims src_iter_tz = {1, D, nstates, N, H};  //  ldsnc
          auto src_iter_md = mkldnn::memory::desc(
              { src_iter_tz }, mkldnn_dtype, mkldnn::memory::format::ldsnc);
          DType* src_iter = src_iter_undi2 + nstates * N * H;  //  D * nstates * N * H
          auto src_iter_memory =
              mkldnn::memory({ src_iter_md, cpu_engine }, src_iter);
          op.concat_iter_memory.push_back(src_iter_memory);
          op.hcx_memory.push_back(src_iter_memory);

          DType* dst_layer_n = src_iter + D * nstates * N * H;  //  T * N * D * H
          auto dst_layer_memory_n
              = mkldnn::memory({ dst_layer_md, cpu_engine }, dst_layer_n);
          op.y_memory.push_back(dst_layer_memory_n);

          mkldnn::memory::dims dst_iter_tz_n = {1, D, nstates, N, H};  //  ldsnc
          auto dst_iter_md_n = mkldnn::memory::desc(
              { dst_iter_tz_n }, mkldnn_dtype, mkldnn::memory::format::ldsnc);
          DType* dst_iter_n = dst_layer_n + T * N * D * H;  //  D * nstates * N * H
          auto dst_iter_memory_n =
              mkldnn::memory({ dst_iter_md_n, cpu_engine }, dst_iter_n);
          op.hcy_memory.push_back(dst_iter_memory_n);
        }
      }
    }

    op.Forward(ctx, in_blobs, req, out_blobs);
  });
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_USE_MKLDNN == 1
#endif  // MXNET_OPERATOR_NN_MKLDNN_MKLDNN_RNN_IMPL_H_
