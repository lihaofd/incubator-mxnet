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
                           std::vector<mkldnn::memory> *hcx_memory,
                           std::vector<mkldnn::memory> *wx_memory,
                           std::vector<mkldnn::memory> *wh_memory,
                           std::vector<mkldnn::memory> *bias_memory,
                           std::vector<mkldnn::memory> *y_memory,
                           std::vector<mkldnn::memory> *hcy_memory,
                           std::vector<primitive> *rnn_forward_prim,
                           bool has_nextlayer,
                           int dtype,
                           int mode) {
  using namespace mkldnn;
  int ngates = 0, nstates = 0;
  algorithm nalgorithm = GetMKLDNNRNNAlgo(mode, &ngates, &nstates);
  mkldnn::memory::data_type mkldnn_dtype = get_mkldnn_type(dtype);
  const int cell_size = N * H;
  const int single_cell_size = N * H;
  const int single_b_size = ngates * H;
  int w_size = (I + H) * H * ngates;

  const int omp_threads = mxnet::engine::OpenMP::Get()->GetRecommendedOMPThreadCount();
  auto cpu_engine = CpuEngine::Get()->get_engine();
  auto null_memory_ = null_memory(cpu_engine);
  int offset1 = 0, offset2 = 0;
  bool cached = false;
 
  const float data_shift = 64.;
  const float data_scale = 63.;

  const int weights_scale_mask = 3; // 11 for last two dimensions of ldigo
  std::vector<float> weights_scales(ngates * H);
  const dim_t scales_half = ngates * H / 2;
  std::fill(
          weights_scales.begin(), weights_scales.begin() + scales_half, 30.f);
  std::fill(weights_scales.begin() + scales_half + 1, weights_scales.end(), 65.5f);

  mkldnn::memory::dims src_layer_tz = {T, N, I};
  mkldnn::memory::dims dst_layer_tz = {T, N, H};
  mkldnn::memory::dims weights_layer_tz = {L, 1, I, ngates, H};  //  ldigo
  mkldnn::memory::dims weights_iter_tz = {L, 1, H, ngates, H};  //  ldigo
  mkldnn::memory::dims bias_tz = {L, 1, ngates, H};
  mkldnn::memory::dims src_iter_tz = {L, 1, nstates, N, H};  //  ldsnc
  mkldnn::memory::dims dst_iter_tz = {L, 1, nstates, N, H};  //  ldsnc
  mkldnn::memory::dims weights_layer_r_tz = {1, 1, I, ngates, H};  //  ldigo for reorder
  mkldnn::memory::dims weights_iter_r_tz = {1, 1, H, ngates, H};  //  ldigo for reorder

  auto src_layer_md = mkldnn::memory::desc(
      {src_layer_tz}, mkldnn_dtype, mkldnn::memory::format::tnc);
  auto dst_layer_md = mkldnn::memory::desc(
      {dst_layer_tz}, mkldnn_dtype, mkldnn::memory::format::tnc);
  auto src_iter_md = mkldnn::memory::desc(
      {src_iter_tz}, mkldnn_dtype, mkldnn::memory::format::ldsnc);
  auto weight_layer_md = mkldnn::memory::desc(
      {weights_layer_tz}, mkldnn_dtype, mkldnn::memory::format::ldigo);
  auto weight_iter_md = mkldnn::memory::desc(
      {weights_iter_tz}, mkldnn_dtype, mkldnn::memory::format::ldigo);
  auto bias_md = mkldnn::memory::desc({bias_tz},
      mkldnn_dtype, mkldnn::memory::format::ldgo);
  auto dst_iter_md = mkldnn::memory::desc(
      {dst_iter_tz}, mkldnn_dtype, mkldnn::memory::format::ldsnc);

  auto src_layer_memory = memory({src_layer_md, cpu_engine});
  src_layer_memory.set_data_handle(x_ptr);

  DType* net_src_iter_0 = new DType[nstates * N * H];
  #pragma omp parallel for num_threads(omp_threads)
  for (int i = 0; i < N * H; i++) {
    net_src_iter_0[i] = hx_ptr[i];
  }
  offset1 = N * H;
  #pragma omp parallel for num_threads(omp_threads)
  for (int i = 0; i < N * H; i++) {
    net_src_iter_0[i + offset1] = cx_ptr[i];
  }
  auto src_iter_memory = memory({src_iter_md, cpu_engine});
  src_iter_memory.set_data_handle(net_src_iter_0);

  auto weight_layer_memory = memory({weight_layer_md, cpu_engine});
  auto weight_iter_memory = memory({weight_iter_md, cpu_engine});
  auto b_memory = memory({bias_md, cpu_engine});

  auto mpd_x = mkldnn::memory::primitive_desc({weights_layer_r_tz,
        mkldnn_dtype, mkldnn::memory::format::ldgoi}, cpu_engine);
  auto src_wx_f = mkldnn::memory(mpd_x);
  auto mpd_h = mkldnn::memory::primitive_desc({weights_iter_r_tz,
        mkldnn_dtype, mkldnn::memory::format::ldgoi}, cpu_engine);
  auto src_wh_f = mkldnn::memory(mpd_h);
  std::vector<void*> srcs_data_x;
  std::vector<void*> srcs_data_h;
  std::vector<mkldnn::memory::dims> src_l_dim_x;
  std::vector<mkldnn::memory::dims> src_l_dim_h;
  if (!cached) {
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
      ConcatData(mkldnn::memory::format::ldgoi, mkldnn::memory::format::ldgoi,
          src_l_dim_x, weights_layer_tz, mkldnn_dtype, 0, srcs_data_x, src_wx_f);
      ConcatData(mkldnn::memory::format::ldgoi, mkldnn::memory::format::ldgoi,
          src_l_dim_h, weights_iter_tz, mkldnn_dtype, 0, srcs_data_h, src_wh_f);
    }
    ReorderData(src_wx_f, weight_layer_memory);
    ReorderData(src_wh_f, weight_iter_memory);
  }

  DType* user_bias_f = reinterpret_cast<DType *> (b_memory.get_data_handle());
  #pragma omp parallel for num_threads(omp_threads)
  for (int j = 0; j < L * single_b_size; j++) {
    int k = j / single_b_size;
    user_bias_f[j] = b_ptr[j + k * single_b_size] + b_ptr[j + k * single_b_size + single_b_size];
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
  std::vector<primitive> weights_reorders;

  auto dst_layer_memory = null_memory_;
  auto dst_layer_memory_u8 = null_memory_;
  auto dst_iter_memory = null_memory_;
  auto src_layer_memory_u8 = null_memory_;
  auto weight_layer_memory_s8 = null_memory_;
  auto weight_iter_memory_s8 = null_memory_;

  if (!has_nextlayer) {  //  last layer output
    rnn_forward::desc layer_desc(prop_kind::forward_inference, lstm_cell,
        rnn_direction::unidirectional, src_layer_md_u8,
        src_iter_md, weight_layer_md_s8, weight_iter_md_s8,
        bias_md, dst_layer_md, dst_iter_md); 

    auto prim_desc
         = rnn_forward::primitive_desc(layer_desc, attr, cpu_engine);
  
    src_layer_memory_u8
       = memory(prim_desc.src_layer_primitive_desc());
    auto src_layer_reorder_pd = reorder::primitive_desc(
       src_layer_memory.get_primitive_desc(),
       src_layer_memory_u8.get_primitive_desc(), attr);
    rnn_net.push_back(reorder(src_layer_reorder_pd,
             src_layer_memory, src_layer_memory_u8));

    weight_layer_memory_s8
            = memory(prim_desc.weights_layer_primitive_desc());
    auto weight_layer_reorder_pd = reorder::primitive_desc(
          weight_layer_memory.get_primitive_desc(),
          weight_layer_memory_s8.get_primitive_desc(), attr);
    weights_reorders.push_back(reorder(weight_layer_reorder_pd,
          weight_layer_memory, weight_layer_memory_s8));

    weight_iter_memory_s8
          = memory(prim_desc.weights_iter_primitive_desc());
    auto weight_iter_reorder_pd = reorder::primitive_desc(
          weight_iter_memory.get_primitive_desc(),
          weight_iter_memory_s8.get_primitive_desc(), attr);
    weights_reorders.push_back(reorder(weight_iter_reorder_pd,
          weight_iter_memory, weight_iter_memory_s8));

    dst_iter_memory
          = memory(prim_desc.dst_iter_primitive_desc());

    dst_layer_memory
          = memory(prim_desc.dst_layer_primitive_desc());
    dst_layer_memory.set_data_handle(y_ptr);
    rnn_net.push_back(
        rnn_forward(prim_desc, src_layer_memory_u8,
            src_iter_memory, weight_layer_memory_s8,
            weight_iter_memory_s8, b_memory,
            dst_layer_memory, dst_iter_memory, null_memory_));

  } else {
    rnn_forward::desc layer_desc(prop_kind::forward_inference, lstm_cell,
        rnn_direction::unidirectional, src_layer_md_u8,
        src_iter_md, weight_layer_md_s8, weight_iter_md_s8,
        bias_md, dst_layer_md_u8, dst_iter_md); 

    auto prim_desc
         = rnn_forward::primitive_desc(layer_desc, attr, cpu_engine);

    src_layer_memory_u8
       = memory(prim_desc.src_layer_primitive_desc());
    auto src_layer_reorder_pd = reorder::primitive_desc(
       src_layer_memory.get_primitive_desc(),
       src_layer_memory_u8.get_primitive_desc(), attr);
    rnn_net.push_back(reorder(src_layer_reorder_pd,
             src_layer_memory, src_layer_memory_u8));

    weight_layer_memory_s8
            = memory(prim_desc.weights_layer_primitive_desc());
    auto weight_layer_reorder_pd = reorder::primitive_desc(
          weight_layer_memory.get_primitive_desc(),
          weight_layer_memory_s8.get_primitive_desc(), attr);
    weights_reorders.push_back(reorder(weight_layer_reorder_pd,
          weight_layer_memory, weight_layer_memory_s8));

    weight_iter_memory_s8
          = memory(prim_desc.weights_iter_primitive_desc());
    auto weight_iter_reorder_pd = reorder::primitive_desc(
          weight_iter_memory.get_primitive_desc(),
          weight_iter_memory_s8.get_primitive_desc(), attr);
    weights_reorders.push_back(reorder(weight_iter_reorder_pd,
          weight_iter_memory, weight_iter_memory_s8));

    dst_iter_memory
          = memory(prim_desc.dst_iter_primitive_desc());

    dst_layer_memory_u8
          = memory(prim_desc.dst_layer_primitive_desc());
    rnn_net.push_back(
        rnn_forward(prim_desc, src_layer_memory_u8,
            src_iter_memory, weight_layer_memory_s8,
            weight_iter_memory_s8, b_memory,
            dst_layer_memory_u8, dst_iter_memory, null_memory_));
  }

/*
  int tmpnum = 3;
  DType* x = reinterpret_cast<DType *> (src_layer_memory.get_data_handle());
  uint8_t* x_int8 = reinterpret_cast<uint8_t *> (src_layer_memory_u8.get_data_handle());
  for (int i =0 ; i< tmpnum ;i++ ) {
    LOG(INFO) << "x[" << i << "]:" << x[i] << " x_int8[" << i << "]:" << (int)x_int8[i];
  }

  DType* w_x = reinterpret_cast<DType *> (weight_layer_memory.get_data_handle());
  int8_t* w_x_int8 = reinterpret_cast<int8_t *> (weight_layer_memory_s8.get_data_handle());
  for (int i =0 ; i< tmpnum ;i++ ) {
    LOG(INFO) << "w_x[" << i << "]:" << w_x[i] << " w_x_int8[" << i << "]:" << (int)w_x_int8[i];
  }

  DType* w_h = reinterpret_cast<DType *> (weight_iter_memory.get_data_handle());
  int8_t* w_h_int8 = reinterpret_cast<int8_t *> (weight_iter_memory_s8.get_data_handle());
  for (int i =0 ; i< tmpnum ;i++ ) {
    LOG(INFO) << "w_h[" << i << "]:" << w_h[i] << " w_h_int8[" << i << "]:" << (int)w_h_int8[i];
  }

  DType* y = reinterpret_cast<DType *> (dst_layer_memory.get_data_handle());
  for (int i =0 ; i< tmpnum ;i++ ) {
    LOG(INFO) << "y[" << i << "]:" << y[i];
  }
*/
  stream(stream::kind::eager).submit(weights_reorders).wait();
  stream(stream::kind::eager).submit(rnn_net).wait();

  if (state_outputs) {
    //DType* dst_hcy = reinterpret_cast<DType *> ((*hcy_memory)[layer_index].get_data_handle());
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

/*
  for (int i =0 ; i< tmpnum ;i++ ) {
    LOG(INFO) << "x[" << i << "]:" << x[i] << " x_int8[" << i << "]:" << (int)x_int8[i];
  }

  for (int i =0 ; i< tmpnum ;i++ ) {
    LOG(INFO) << "w_x[" << i << "]:" << w_x[i] << " w_x_int8[" << i << "]:" << (int)w_x_int8[i];
  }

  for (int i =0 ; i< tmpnum ;i++ ) {
    LOG(INFO) << "w_h[" << i << "]:" << w_h[i] << " w_h_int8[" << i << "]:" << (int)w_h_int8[i];
  }

  for (int i =0 ; i< tmpnum ;i++ ) {
    LOG(INFO) << "y[" << i << "]:" << y[i];
  }
*/
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
                          std::vector<mkldnn::memory> *hcx_memory,
                          std::vector<mkldnn::memory> *wx_memory,
                          std::vector<mkldnn::memory> *wh_memory,
                          std::vector<mkldnn::memory> *bias_memory,
                          std::vector<mkldnn::memory> *y_memory,
                          std::vector<mkldnn::memory> *hcy_memory,
                          std::vector<primitive> *rnn_forward_prim,
                          int dtype,
                          int mode) {
  int ngates = 0, nstates = 0;
  GetMKLDNNRNNAlgo(mode, &ngates, &nstates);
  const int b_size = 2 * H * ngates * D;
  const int cell_size = N * H * D;
  //  First layer
  int w_size = (I + H) * H * ngates * D;
  auto cpu_engine = CpuEngine::Get()->get_engine();
  auto null_memory_ = null_memory(cpu_engine);
  DType* tmpNull = NULL;
  
  // when D = 1 and I == H, L layers can be fused together
  if (D == 1 && I == H) {
    MKLDNNRNNForwardUnidi(state_outputs, L, T, N, I, H, x_ptr, null_memory_,
        hx_ptr, cx_ptr, w_ptr, b_ptr, y_ptr, hy_ptr, cy_ptr, concat_weight_memory,
        concat_iter_memory, x_memory, hcx_memory, wx_memory, wh_memory,
        bias_memory, y_memory, hcy_memory, rnn_forward_prim, false, dtype, mode);
  } else {
  }
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
                                   std::vector<mkldnn::memory> *hcx_memory,
                                   std::vector<mkldnn::memory> *wx_memory,
                                   std::vector<mkldnn::memory> *wh_memory,
                                   std::vector<mkldnn::memory> *bias_memory,
                                   std::vector<mkldnn::memory> *y_memory,
                                   std::vector<mkldnn::memory> *hcy_memory,
                                   std::vector<primitive> *rnn_forward_prim,
                                   int dtype,
                                   int mode) {
  switch (mode) {
    case rnn_enum::kLstm:
      MKLDNNRNNForwardINT8<DType>(state_outputs, num_layers, direction, seq_length,
                                  batch_size, input_size, state_size, x_ptr, hx_ptr,
                                  cx_ptr, w_ptr, b_ptr, y_ptr, hy_ptr, cy_ptr,
                                  concat_weight_memory, concat_iter_memory, x_memory,
                                  hcx_memory, wx_memory, wh_memory, bias_memory,
                                  y_memory, hcy_memory, rnn_forward_prim, dtype, mode);
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
                                           &hcx_memory,
                                           &wx_memory,
                                           &wh_memory,
                                           &bias_memory,
                                           &y_memory,
                                           &hcy_memory,
                                           &rnn_forward_prim,
                                           dtype,
                                           param_.mode);
    }
  }

  RNNParam param_;
  std::vector<mkldnn::memory> concat_weight_memory;
  std::vector<mkldnn::memory> concat_iter_memory;

  std::vector<primitive> rnn_forward_prim;
  std::vector<mkldnn::memory> x_memory;
  std::vector<mkldnn::memory> hcx_memory;
  std::vector<mkldnn::memory> wx_memory;
  std::vector<mkldnn::memory> wh_memory;
  std::vector<mkldnn::memory> bias_memory;
  std::vector<mkldnn::memory> y_memory;
  std::vector<mkldnn::memory> hcy_memory;
  bool init_mem_;
  size_t reserve_mem_size_;
  Storage::Handle mem_space_;
};  // class MKLDNNRNNOp

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
    op.Forward(ctx, in_blobs, req, out_blobs);
  });
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_USE_MKLDNN == 1
#endif  // MXNET_OPERATOR_NN_MKLDNN_MKLDNN_RNN_IMPL_H_
