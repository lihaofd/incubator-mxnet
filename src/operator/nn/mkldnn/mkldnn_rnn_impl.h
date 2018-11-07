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
void MKLDNNRNNForwardInference(DType* ws,
                                bool state_outputs,
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
                                int mode) {
  int ngates;
  int ninputs;
  mkldnn::algorithm nalgorithm;
  switch (mode) {
    case mkldnn_rnn_enum::kLstm:
      ngates = 4;
      ninputs = 2;
      nalgorithm = algorithm::vanilla_lstm;
      break;
    default:
      LOG(FATAL) << "unknown RNN mode" << mode;
      break;
  }

  const int b_size = 2 * H * ngates * D;
  const int cell_size = N * H * D;
  //  First layer
  int w_size = (I + H) * H * ngates * D;
  Tensor<cpu, 2, DType> x_0(x_ptr, Shape2(T * N, I));
  Tensor<cpu, 3, DType> y(y_ptr, Shape3(T, N, H * D));
  DType* back_w_ptr = w_ptr;
  DType* back_b_ptr = b_ptr;
  const Tensor<cpu, 2, DType> wx_0(w_ptr, Shape2(ngates * H, I));
  const Tensor<cpu, 2, DType> wh_0(w_ptr + I * H * ngates, Shape2(ngates * H, H));

  if (D == 2) {
    back_w_ptr = w_ptr + ngates * H * (I + H);
    back_b_ptr = b_ptr + ngates * H * 2;
  }
  const Tensor<cpu, 2, DType> back_wx_0(back_w_ptr , Shape2(ngates * H, I));
  const Tensor<cpu, 2, DType> back_wh_0(back_w_ptr + I * H * ngates, Shape2(ngates * H, H));
  const Tensor<cpu, 2, DType> bx_0(b_ptr, Shape2(ngates, H));
  const Tensor<cpu, 2, DType> bh_0(b_ptr + H * ngates, Shape2(ngates, H));
  const Tensor<cpu, 2, DType> back_bx_0(back_b_ptr, Shape2(ngates, H));
  const Tensor<cpu, 2, DType> back_bh_0(back_b_ptr + H * ngates, Shape2(ngates, H));
  const int omp_threads = mxnet::engine::OpenMP::Get()->GetRecommendedOMPThreadCount();
  using namespace mkldnn;
  auto cpu_engine = CpuEngine::Get()->get_engine();
  auto null_memory_ = null_memory(cpu_engine);
  std::vector<primitive> rnn_net;

  memory::dims src_layer_tz_0 = {T, N, I};
  memory::dims weights_layer_tz_0 = {1, D, I, ngates, H};  //  ldigo
  memory::dims weights_iter_tz_0 = {1, D, H, ngates, H};  //  ldigo
  memory::dims bias_tz_0 = {1, D, ngates, H};
  memory::dims dst_layer_tz_0 = {T, N, D * H};
  memory::dims src_iter_tz_0 = {1, D, ninputs, N, H};  //  ldsnc
  memory::dims dst_iter_tz_0 = {1, D, ninputs, N, H};  //  ldsnc

  std::vector<float> net_src_0(N * T * I, 0.0f);
  #pragma omp parallel for num_threads(omp_threads)
  for (int i = 0; i < N * T * I; i++) {
    net_src_0[i] = x_0.dptr_[i];
  }
  std::vector<float> net_src_iter_0(D * ninputs * N * H, 0.0f);
  #pragma omp parallel for num_threads(omp_threads)
  for (int i = 0; i < N * H; i++) {
    net_src_iter_0[i] = hx_ptr[i];
  }
  if (mode == mkldnn_rnn_enum::kLstm) {
    #pragma omp parallel for num_threads(omp_threads)
    for (int i = 0; i < N * H; i++) {
      net_src_iter_0[i + N * H] = cx_ptr[i];
    }
  }
  if (D == 2) {
    #pragma omp parallel for num_threads(omp_threads)
    for (int i = 0; i < N * H; i++) {
      net_src_iter_0[i + ninputs * N * H] = hx_ptr[i + N * H];
    }
    if (mode == mkldnn_rnn_enum::kLstm) {
      #pragma omp parallel for num_threads(omp_threads)
      for (int i = 0; i < N * H; i++) {
        net_src_iter_0[i + (ninputs + 1) * N * H] = cx_ptr[i + N * H];
      }
    }
  }
  std::vector<float> user_wei_layer_0(D * I * ngates * H, 0.0f);
  #pragma omp parallel for num_threads(omp_threads)
  for (int i = 0; i < I; i++) {
    for (int j = 0; j < ngates * H; j++) {
      user_wei_layer_0[i * ngates * H + j] = wx_0[j][i];
    }
  }
  if (D == 2) {
    #pragma omp parallel for num_threads(omp_threads)
    for (int i = 0; i < I; i++) {
      for (int j = 0; j < ngates * H; j++) {
        user_wei_layer_0[I * ngates * H + i * ngates * H + j] = back_wx_0[j][i];
      }
    }
  }
  std::vector<float> user_wei_iter_0(D * H * ngates * H, 0.0f);
  #pragma omp parallel for num_threads(omp_threads)
  for (int i = 0; i < H; i++) {
    for (int j = 0; j < ngates * H; j++) {
      user_wei_iter_0[i * ngates * H + j] = wh_0[j][i];
    }
  }
  if (D == 2) {
    #pragma omp parallel for num_threads(omp_threads)
    for (int i = 0; i < H; i++) {
      for (int j = 0; j < ngates * H; j++) {
        user_wei_iter_0[H * ngates * H + i * ngates * H + j] = back_wh_0[j][i];
      }
    }
  }
  std::vector<float> user_bias_0(D * ngates * H, 0.0f);
  #pragma omp parallel for num_threads(omp_threads)
  for (int j = 0; j < ngates * H; j++) {
    user_bias_0[j] = bx_0.dptr_[j] + bh_0.dptr_[j];
  }
  if (D == 2) {
    #pragma omp parallel for num_threads(omp_threads)
    for (int j = 0; j < ngates * H; j++) {
      user_bias_0[ngates * H + j] = back_bx_0.dptr_[j] + back_bh_0.dptr_[j];
    }
  }

  auto user_src_layer_md_0 = mkldnn::memory::desc(
      { src_layer_tz_0 }, mkldnn::memory::data_type::f32,
      mkldnn::memory::format::tnc);

  auto user_src_iter_md_0 = mkldnn::memory::desc(
      { src_iter_tz_0 }, mkldnn::memory::data_type::f32,
      mkldnn::memory::format::ldsnc);

  auto user_wei_layer_md_0 = mkldnn::memory::desc(
      { weights_layer_tz_0 }, mkldnn::memory::data_type::f32,
      mkldnn::memory::format::ldigo);

  auto user_wei_iter_md_0 = mkldnn::memory::desc(
      { weights_iter_tz_0 }, mkldnn::memory::data_type::f32,
      mkldnn::memory::format::ldigo);

  auto user_bias_md_0 = mkldnn::memory::desc({ bias_tz_0 },
      mkldnn::memory::data_type::f32, mkldnn::memory::format::ldgo);

  auto dst_layer_md_0 = mkldnn::memory::desc(
      { dst_layer_tz_0 }, mkldnn::memory::data_type::f32,
      mkldnn::memory::format::tnc);

  auto dst_iter_md_0 = mkldnn::memory::desc(
      { dst_iter_tz_0 }, mkldnn::memory::data_type::f32,
      mkldnn::memory::format::ldsnc);

  auto user_src_layer_memory_0 = mkldnn::memory(
      { user_src_layer_md_0, cpu_engine }, net_src_0.data());

  auto user_wei_layer_memory_0
      = mkldnn::memory({ user_wei_layer_md_0, cpu_engine },
        user_wei_layer_0.data());

  auto user_wei_iter_memory_0
      = mkldnn::memory({ user_wei_iter_md_0, cpu_engine },
      user_wei_iter_0.data());

  auto user_bias_memory_0 = mkldnn::memory(
      { user_bias_md_0, cpu_engine }, user_bias_0.data());

  auto user_src_iter_memory_0 = mkldnn::memory(
      { user_src_iter_md_0, cpu_engine }, net_src_iter_0.data());

  rnn_cell::desc cell_0(nalgorithm);
  rnn_forward::desc layer_desc_0(prop_kind::forward_inference, cell_0,
      D == 1 ? rnn_direction::unidirectional : rnn_direction::bidirectional_concat,
      user_src_layer_md_0, user_src_iter_md_0, user_wei_layer_md_0, user_wei_iter_md_0,
      user_bias_md_0, dst_layer_md_0, dst_iter_md_0);

  auto prim_desc_0
      = mkldnn::rnn_forward::primitive_desc(layer_desc_0, cpu_engine);

  auto dst_layer_memory_0
      = mkldnn::memory(prim_desc_0.dst_layer_primitive_desc());

  auto dst_iter_memory_0
      = mkldnn::memory(prim_desc_0.dst_iter_primitive_desc());

  rnn_net.push_back(
      rnn_forward(prim_desc_0, user_src_layer_memory_0,
                  user_src_iter_memory_0, user_wei_layer_memory_0,
                  user_wei_iter_memory_0, user_bias_memory_0,
                  dst_layer_memory_0, dst_iter_memory_0, null_memory_));

  stream(stream::kind::eager).submit(rnn_net).wait();
  float* dst_y_0 = reinterpret_cast<float *> (dst_layer_memory_0.get_data_handle());
  float* dst_hcy_0 = reinterpret_cast<float *> (dst_iter_memory_0.get_data_handle());
  auto user_src_layer_memory_l = dst_layer_memory_0;
  if (L == 1) {
    #pragma omp parallel for num_threads(omp_threads)
    for (size_t n = 0; n < T * N * H * D; n++) {
      y.dptr_[n] = dst_y_0[n];
    }
  }
  if (state_outputs) {
    #pragma omp parallel for num_threads(omp_threads)
    for (size_t n = 0; n < N * H; n++) {
      hy_ptr[n] = dst_hcy_0[n];
    }
    if (mode == mkldnn_rnn_enum::kLstm) {
      #pragma omp parallel for num_threads(omp_threads)
      for (size_t n = 0; n < N * H; n++) {
        cy_ptr[n] = dst_hcy_0[n + N * H];
      }
    }
    if (D == 2) {
      #pragma omp parallel for num_threads(omp_threads)
      for (size_t n = 0; n < N * H; n++) {
        hy_ptr[n + N * H] = dst_hcy_0[n + ninputs * N * H];
      }
      if (mode == mkldnn_rnn_enum::kLstm) {
        #pragma omp parallel for num_threads(omp_threads)
        for (size_t n = 0; n < N * H; n++) {
          cy_ptr[n + N * H] = dst_hcy_0[n + (ninputs + 1) * N * H];
        }
      }
    }
  }
  //  go to next L - 1 layers
  if (L > 1) {
    w_ptr += w_size;
    b_ptr += b_size;
    I = H * D;
    w_size = (I + H) * H * ngates * D;
    memory::dims src_layer_tz = {T, N, I};
    memory::dims weights_layer_tz = {1, D, I, ngates, H};  //  ldigo
    memory::dims weights_iter_tz = {1, D, H, ngates, H};  //  ldigo
    memory::dims bias_tz = {1, D, ngates, H};
    memory::dims dst_layer_tz = {T, N, D * H};
    memory::dims src_iter_tz = {1, D, ninputs, N, H};  //  ldsnc
    memory::dims dst_iter_tz = {1, D, ninputs, N, H};  //  ldsnc
    std::vector<float> net_src_iter(1 * D * ninputs * N * H);
    std::vector<float> user_wei_layer(1 * D * I * ngates * H);
    std::vector<float> user_wei_iter(1 * D * H * ngates * H);
    std::vector<float> user_bias(1 * D * ngates * H);

    auto user_src_layer_md = mkldnn::memory::desc(
        { src_layer_tz }, mkldnn::memory::data_type::f32,
        mkldnn::memory::format::tnc);

    auto user_src_iter_md = mkldnn::memory::desc(
        { src_iter_tz }, mkldnn::memory::data_type::f32,
        mkldnn::memory::format::ldsnc);

    auto user_wei_layer_md = mkldnn::memory::desc(
        { weights_layer_tz }, mkldnn::memory::data_type::f32,
        mkldnn::memory::format::ldigo);

    auto user_wei_iter_md = mkldnn::memory::desc(
        { weights_iter_tz }, mkldnn::memory::data_type::f32,
        mkldnn::memory::format::ldigo);

    auto user_bias_md = mkldnn::memory::desc({ bias_tz },
        mkldnn::memory::data_type::f32, mkldnn::memory::format::ldgo);

    auto dst_layer_md = mkldnn::memory::desc(
        { dst_layer_tz }, mkldnn::memory::data_type::f32,
        mkldnn::memory::format::tnc);

    auto dst_iter_md = mkldnn::memory::desc(
        { dst_iter_tz }, mkldnn::memory::data_type::f32,
         mkldnn::memory::format::ldsnc);

    rnn_cell::desc cell(nalgorithm);
    rnn_forward::desc layer_desc(prop_kind::forward_inference, cell,
        D == 1 ? rnn_direction::unidirectional : rnn_direction::bidirectional_concat,
        user_src_layer_md, user_src_iter_md, user_wei_layer_md, user_wei_iter_md,
        user_bias_md, dst_layer_md, dst_iter_md);

    auto prim_desc
        = mkldnn::rnn_forward::primitive_desc(layer_desc, cpu_engine);

    auto dst_layer_memory
        = mkldnn::memory(prim_desc.dst_layer_primitive_desc());

    auto dst_iter_memory
        = mkldnn::memory(prim_desc.dst_iter_primitive_desc());

    for (int l = 0; l < L - 1; l++) {
      std::vector<primitive> rnn_net2;
      if (state_outputs) {
        hy_ptr += cell_size;
        cy_ptr += cell_size;
      }
      hx_ptr += cell_size;
      if (mode == mkldnn_rnn_enum::kLstm) {
        cx_ptr += cell_size;
      }

      #pragma omp parallel for num_threads(omp_threads)
      for (int i = 0; i < N * H; i++) {
        net_src_iter[i] = hx_ptr[i];
      }
      if (mode == mkldnn_rnn_enum::kLstm) {
        #pragma omp parallel for num_threads(omp_threads)
        for (int i = 0; i < N * H; i++) {
          net_src_iter[i + N * H] = cx_ptr[i];
        }
      }
      if (D == 2) {
        #pragma omp parallel for num_threads(omp_threads)
        for (int i = 0; i < N * H; i++) {
          net_src_iter[i + ninputs * N * H] = hx_ptr[i + N * H];
        }
        if (mode == mkldnn_rnn_enum::kLstm) {
          #pragma omp parallel for num_threads(omp_threads)
          for (int i = 0; i < N * H; i++) {
            net_src_iter[i + (ninputs + 1) * N * H] = cx_ptr[i + N * H];
          }
        }
      }
      back_w_ptr = w_ptr;
      back_b_ptr = b_ptr;
      const Tensor<cpu, 2, DType> wx(w_ptr, Shape2(ngates * H, I));
      const Tensor<cpu, 2, DType> wh(w_ptr + I * H * ngates, Shape2(ngates * H, H));
      if (D == 2) {
        back_w_ptr = w_ptr + ngates * H * (I + H);
        back_b_ptr = b_ptr + ngates * H * 2;
      }
      const Tensor<cpu, 2, DType> back_wx(back_w_ptr , Shape2(ngates * H, I));
      const Tensor<cpu, 2, DType> back_wh(back_w_ptr + I * H * ngates, Shape2(ngates * H, H));
      const Tensor<cpu, 2, DType> bx(b_ptr, Shape2(ngates, H));
      const Tensor<cpu, 2, DType> bh(b_ptr + H * ngates, Shape2(ngates, H));
      const Tensor<cpu, 2, DType> back_bx(back_b_ptr, Shape2(ngates, H));
      const Tensor<cpu, 2, DType> back_bh(back_b_ptr + H * ngates, Shape2(ngates, H));
      #pragma omp parallel for num_threads(omp_threads)
      for (int i = 0; i < I; i++) {
        for (int j = 0; j < ngates * H; j++) {
          user_wei_layer[i * ngates * H + j] = wx[j][i];
        }
      }
      if (D == 2) {
        #pragma omp parallel for num_threads(omp_threads)
        for (int i = 0; i < I; i++) {
          for (int j = 0; j < ngates * H; j++) {
            user_wei_layer[I * ngates * H + i * ngates * H + j] = back_wx[j][i];
          }
        }
      }
      #pragma omp parallel for num_threads(omp_threads)
      for (int i = 0; i < H; i++) {
        for (int j = 0; j < ngates * H; j++) {
          user_wei_iter[i * ngates * H + j] = wh[j][i];
        }
      }
      if (D == 2) {
        #pragma omp parallel for num_threads(omp_threads)
        for (int i = 0; i < H; i++) {
          for (int j = 0; j < ngates * H; j++) {
            user_wei_iter[H * ngates * H + i * ngates * H + j] = back_wh[j][i];
          }
        }
      }
      #pragma omp parallel for num_threads(omp_threads)
      for (int j = 0; j < ngates * H; j++) {
        user_bias[j] = bx.dptr_[j] + bh.dptr_[j];
      }
      if (D == 2) {
        #pragma omp parallel for num_threads(omp_threads)
        for (int j = 0; j < ngates * H; j++) {
          user_bias[ngates * H + j] = back_bx.dptr_[j] + back_bh.dptr_[j];
        }
      }
      w_ptr += w_size;
      b_ptr += b_size;

      auto user_wei_layer_memory
          = mkldnn::memory({ user_wei_layer_md, cpu_engine },
          user_wei_layer.data());

      auto user_wei_iter_memory
          = mkldnn::memory({ user_wei_iter_md, cpu_engine },
          user_wei_iter.data());

      auto user_bias_memory = mkldnn::memory(
          { user_bias_md, cpu_engine }, user_bias.data());

      auto user_src_iter_memory = mkldnn::memory(
          { user_src_iter_md, cpu_engine }, net_src_iter.data());      

      rnn_net2.push_back(
          rnn_forward(prim_desc, user_src_layer_memory_l,
                      user_src_iter_memory, user_wei_layer_memory,
                      user_wei_iter_memory, user_bias_memory,
                      dst_layer_memory, dst_iter_memory, null_memory_));

      stream(stream::kind::eager).submit(rnn_net2).wait();

      float* dst_y = reinterpret_cast<float *> (dst_layer_memory.get_data_handle());
      float* dst_hcy = reinterpret_cast<float *> (dst_iter_memory.get_data_handle());
      user_src_layer_memory_l = dst_layer_memory;
      if (l == L - 2) {
        #pragma omp parallel for num_threads(omp_threads)
        for (size_t n = 0; n < T * N * H * D; n++) {
          y.dptr_[n] = dst_y[n];
        }
      }
      if (state_outputs) {
        #pragma omp parallel for num_threads(omp_threads)
        for (size_t n = 0; n < N * H; n++) {
          hy_ptr[n] = dst_hcy[n];
        }
        if (mode == mkldnn_rnn_enum::kLstm) {
          #pragma omp parallel for num_threads(omp_threads)
          for (size_t n = 0; n < N * H; n++) {
            cy_ptr[n] = dst_hcy[n + N * H];
          }
        }
        if (D == 2) {
          #pragma omp parallel for num_threads(omp_threads)
          for (size_t n = 0; n < N * H; n++) {
            hy_ptr[n + N * H] = dst_hcy[n + ninputs * N * H];
          }
          if (mode == mkldnn_rnn_enum::kLstm) {
            #pragma omp parallel for num_threads(omp_threads)
              for (size_t n = 0; n < N * H; n++) {
              cy_ptr[n + N * H] = dst_hcy[n + (ninputs + 1) * N * H];
            }
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