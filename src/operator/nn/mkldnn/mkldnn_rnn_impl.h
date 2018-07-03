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

template <typename DType>
void MKLDNNLstmForwardInference(DType* ws,
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
                                DType* cy_ptr) {
  const int b_size = 2 * H * 4 * D;
  const int cell_size = N * H * D;
  DType* y_cur_ptr = y_ptr;
  //  First layer
  int w_size = (I + H) * H * 4 * D;
  Tensor<cpu, 2, DType> x_0(x_ptr, Shape2(T * N, I));
  Tensor<cpu, 3, DType> y_0(y_cur_ptr, Shape3(T, N, H * D));
  DType* hx = hx_ptr;
  DType* cx = cx_ptr;
  DType* back_w_ptr = w_ptr;
  DType* back_b_ptr = b_ptr;
  const Tensor<cpu, 2, DType> wx_i_0(w_ptr, Shape2(H, I));
  const Tensor<cpu, 2, DType> wh_i_0(w_ptr + I * H * 4, Shape2(H, H));
  const Tensor<cpu, 2, DType> wx_f_0(w_ptr + H * I, Shape2(H, I));
  const Tensor<cpu, 2, DType> wh_f_0(w_ptr + I * H * 4 + H * H, Shape2(H, H));
  const Tensor<cpu, 2, DType> wx_g_0(w_ptr + H * I * 2, Shape2(H, I));
  const Tensor<cpu, 2, DType> wh_g_0(w_ptr + I * H * 4 + H * H * 2, Shape2(H, H));
  const Tensor<cpu, 2, DType> wx_o_0(w_ptr + H * I * 3, Shape2(H, I));
  const Tensor<cpu, 2, DType> wh_o_0(w_ptr + I * H * 4 + H * H * 3, Shape2(H, H));
  if (D == 2) {
    back_w_ptr = w_ptr + 4 * H * (I + H);
    back_b_ptr = b_ptr + 4 * H * 2;
  }
  const Tensor<cpu, 2, DType> back_wx_i_0(back_w_ptr , Shape2(H, I));
  const Tensor<cpu, 2, DType> back_wh_i_0(back_w_ptr + I * H * 4, Shape2(H, H));
  const Tensor<cpu, 2, DType> back_wx_f_0(back_w_ptr + H * I, Shape2(H, I));
  const Tensor<cpu, 2, DType> back_wh_f_0(back_w_ptr + I * H * 4 + H * H, Shape2(H, H));
  const Tensor<cpu, 2, DType> back_wx_g_0(back_w_ptr + H * I * 2, Shape2(H, I));
  const Tensor<cpu, 2, DType> back_wh_g_0(back_w_ptr + I * H * 4 + H * H * 2, Shape2(H, H));
  const Tensor<cpu, 2, DType> back_wx_o_0(back_w_ptr + H * I * 3, Shape2(H, I));
  const Tensor<cpu, 2, DType> back_wh_o_0(back_w_ptr + I * H * 4 + H * H * 3, Shape2(H, H));

  const Tensor<cpu, 2, DType> bx_0(b_ptr, Shape2(4, H));
  const Tensor<cpu, 2, DType> bh_0(b_ptr + H * 4, Shape2(4, H));
  const Tensor<cpu, 2, DType> back_bx_0(back_b_ptr, Shape2(4, H));
  const Tensor<cpu, 2, DType> back_bh_0(back_b_ptr + H * 4, Shape2(4, H));
  const int omp_threads = mxnet::engine::OpenMP::Get()->GetRecommendedOMPThreadCount();

  using namespace mkldnn;
  auto cpu_engine = CpuEngine::Get()->get_engine();
  auto null_memory_ = null_memory(cpu_engine);
  std::vector<primitive> rnn_net;

  memory::dims src_layer_tz_0 = {T, N, I};
  memory::dims weights_layer_tz_0 = {1, D, I, 4, H};  //  ldigo
  memory::dims weights_iter_tz_0 = {1, D, H, 4, H};  //  ldigo
  memory::dims bias_tz_0 = {1, D, 4, H};
  memory::dims dst_layer_tz_0 = {T, N, D * H};
  memory::dims src_iter_tz_0 = {1, D, 2, N, H};  //  ldsnc
  memory::dims dst_iter_tz_0 = {1, D, 2, N, H};  //  ldsnc

  std::vector<float> net_src_0(N * T * I, 0.0f);
  #pragma omp parallel for num_threads(omp_threads)
  for (int i = 0; i < N * T * I; i++) {
    net_src_0[i] = x_0.dptr_[i];
  }
  std::vector<float> net_src_iter_0(D * 2 * N * H, 0.0f);
  #pragma omp parallel for num_threads(omp_threads)
  for (int i = 0; i < N * H; i++) {
    net_src_iter_0[i] = hx[i];
    net_src_iter_0[i + N * H] = cx[i];
  }
  if (D == 2) {
    #pragma omp parallel for num_threads(omp_threads)
    for (int i = 0; i < N * H; i++) {
      net_src_iter_0[i + 2 * N * H] = hx[i + N * H];
      net_src_iter_0[i + 3 * N * H] = cx[i + N * H];
    }
  }

  std::vector<float> user_wei_layer_0(D * I * 4 * H, 0.0f);
  //  MKLDNN order is fiog, wx, wh, bx, bh order is ifgo
  #pragma omp parallel for num_threads(omp_threads)
  for (int i = 0; i < I; i++) {
    for (int j = 0; j < H; j++) {
      user_wei_layer_0[i * 4 * H + j] = wx_f_0[j][i];
      user_wei_layer_0[i * 4 * H + H + j] = wx_i_0[j][i];
      user_wei_layer_0[i * 4 * H + 2 * H + j] = wx_o_0[j][i];
      user_wei_layer_0[i * 4 * H + 3 * H + j] = wx_g_0[j][i];
    }
  }
  if (D == 2) {
    #pragma omp parallel for num_threads(omp_threads)
    for (int i = 0; i < I; i++) {
      for (int j = 0; j < H; j++) {
        user_wei_layer_0[I * 4 * H + i * 4 * H + j] = back_wx_f_0[j][i];
        user_wei_layer_0[I * 4 * H + i * 4 * H + H + j] = back_wx_i_0[j][i];
        user_wei_layer_0[I * 4 * H + i * 4 * H + 2 * H + j] = back_wx_o_0[j][i];
        user_wei_layer_0[I * 4 * H + i * 4 * H + 3 * H + j] = back_wx_g_0[j][i];
      }
    }
  }
  std::vector<float> user_wei_iter_0(D * H * 4 * H, 0.0f);
  //  MKLDNN order is fiog, wx, wh, bx, bh order is ifgo
  #pragma omp parallel for num_threads(omp_threads)
  for (int i = 0; i < H; i++) {
    for (int j = 0; j < H; j++) {
      user_wei_iter_0[i * 4 * H + j] = wh_f_0[j][i];
      user_wei_iter_0[i * 4 * H + H + j] = wh_i_0[j][i];
      user_wei_iter_0[i * 4 * H + 2 * H + j] = wh_o_0[j][i];
      user_wei_iter_0[i * 4 * H + 3 * H + j] = wh_g_0[j][i];
    }
  }
  if (D == 2) {
    #pragma omp parallel for num_threads(omp_threads)
    for (int i = 0; i < H; i++) {
      for (int j = 0; j < H; j++) {
        user_wei_iter_0[H * 4 * H + i * 4 * H + j] = back_wh_f_0[j][i];
        user_wei_iter_0[H * 4 * H + i * 4 * H + H + j] = back_wh_i_0[j][i];
        user_wei_iter_0[H * 4 * H + i * 4 * H + 2 * H + j] = back_wh_o_0[j][i];
        user_wei_iter_0[H * 4 * H + i * 4 * H + 3 * H + j] = back_wh_g_0[j][i];
      }
    }
  }
  std::vector<float> user_bias_0(D * 4 * H, 0.0f);
  //  MKLDNN order is fiog, wx, wh, bx, bh order is ifgo
  #pragma omp parallel for num_threads(omp_threads)
  for (int j = 0; j < H; j++) {
    user_bias_0[j] = bx_0[1][j] + bh_0[1][j];
    user_bias_0[H + j] = bx_0[0][j] + bh_0[0][j];
    user_bias_0[2 * H + j] = bx_0[3][j] + bh_0[3][j];
    user_bias_0[3 * H + j] = bx_0[2][j] + bh_0[2][j];
  }
  if (D == 2) {
    #pragma omp parallel for num_threads(omp_threads)
    for (int j = 0; j < H; j++) {
      user_bias_0[4 * H + j] = back_bx_0[1][j] + back_bh_0[1][j];
      user_bias_0[4 * H + H + j] = back_bx_0[0][j] + back_bh_0[0][j];
      user_bias_0[4 * H + 2 * H + j] = back_bx_0[3][j] + back_bh_0[3][j];
      user_bias_0[4 * H + 3 * H + j] = back_bx_0[2][j] + back_bh_0[2][j];
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

  rnn_cell::desc cell_0(algorithm::vanilla_lstm);
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
  if (L == 1) {
    #pragma omp parallel for num_threads(omp_threads)
    for (size_t n = 0; n < T * N * H * D; n++) {
      y_0.dptr_[n] = dst_y_0[n];
    }
  }
  if (state_outputs) {
    #pragma omp parallel for num_threads(omp_threads)
    for (size_t n = 0; n < N * H; n++) {
        hy_ptr[n] = dst_hcy_0[n];
        cy_ptr[n] = dst_hcy_0[n + N * H];
    }
    if (D == 2) {
      #pragma omp parallel for num_threads(omp_threads)
      for (size_t n = 0; n < N * H; n++) {
        hy_ptr[n + N * H] = dst_hcy_0[n + 2 * N * H];
        cy_ptr[n + N * H] = dst_hcy_0[n + 3 * N * H];
      }
    }
  }
  if (L > 1) {  //  go to next L - 1 layers
    w_ptr += w_size;
    b_ptr += b_size;
    if (state_outputs) {
      hy_ptr += cell_size;
      cy_ptr += cell_size;
    }
    I = H * D;
    w_size = (I + H) * H * 4 * D;
    Tensor<cpu, 3, DType> y(y_cur_ptr, Shape3(T, N, H * D));

    hx = hx_ptr + D * N * H;
    cx = cx_ptr + D * N * H;

    memory::dims src_layer_tz = {T, N, I};
    memory::dims weights_layer_tz = {L - 1, D, I, 4, H};  //  ldigo
    memory::dims weights_iter_tz = {L - 1, D, H, 4, H};  //  ldigo
    memory::dims bias_tz = {L - 1, D, 4, H};
    memory::dims dst_layer_tz = {T, N, D * H};
    memory::dims src_iter_tz = {L - 1, D, 2, N, H};  //  ldsnc
    memory::dims dst_iter_tz = {L - 1, D, 2, N, H};  //  ldsnc

    std::vector<float> net_src_iter((L - 1) * D * 2 * N * H, 0.0f);
    std::vector<float> user_wei_layer((L - 1) * D * I * 4 * H, 0.0f);
    std::vector<float> user_wei_iter((L - 1) * D * H * 4 * H, 0.0f);
    std::vector<float> user_bias((L - 1) * D * 4 * H, 0.0f);
    for (int l = 0; l < L - 1; l++) {
      #pragma omp parallel for num_threads(omp_threads)
      for (int i = 0; i < N * H; i++) {
        net_src_iter[l * D * 2 * N * H + i] = hx[l * D * N * H + i];
        net_src_iter[l * D * 2 * N * H + i + N * H] = cx[l * D * N * H + i];
      }
      if (D == 2) {
        #pragma omp parallel for num_threads(omp_threads)
        for (int i = 0; i < N * H; i++) {
          net_src_iter[l * D * 2 * N * H + i + 2 * N * H] = hx[l * D * N * H + i + N * H];
          net_src_iter[l * D * 2 * N * H + i + 3 * N * H] = cx[l * D * N * H + i + N * H];
        }
      }
      back_w_ptr = w_ptr;
      back_b_ptr = b_ptr;
      const Tensor<cpu, 2, DType> wx_i(w_ptr, Shape2(H, I));
      const Tensor<cpu, 2, DType> wh_i(w_ptr + I * H * 4, Shape2(H, H));
      const Tensor<cpu, 2, DType> wx_f(w_ptr + H * I, Shape2(H, I));
      const Tensor<cpu, 2, DType> wh_f(w_ptr + I * H * 4 + H * H, Shape2(H, H));
      const Tensor<cpu, 2, DType> wx_g(w_ptr + H * I * 2, Shape2(H, I));
      const Tensor<cpu, 2, DType> wh_g(w_ptr + I * H * 4 + H * H * 2, Shape2(H, H));
      const Tensor<cpu, 2, DType> wx_o(w_ptr + H * I * 3, Shape2(H, I));
      const Tensor<cpu, 2, DType> wh_o(w_ptr + I * H * 4 + H * H * 3, Shape2(H, H));
      if (D == 2) {
        back_w_ptr = w_ptr + 4 * H * (I + H);
        back_b_ptr = b_ptr + 4 * H * 2;
      }
      const Tensor<cpu, 2, DType> back_wx_i(back_w_ptr , Shape2(H, I));
      const Tensor<cpu, 2, DType> back_wh_i(back_w_ptr + I * H * 4, Shape2(H, H));
      const Tensor<cpu, 2, DType> back_wx_f(back_w_ptr + H * I, Shape2(H, I));
      const Tensor<cpu, 2, DType> back_wh_f(back_w_ptr + I * H * 4 + H * H, Shape2(H, H));
      const Tensor<cpu, 2, DType> back_wx_g(back_w_ptr + H * I * 2, Shape2(H, I));
      const Tensor<cpu, 2, DType> back_wh_g(back_w_ptr + I * H * 4 + H * H * 2, Shape2(H, H));
      const Tensor<cpu, 2, DType> back_wx_o(back_w_ptr + H * I * 3, Shape2(H, I));
      const Tensor<cpu, 2, DType> back_wh_o(back_w_ptr + I * H * 4 + H * H * 3, Shape2(H, H));

      const Tensor<cpu, 2, DType> bx(b_ptr, Shape2(4, H));
      const Tensor<cpu, 2, DType> bh(b_ptr + H * 4, Shape2(4, H));
      const Tensor<cpu, 2, DType> back_bx(back_b_ptr, Shape2(4, H));
      const Tensor<cpu, 2, DType> back_bh(back_b_ptr + H * 4, Shape2(4, H));

      //  MKLDNN order is fiog, wx, wh, bx, bh order is ifgo
      #pragma omp parallel for num_threads(omp_threads)
      for (int i = 0; i < I; i++) {
        for (int j = 0; j < H; j++) {
          user_wei_layer[l * D * I * 4 * H + i * 4 * H + j] = wx_f[j][i];
          user_wei_layer[l * D * I * 4 * H + i * 4 * H + H + j] = wx_i[j][i];
          user_wei_layer[l * D * I * 4 * H + i * 4 * H + 2 * H + j] = wx_o[j][i];
          user_wei_layer[l * D * I * 4 * H + i * 4 * H + 3 * H + j] = wx_g[j][i];
        }
      }
      if (D == 2) {
        #pragma omp parallel for num_threads(omp_threads)
        for (int i = 0; i < I; i++) {
          for (int j = 0; j < H; j++) {
            user_wei_layer[l * D * I * 4 * H + I * 4 * H + i * 4 * H + j] = back_wx_f[j][i];
            user_wei_layer[l * D * I * 4 * H + I * 4 * H + i * 4 * H + H + j] = back_wx_i[j][i];
            user_wei_layer[l * D * I * 4 * H + I * 4 * H + i * 4 * H + 2 * H + j] = back_wx_o[j][i];
            user_wei_layer[l * D * I * 4 * H + I * 4 * H + i * 4 * H + 3 * H + j] = back_wx_g[j][i];
          }
        }
      }
      //  MKLDNN order is fiog, wx, wh, bx, bh order is ifgo
      #pragma omp parallel for num_threads(omp_threads)
      for (int i = 0; i < H; i++) {
        for (int j = 0; j < H; j++) {
          user_wei_iter[l * D * H * 4 * H + i * 4 * H + j] = wh_f[j][i];
          user_wei_iter[l * D * H * 4 * H + i * 4 * H + H + j] = wh_i[j][i];
          user_wei_iter[l * D * H * 4 * H + i * 4 * H + 2 * H + j] = wh_o[j][i];
          user_wei_iter[l * D * H * 4 * H + i * 4 * H + 3 * H + j] = wh_g[j][i];
        }
      }
      if (D == 2) {
        #pragma omp parallel for num_threads(omp_threads)
        for (int i = 0; i < H; i++) {
          for (int j = 0; j < H; j++) {
            user_wei_iter[l * D * H * 4 * H + H * 4 * H + i * 4 * H + j] = back_wh_f[j][i];
            user_wei_iter[l * D * H * 4 * H + H * 4 * H + i * 4 * H + H + j] = back_wh_i[j][i];
            user_wei_iter[l * D * H * 4 * H + H * 4 * H + i * 4 * H + 2 * H + j] = back_wh_o[j][i];
            user_wei_iter[l * D * H * 4 * H + H * 4 * H + i * 4 * H + 3 * H + j] = back_wh_g[j][i];
          }
        }
      }
      //  MKLDNN order is fiog, wx, wh, bx, bh order is ifgo
      #pragma omp parallel for num_threads(omp_threads)
      for (int j = 0; j < H; j++) {
        user_bias[l * D * H * 4 + j] = bx[1][j] + bh[1][j];
        user_bias[l * D * H * 4 + H + j] = bx[0][j] + bh[0][j];
        user_bias[l * D * H * 4 + 2 * H + j] = bx[3][j] + bh[3][j];
        user_bias[l * D * H * 4 + 3 * H + j] = bx[2][j] + bh[2][j];
      }
      if (D == 2) {
        #pragma omp parallel for num_threads(omp_threads)
        for (int j = 0; j < H; j++) {
          user_bias[l * D * H * 4 + 4 * H + j] = back_bx[1][j] + back_bh[1][j];
          user_bias[l * D * H * 4 + 4 * H + H + j] = back_bx[0][j] + back_bh[0][j];
          user_bias[l * D * H * 4 + 4 * H + 2 * H + j] = back_bx[3][j] + back_bh[3][j];
          user_bias[l * D * H * 4 + 4 * H + 3 * H + j] = back_bx[2][j] + back_bh[2][j];
        }
      }
      w_ptr = w_ptr + w_size;
      b_ptr = b_ptr + b_size;
    }
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

    rnn_cell::desc cell(algorithm::vanilla_lstm);
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
    rnn_net.push_back(
        rnn_forward(prim_desc, dst_layer_memory_0,
                    user_src_iter_memory, user_wei_layer_memory,
                    user_wei_iter_memory, user_bias_memory,
                    dst_layer_memory, dst_iter_memory, null_memory_));

    stream(stream::kind::eager).submit(rnn_net).wait();
    float* dst_y = reinterpret_cast<float *> (dst_layer_memory.get_data_handle());
    float* dst_hcy = reinterpret_cast<float *> (dst_iter_memory.get_data_handle());
    #pragma omp parallel for num_threads(omp_threads)
    for (size_t n = 0; n < T * N * H * D; n++) {
        y.dptr_[n] = dst_y[n];
    }
    if (state_outputs) {
      for (int l = 0; l < L - 1; l++) {
        #pragma omp parallel for num_threads(omp_threads)
        for (size_t n = 0; n < N * H; n++) {
          hy_ptr[l * D * N * H + n] = dst_hcy[l * D * 2 * N * H + n];
          cy_ptr[l * D * N * H + n] = dst_hcy[l * D * 2 * N * H + n + N * H];
        }
        if (D == 2) {
          #pragma omp parallel for num_threads(omp_threads)
          for (size_t n = 0; n < N * H; n++) {
            hy_ptr[l * D * N * H + n + N * H] = dst_hcy[l * D * 2 * N * H + n + 2 * N * H];
            cy_ptr[l * D * N * H + n + N * H] = dst_hcy[l * D * 2 * N * H + n + 3 * N * H];
          }
        }
      }
    }
  }
}

template <typename DType>
void MKLDNNGruForwardInference(DType* ws,
                               bool state_outputs,
                               int L,
                               int D,
                               const int T,
                               const int N,
                               int I,
                               const int H,
                               DType* x_ptr,
                               DType* hx_ptr,
                               DType* w_ptr,
                               DType* b_ptr,
                               DType* y_ptr,
                               DType* hy_ptr) {
  int ngates = 3;
  const int b_size = 2 * H * ngates * D;
  const int cell_size = N * H * D;
  DType* y_cur_ptr = y_ptr;
  //  First layer
  int w_size = (I + H) * H * ngates * D;
  Tensor<cpu, 2, DType> x_0(x_ptr, Shape2(T * N, I));
  Tensor<cpu, 3, DType> y_0(y_cur_ptr, Shape3(T, N, H * D));
  DType* hx = hx_ptr;
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
  memory::dims src_iter_tz_0 = {1, D, 1, N, H};  //  ldsnc
  memory::dims dst_iter_tz_0 = {1, D, 1, N, H};  //  ldsnc

  std::vector<float> net_src_0(N * T * I, 0.0f);
  #pragma omp parallel for num_threads(omp_threads)
  for (int i = 0; i < N * T * I; i++) {
    net_src_0[i] = x_0.dptr_[i];
  }
  std::vector<float> net_src_iter_0(D * 1 * N * H, 0.0f);
  #pragma omp parallel for num_threads(omp_threads)
  for (int i = 0; i < N * H; i++) {
    net_src_iter_0[i] = hx[i];
  }
  if (D == 2) {
    #pragma omp parallel for num_threads(omp_threads)
    for (int i = 0; i < N * H; i++) {
      net_src_iter_0[i + N * H] = hx[i + N * H];
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
      for (int j = 0; j < H; j++) {
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
  rnn_cell::desc cell_0(algorithm::vanilla_gru);
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
  float* dst_hy_0 = reinterpret_cast<float *> (dst_iter_memory_0.get_data_handle());
  if (L == 1) {
    #pragma omp parallel for num_threads(omp_threads)
    for (size_t n = 0; n < T * N * H * D; n++) {
      y_0.dptr_[n] = dst_y_0[n];
    }
  }
  if (state_outputs) {
    #pragma omp parallel for num_threads(omp_threads)
    for (size_t n = 0; n < N * H; n++) {
        hy_ptr[n] = dst_hy_0[n];
    }
    if (D == 2) {
      #pragma omp parallel for num_threads(omp_threads)
      for (size_t n = 0; n < N * H; n++) {
        hy_ptr[n + N * H] = dst_hy_0[n + N * H];
      }
    }
  }
  if (L > 1) {  //  go to next L - 1 layers
    w_ptr += w_size;
    b_ptr += b_size;
    if (state_outputs) {
      hy_ptr += cell_size;
    }
    I = H * D;
    w_size = (I + H) * H * ngates * D;
    Tensor<cpu, 3, DType> y(y_cur_ptr, Shape3(T, N, H * D));

    hx = hx_ptr + D * N * H;

    memory::dims src_layer_tz = {T, N, I};
    memory::dims weights_layer_tz = {L - 1, D, I, ngates, H};  //  ldigo
    memory::dims weights_iter_tz = {L - 1, D, H, ngates, H};  //  ldigo
    memory::dims bias_tz = {L - 1, D, ngates, H};
    memory::dims dst_layer_tz = {T, N, D * H};
    memory::dims src_iter_tz = {L - 1, D, 1, N, H};  //  ldsnc
    memory::dims dst_iter_tz = {L - 1, D, 1, N, H};  //  ldsnc

    std::vector<float> net_src_iter((L - 1) * D * 1 * N * H, 0.0f);
    std::vector<float> user_wei_layer((L - 1) * D * I * ngates * H, 0.0f);
    std::vector<float> user_wei_iter((L - 1) * D * H * ngates * H, 0.0f);
    std::vector<float> user_bias((L - 1) * D * ngates * H, 0.0f);
    for (int l = 0; l < L - 1; l++) {
      #pragma omp parallel for num_threads(omp_threads)
      for (int i = 0; i < N * H; i++) {
        net_src_iter[l * D * 1 * N * H + i] = hx[l * D * N * H + i];
      }
      if (D == 2) {
        #pragma omp parallel for num_threads(omp_threads)
        for (int i = 0; i < N * H; i++) {
          net_src_iter[l * D * 1 * N * H + i + N * H] = hx[l * D * N * H + i + N * H];
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
          user_wei_layer[l * D * I * ngates * H + i * ngates * H + j] = wx[j][i];
        }
      }
      if (D == 2) {
        #pragma omp parallel for num_threads(omp_threads)
        for (int i = 0; i < I; i++) {
          for (int j = 0; j < ngates * H; j++) {
            user_wei_layer[l * D * I * ngates * H +
              I * ngates * H + i * ngates * H + j] = back_wx[j][i];
          }
        }
      }
      #pragma omp parallel for num_threads(omp_threads)
      for (int i = 0; i < H; i++) {
        for (int j = 0; j < ngates * H; j++) {
          user_wei_iter[l * D * H * ngates * H + i * ngates * H + j] = wh[j][i];
        }
      }
      if (D == 2) {
        #pragma omp parallel for num_threads(omp_threads)
        for (int i = 0; i < H; i++) {
          for (int j = 0; j < ngates * H; j++) {
            user_wei_iter[l * D * H * ngates * H
              + H * ngates * H + i * ngates * H + j] = back_wh[j][i];
          }
        }
      }
      #pragma omp parallel for num_threads(omp_threads)
      for (int j = 0; j < ngates * H; j++) {
        user_bias[l * D * H * ngates + j] = bx.dptr_[j] + bh.dptr_[j];
      }
      if (D == 2) {
        #pragma omp parallel for num_threads(omp_threads)
        for (int j = 0; j < ngates * H; j++) {
          user_bias[l * D * H * ngates + ngates * H + j] = back_bx.dptr_[j] + back_bh.dptr_[j];
        }
      }
      w_ptr = w_ptr + w_size;
      b_ptr = b_ptr + b_size;
    }
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

    rnn_cell::desc cell(algorithm::vanilla_gru);
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
    rnn_net.push_back(
        rnn_forward(prim_desc, dst_layer_memory_0,
                    user_src_iter_memory, user_wei_layer_memory,
                    user_wei_iter_memory, user_bias_memory,
                    dst_layer_memory, dst_iter_memory, null_memory_));

    stream(stream::kind::eager).submit(rnn_net).wait();
    float* dst_y = reinterpret_cast<float *> (dst_layer_memory.get_data_handle());
    float* dst_hy = reinterpret_cast<float *> (dst_iter_memory.get_data_handle());
    #pragma omp parallel for num_threads(omp_threads)
    for (size_t n = 0; n < T * N * H * D; n++) {
        y.dptr_[n] = dst_y[n];
    }
    if (state_outputs) {
      for (int l = 0; l < L - 1; l++) {
        #pragma omp parallel for num_threads(omp_threads)
        for (size_t n = 0; n < N * H; n++) {
          hy_ptr[l * D * N * H + n] = dst_hy[l * D * N * H + n];
        }
        if (D == 2) {
          #pragma omp parallel for num_threads(omp_threads)
          for (size_t n = 0; n < N * H; n++) {
            hy_ptr[l * D * N * H + n + N * H] = dst_hy[l * D * N * H + n + N * H];
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
