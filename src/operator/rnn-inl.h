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

/*!
 * Copyright (c) 2015 by Contributors
 * \file rnn-inl.h
 * \brief
 * \author Sebastian Bodenstein, Shu Zhang
*/
#ifndef MXNET_OPERATOR_RNN_INL_H_
#define MXNET_OPERATOR_RNN_INL_H_

#define MXNET_USE_CUDNN_RNN MXNET_USE_CUDNN == 1 && CUDNN_MAJOR >= 5
#define USE_CUDNN_LSTM_PROJ MXNET_USE_CUDNN == 1 && CUDNN_VERSION >= 7200

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <mxnet/storage.h>
#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include <cstdint>
#include "./math.h"
#include "./math_functions-inl.h"
#include "./operator_common.h"
#include "./rnn_impl.h"
#if MXNET_USE_MKLDNN == 1
#include "./nn/mkldnn/mkldnn_rnn_impl.h"
#endif

namespace mxnet {
namespace op {

inline int GetRnnParamSize(int num_layer,
                           int input_size,
                           int state_size,
                           int direction,
                           int mode,
                           const dmlc::optional<int>& projection_size) {
  int size = state_size * direction;
  switch (mode) {
    case rnn_enum::kRnnRelu:
    case rnn_enum::kRnnTanh:
      break;
    case rnn_enum::kLstm:
      size *= 4;
      break;
    case rnn_enum::kGru:
      size *= 3;
      break;
  }
  int size1 = (input_size + state_size + 2) * size;  // first layer size
  int size2 = (state_size * direction + state_size + 2) * size;  // other layers size
  if (projection_size.has_value()) {
    int proj_size = projection_size.value();
    size1 = (input_size + proj_size + 2) * size;
    size2 = (proj_size * direction + proj_size + 2) * size;
  }
  int param_size = size1 + (num_layer - 1) * size2;
  if (projection_size.has_value()) {
    param_size += projection_size.value() * state_size * num_layer * direction;
  }
  return param_size;
}

inline int GetRnnBiasSize(int num_layer,
                           int state_size,
                           int direction,
                           int mode) {
  int size = 2 * state_size * direction * num_layer;
  switch (mode) {
    case rnn_enum::kRnnRelu:
    case rnn_enum::kRnnTanh:
      break;
    case rnn_enum::kLstm:
      size *= 4;
      break;
    case rnn_enum::kGru:
      size *= 3;
      break;
  }
  return size;
}

inline size_t GetRNNWorkspaceSize(int seq_length,
                                  int batch_size,
                                  int hidden_size,
                                  int direction,
                                  int mode) {
  size_t size = 0;
  switch (mode) {
    case rnn_enum::kLstm:
      size = (seq_length + 1) * batch_size * hidden_size * 4 + batch_size * hidden_size * 2
             + seq_length * batch_size * hidden_size * direction + hidden_size * seq_length * 8;
      break;
    case rnn_enum::kGru:
      size = seq_length * batch_size * hidden_size * direction * 4 + batch_size * hidden_size * 8;
      break;
    case rnn_enum::kRnnRelu:
    case rnn_enum::kRnnTanh:
      size = seq_length * batch_size * hidden_size * direction * 2 + batch_size * hidden_size * 4;
      break;
    default:
      LOG(FATAL) << "unknown RNN mode " << mode;
      break;
  }
  return size;
}

inline size_t GetRNNReserveSpaceSize(int num_layer,
                                     int direction,
                                     int seq_length,
                                     int batch_size,
                                     int hidden_size,
                                     int mode) {
  size_t size = 0;
  switch (mode) {
    case rnn_enum::kLstm:
      size = direction * seq_length * batch_size * hidden_size * (num_layer * 7 - 1);
      break;
    case rnn_enum::kGru:
      size = seq_length * batch_size * hidden_size * direction * (num_layer * 9 - 1) +
          batch_size * hidden_size * direction * 9 + hidden_size * seq_length * 6 +
          seq_length * batch_size * 7 * hidden_size * direction;
      break;
    case rnn_enum::kRnnRelu:
    case rnn_enum::kRnnTanh:
      size = seq_length * batch_size * hidden_size * direction * (num_layer * 6 - 1) +
          batch_size * hidden_size * direction * 3 + hidden_size * seq_length * 2 +
          seq_length * batch_size * 2 * hidden_size * direction;
      break;
    default:
      LOG(FATAL) << "unknown RNN mode " << mode;
      break;
  }
  return size;
}

struct RNNParam : public dmlc::Parameter<RNNParam> {
  uint32_t state_size;
  uint32_t num_layers;
  bool bidirectional, state_outputs;
  int mode;
  float p;
  int seq_length_, batch_size_, input_size_;
  dmlc::optional<int> projection_size;
  dmlc::optional<double> lstm_state_clip_min, lstm_state_clip_max;
  bool lstm_state_clip_nan;

  DMLC_DECLARE_PARAMETER(RNNParam) {
    DMLC_DECLARE_FIELD(state_size)
    .describe("size of the state for each layer");

    DMLC_DECLARE_FIELD(num_layers)
    .describe("number of stacked layers");

    DMLC_DECLARE_FIELD(bidirectional).set_default(false)
    .describe("whether to use bidirectional recurrent layers");

    DMLC_DECLARE_FIELD(mode)
    .add_enum("rnn_relu", rnn_enum::kRnnRelu)
    .add_enum("rnn_tanh", rnn_enum::kRnnTanh)
    .add_enum("lstm", rnn_enum::kLstm)
    .add_enum("gru", rnn_enum::kGru)
    .describe("the type of RNN to compute");

    DMLC_DECLARE_FIELD(p).set_default(0.)
    .set_range(0, 1)
    .describe("drop rate of the dropout on the outputs of each RNN layer, except the last layer.");

    DMLC_DECLARE_FIELD(state_outputs).set_default(false)
    .describe("Whether to have the states as symbol outputs.");

    DMLC_DECLARE_FIELD(projection_size)
    .set_default(dmlc::optional<int>())
    .describe("size of project size");

    DMLC_DECLARE_FIELD(lstm_state_clip_min)
    .set_default(dmlc::optional<double>())
    .describe("Minimum clip value of LSTM states. This option must be used together with "
              "lstm_state_clip_max.");

    DMLC_DECLARE_FIELD(lstm_state_clip_max)
    .set_default(dmlc::optional<double>())
    .describe("Maximum clip value of LSTM states. This option must be used together with "
              "lstm_state_clip_min.");

    DMLC_DECLARE_FIELD(lstm_state_clip_nan)
    .set_default(false)
    .describe("Whether to stop NaN from propagating in state by clipping it to min/max. "
              "If clipping range is not specified, this option is ignored.");
  }
};

/**
 * @params: ws: Temp workspace for gemm's output storage.
 *          rs: Reserve space of forward intermediate data used for training.
 *          num_layers: The number of recurrent layers.
 *          direction: direction is 2 if use bidirectional recurrent layers, else is 1;
 *          seq_length: The number of iterations to unroll over.
 *          batch_size: size of batch.
 *          input_size: The number of expected input features.
 *          state_size: The number of hidden state features.
 *          x_ptr: Pointer of tensor x containing the features of the input sequence.
 *                 x's shape is [seq_length, batch_size, input_size]
 *          hx_ptr: Pointer of tensor hx containing the initial hidden state.
 *                  hx's shape is [num_layers, batch_size, state_size]
 *          cx_ptr: Only used in lstm mode. pointer of tensor cx containing the initial cell state.
 *                  cx's shape is [num_layers, batch_size, state_size]
 *          w_ptr: Pointer of tensor w containing weights.
 *          b_ptr: Pointer of tensor w containing bias.
 *          y_ptr: Pointer of tensor y containing the features of the output features from the
 *                 last layers of the RNN. y's shape is [seq_length, batch_size, state_size]
 *          hy_ptr: Pointer of tensor hy containing the hidden state for t=seq_length.
 *                  hy's shape is [num_layers, batch_size, state_size]
 *          cy_ptr: Only used in lstm mode. pointer of tensor cy  containing the cell state
 *                  for t=seq_length. cy' shape is [num_layers, batch_size, state_size]
 *          dropout: should be 0 <= dropout < 1
 *          mode: Specifies the type of RNN to compute.
 */
template <typename DType>
void RNNForwardTraining(DType* ws,
                        DType* rs,
                        bool state_outputs,
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
      LstmForwardTraining<DType>(ws, rs, state_outputs, num_layers, direction, seq_length,
                                 batch_size, input_size, state_size, x_ptr, hx_ptr, cx_ptr,
                                 w_ptr, b_ptr, y_ptr, hy_ptr, cy_ptr, dropout);
      break;
    case rnn_enum::kGru:
      GruForwardTraining<DType>(ws, rs, state_outputs, num_layers, direction, seq_length,
                                batch_size, input_size, state_size, x_ptr, hx_ptr,
                                w_ptr, y_ptr, hy_ptr, dropout);
      break;
    case rnn_enum::kRnnTanh:
    case rnn_enum::kRnnRelu:
      VanillaRNNForwardTraining<DType>(ws, rs, state_outputs, num_layers, direction, seq_length,
                                       batch_size, input_size, state_size, x_ptr, hx_ptr,
                                       w_ptr, y_ptr, hy_ptr, dropout, mode);
      break;
    default:
      LOG(FATAL) << "unknown RNN mode " << mode;
      break;
  }
}

template <typename DType>
void RNNForwardInference(DType* ws,
                         bool state_outputs,
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
                         int mode) {
  switch (mode) {
    case rnn_enum::kLstm:
      LstmForwardInference<DType>(ws, state_outputs, num_layers, direction, seq_length,
                                  batch_size, input_size, state_size, x_ptr, hx_ptr, cx_ptr,
                                  w_ptr, b_ptr, y_ptr, hy_ptr, cy_ptr);
      break;
    case rnn_enum::kGru:
      GruForwardInference<DType>(ws, state_outputs, num_layers, direction, seq_length,
                                 batch_size, input_size, state_size, x_ptr, hx_ptr,
                                 w_ptr, y_ptr, hy_ptr);
      break;
    case rnn_enum::kRnnTanh:
    case rnn_enum::kRnnRelu:
      VanillaRNNForwardInference<DType>(ws, state_outputs, num_layers, direction, seq_length,
                                        batch_size, input_size, state_size, x_ptr, hx_ptr,
                                        w_ptr, y_ptr, hy_ptr, mode);
      break;
    default:
      LOG(FATAL) << "unknown RNN mode" << mode;
      break;
  }
}

template <typename DType>
void RNNBackward(DType* ws,
                 DType* rs,
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
      LstmBackward<DType>(ws, rs, num_layers, direction, seq_length, batch_size,
                          input_size, state_size, x_ptr, hx_ptr, cx_ptr, w_ptr, y_ptr,
                          dy_ptr, dhy_ptr, dcy_ptr, dx_ptr, dhx_ptr, dcx_ptr, dw_ptr, db_ptr,
                          req_data, req_params, req_state, req_statecell, dropout);
      break;
    case rnn_enum::kGru:
      GruBackward<DType>(ws, rs, num_layers, direction, seq_length, batch_size,
                         input_size, state_size, x_ptr, hx_ptr, w_ptr,
                         dy_ptr, dhy_ptr, dx_ptr, dhx_ptr, dw_ptr,
                         req_data, req_params, req_state, dropout);
      break;
    case rnn_enum::kRnnTanh:
    case rnn_enum::kRnnRelu:
      VanillaRNNBackward<DType>(ws, rs, num_layers, direction, seq_length, batch_size,
                                input_size, state_size, x_ptr, hx_ptr, w_ptr,
                                dy_ptr, dhy_ptr, dx_ptr, dhx_ptr, dw_ptr,
                                req_data, req_params, req_state, dropout, mode);
      break;
    default:
      LOG(FATAL) << "unknown RNN mode" << mode;
      break;
  }
}

template<typename xpu, typename DType>
class RNNOp {
 public:
  RNNParam param_;
  Context ctx_;
  #if MXNET_USE_MKLDNN == 1
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
  bool has_cache;
  bool init_mem_;
  size_t reserve_mem_size_;
  Storage::Handle mem_space_;
  #endif
  explicit RNNOp(RNNParam param, Context ctx) {
    this->param_ = param;
    this->ctx_ = ctx;
    #if MXNET_USE_MKLDNN == 1
    init_mem_ = false;
    reserve_mem_size_ = 0;
    #endif
    #if MXNET_USE_CUDNN_RNN
    init_cudnn_ = false;
    dtype_ = mshadow::DataType<DType>::kCudnnFlag;
    // TensorCore algos only allowed on fp16-I/O convolutions if permitted by the global policy.
    // No tests in place for fp16 RNNs, so leave TensorCore disabled for now.
    cudnn_tensor_core_ = false;
    // When fp16 RNN tests are introduced, we can enable TensorCore as follows:
//    cudnn_tensor_core =
//        mshadow::DataType<DType>::kFlag == mshadow::kFloat16 && GetEnvAllowTensorCore();
    // Defaults
    input_mode_ = CUDNN_LINEAR_INPUT;  // Don't support this yet
    // RNN Mode
    switch (param_.mode) {
      case rnn_enum::kRnnRelu:
        mode_ = CUDNN_RNN_RELU;
        break;
      case rnn_enum::kRnnTanh:
        mode_ = CUDNN_RNN_TANH;
        break;
      case rnn_enum::kLstm:
        mode_ = CUDNN_LSTM;
        break;
      case rnn_enum::kGru:
        mode_ = CUDNN_GRU;
        break;
      default:
        LOG(FATAL) << "Not implmented";
    }
#if USE_CUDNN_LSTM_PROJ
    if (param_.projection_size.has_value()) {
      CHECK_EQ(param_.mode, rnn_enum::kLstm)
        << "Projection is only supported for LSTM.";
      CHECK_GE(param_.state_size, param_.projection_size.value())
        << "State size must be larger than projection size.";
    }
#else
    CHECK(!param_.projection_size.has_value())
      << "Projection is only supported for LSTM with CuDNN version later than 7.1.1.";
#endif
#if USE_CUDNN_LSTM_PROJ
    if (param_.lstm_state_clip_min.has_value()
        || param_.lstm_state_clip_max.has_value()) {
      CHECK_EQ(param_.mode, rnn_enum::kLstm)
        << "State clipping is only supported for LSTM.";
      CHECK(param_.lstm_state_clip_min.has_value() && param_.lstm_state_clip_max.has_value())
        << "lstm_state_clip_min and lstm_state_clip_max must be specified together.";
      CHECK_GE(param_.lstm_state_clip_max.value(), param_.lstm_state_clip_min.value())
        << "lstm_state_clip_max must be greater or equal to lstm_state_clip_min";
    }
#else
    CHECK(!param_.lstm_state_clip_min.has_value()
          && !param_.lstm_state_clip_max.has_value())
      << "State clipping is only supported for LSTM with CuDNN version later than 7.2.1.";
#endif
    // RNN Direction
    direction_ = param_.bidirectional ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL;
    // Create descriptors
    CUDNN_CALL(cudnnCreateTensorDescriptor(&hx_desc_));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&cx_desc_));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&hy_desc_));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&cy_desc_));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&dhx_desc_));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&dcx_desc_));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&dhy_desc_));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&dcy_desc_));

    CUDNN_CALL(cudnnCreateFilterDescriptor(&w_desc_));
    CUDNN_CALL(cudnnCreateFilterDescriptor(&dw_desc_));

    CUDNN_CALL(cudnnCreateRNNDescriptor(&rnn_desc_));
    CUDNN_CALL(cudnnCreateDropoutDescriptor(&dropout_desc_));

    #if USE_CUDNN_LSTM_PROJ
    CUDNN_CALL(cudnnCreateRNNDataDescriptor(&x_data_desc_));
    CUDNN_CALL(cudnnCreateRNNDataDescriptor(&y_data_desc_));
    CUDNN_CALL(cudnnCreateRNNDataDescriptor(&dx_data_desc_));
    CUDNN_CALL(cudnnCreateRNNDataDescriptor(&dy_data_desc_));
    #endif
    #else
    if (ctx_.dev_type == kGPU) {
      LOG(FATAL) << "RNN on GPU is only available for cuDNN at the moment.";
    }
    #endif

    if (ctx_.dev_type == kCPU) {
      this->init_space_ = false;
      this->temp_init_space_ = false;
      this->reserve_cpu_space_size_ = 0;
      this->temp_cpu_space_size_ = 0;
      if (param_.projection_size.has_value()) {
        LOG(FATAL) <<
            "hidden layer projection is only supported for GPU with CuDNN later than 7.1.1";
      }
      if (param_.lstm_state_clip_min.has_value()
          || param_.lstm_state_clip_max.has_value()) {
        LOG(FATAL) << "LSTM state clipping is only supported for GPU with CuDNN later than 7.2.1";
      }
    }
  }

  ~RNNOp() {
    #if MXNET_USE_MKLDNN == 1
    if (init_mem_) {
      Storage::Get()->Free(mem_space_);
      init_mem_ = false;
    }
    #endif
    #if MXNET_USE_CUDNN_RNN
    CUDNN_CALL(cudnnDestroyTensorDescriptor(hx_desc_));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(cx_desc_));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(hy_desc_));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(cy_desc_));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(dhx_desc_));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(dcx_desc_));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(dhy_desc_));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(dcy_desc_));

    CUDNN_CALL(cudnnDestroyFilterDescriptor(w_desc_));
    CUDNN_CALL(cudnnDestroyFilterDescriptor(dw_desc_));
    CUDNN_CALL(cudnnDestroyRNNDescriptor(rnn_desc_));
    CUDNN_CALL(cudnnDestroyDropoutDescriptor(dropout_desc_));

    if (init_cudnn_) {
      for (size_t i = 0; i < x_desc_vec_.size(); ++i) {
        CUDNN_CALL(cudnnDestroyTensorDescriptor(x_desc_vec_[i]));
        CUDNN_CALL(cudnnDestroyTensorDescriptor(y_desc_vec_[i]));
        CUDNN_CALL(cudnnDestroyTensorDescriptor(dx_desc_vec_[i]));
        CUDNN_CALL(cudnnDestroyTensorDescriptor(dy_desc_vec_[i]));
      }
      init_cudnn_ = false;
      Storage::Get()->Free(temp_space_);
      Storage::Get()->Free(reserve_space_);
    }
    #if USE_CUDNN_LSTM_PROJ
    CUDNN_CALL(cudnnDestroyRNNDataDescriptor(x_data_desc_));
    CUDNN_CALL(cudnnDestroyRNNDataDescriptor(y_data_desc_));
    CUDNN_CALL(cudnnDestroyRNNDataDescriptor(dx_data_desc_));
    CUDNN_CALL(cudnnDestroyRNNDataDescriptor(dy_data_desc_));
    #endif
    #endif

    if (ctx_.dev_type == kCPU) {
      if (init_space_) {
        Storage::Get()->Free(reserve_cpu_space_);
        init_space_ = false;
      }
      if (temp_init_space_) {
        Storage::Get()->Free(temp_cpu_space_);
        temp_init_space_ = false;
      }
    }
  }

  void Forward(const OpContext &ctx, const std::vector<TBlob> &in_data,
               const std::vector<OpReqType> &req,
               const std::vector<TBlob> &out_data) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK(param_.p >= 0.0f && param_.p < 1.0f)
        << "unsupported dropout value, should be 0 <= dropout < 1";
    size_t num_inputs = (param_.mode == rnn_enum::kLstm) ? 4 : 3;
    //  kOut
    size_t num_outputs = 1;
    if (param_.state_outputs) {
      // kOut, kStateOut, kStateCellOut
      num_outputs = (param_.mode == rnn_enum::kLstm) ? 3 : 2;
    }

    CHECK_EQ(in_data.size(), num_inputs);
    CHECK_EQ(out_data.size(), num_outputs);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    // get input + output tensors
    Tensor<xpu, 3, DType> x = in_data[rnn_enum::kData].get<xpu, 3, DType>(s);
    Tensor<xpu, 1, DType> w = in_data[rnn_enum::kParams].get<xpu, 1, DType>(s);
    Tensor<xpu, 3, DType> hx = in_data[rnn_enum::kState].get<xpu, 3, DType>(s);
    Tensor<xpu, 3, DType> y = out_data[rnn_enum::kOut].get<xpu, 3, DType>(s);

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
    if (param_.mode == rnn_enum::kLstm)
      cx_ptr = (in_data[rnn_enum::kStateCell].get<xpu, 3, DType>(s)).dptr_;
    if (param_.mode == rnn_enum::kLstm && param_.state_outputs)
      cy_ptr = (out_data[rnn_enum::kStateCellOut].get<xpu, 3, DType>(s)).dptr_;

    CHECK_EQ(x.CheckContiguous(), true);
    CHECK_EQ(w.CheckContiguous(), true);
    CHECK_EQ(hx.CheckContiguous(), true);
    CHECK_EQ(y.CheckContiguous(), true);

    #if MXNET_USE_CUDNN_RNN && defined(__CUDACC__)
    if (!init_cudnn_) {
      Init(ctx, s, in_data, out_data);
    }

    #if USE_CUDNN_LSTM_PROJ
    std::vector<int> seqLengthArray(param_.batch_size_, param_.seq_length_);
    CUDNN_CALL(cudnnSetRNNDataDescriptor(x_data_desc_,
                                         dtype_,
                                         CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED,
                                         param_.seq_length_,
                                         param_.batch_size_,
                                         param_.input_size_,
                                         seqLengthArray.data(),
                                         nullptr));
    int out_size =
      (param_.projection_size.has_value()) ? param_.projection_size.value() : param_.state_size;
    out_size = (param_.bidirectional) ? (out_size * 2) : out_size;
    CUDNN_CALL(cudnnSetRNNDataDescriptor(y_data_desc_,
                                         dtype_,
                                         CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED,
                                         param_.seq_length_,
                                         param_.batch_size_,
                                         out_size,
                                         seqLengthArray.data(),
                                         nullptr));
    if (ctx.is_train) {
      CUDNN_CALL(cudnnSetRNNDataDescriptor(dx_data_desc_,
                                           dtype_,
                                           CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED,
                                           param_.seq_length_,
                                           param_.batch_size_,
                                           param_.input_size_,
                                           seqLengthArray.data(),
                                           nullptr));
      CUDNN_CALL(cudnnSetRNNDataDescriptor(dy_data_desc_,
                                           dtype_,
                                           CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED,
                                           param_.seq_length_,
                                           param_.batch_size_,
                                           out_size,
                                           seqLengthArray.data(),
                                           nullptr));
    }
    #endif

    #if USE_CUDNN_LSTM_PROJ
    bool clip_state = param_.lstm_state_clip_min.has_value();
    bool clip_nan = param_.lstm_state_clip_nan;
    CUDNN_CALL(cudnnRNNSetClip(s->dnn_handle_,
                               rnn_desc_,
                               clip_state ? CUDNN_RNN_CLIP_MINMAX : CUDNN_RNN_CLIP_NONE,
                               clip_nan ? CUDNN_NOT_PROPAGATE_NAN : CUDNN_PROPAGATE_NAN,
                               clip_state ? param_.lstm_state_clip_min.value() : 0.0,
                               clip_state ? param_.lstm_state_clip_max.value() : 0.0));
    #endif

    if (ctx.is_train) {
      #if USE_CUDNN_LSTM_PROJ
      CUDNN_CALL(cudnnRNNForwardTrainingEx(s->dnn_handle_,
                                           rnn_desc_,
                                           x_data_desc_,
                                           x.dptr_,
                                           hx_desc_,
                                           hx.dptr_,
                                           cx_desc_,
                                           cx_ptr,
                                           w_desc_,
                                           w.dptr_,
                                           y_data_desc_,
                                           y.dptr_,
                                           hy_desc_,
                                           hy_ptr,
                                           cy_desc_,
                                           cy_ptr,
                                           nullptr,
                                           nullptr,
                                           nullptr,
                                           nullptr,
                                           nullptr,
                                           nullptr,
                                           nullptr,
                                           nullptr,
                                           temp_space_.dptr,
                                           workspace_byte_,
                                           reserve_space_.dptr,
                                           reserve_space_byte_));
      #else
      CUDNN_CALL(cudnnRNNForwardTraining(s->dnn_handle_,
                                         rnn_desc_,
                                         param_.seq_length_,
                                         x_desc_vec_.data(),
                                         x.dptr_,
                                         hx_desc_,
                                         hx.dptr_,
                                         cx_desc_,
                                         cx_ptr,
                                         w_desc_,
                                         w.dptr_,
                                         y_desc_vec_.data(),
                                         y.dptr_,
                                         hy_desc_,
                                         hy_ptr,
                                         cy_desc_,
                                         cy_ptr,
                                         temp_space_.dptr,
                                         workspace_byte_,
                                         reserve_space_.dptr,
                                         reserve_space_byte_));
      #endif
    } else {
      #if USE_CUDNN_LSTM_PROJ
      CUDNN_CALL(cudnnRNNForwardInferenceEx(s->dnn_handle_,
                                            rnn_desc_,
                                            x_data_desc_,
                                            x.dptr_,
                                            hx_desc_,
                                            hx.dptr_,
                                            cx_desc_,
                                            cx_ptr,
                                            w_desc_,
                                            w.dptr_,
                                            y_data_desc_,
                                            y.dptr_,
                                            hy_desc_,
                                            hy_ptr,
                                            cy_desc_,
                                            cy_ptr,
                                            nullptr,
                                            nullptr,
                                            nullptr,
                                            nullptr,
                                            nullptr,
                                            nullptr,
                                            nullptr,
                                            nullptr,
                                            temp_space_.dptr,
                                            workspace_byte_));
      #else
      CUDNN_CALL(cudnnRNNForwardInference(s->dnn_handle_,
                                          rnn_desc_,
                                          param_.seq_length_,
                                          x_desc_vec_.data(),
                                          x.dptr_,
                                          hx_desc_,
                                          hx.dptr_,
                                          cx_desc_,
                                          cx_ptr,
                                          w_desc_,
                                          w.dptr_,
                                          y_desc_vec_.data(),
                                          y.dptr_,
                                          hy_desc_,
                                          hy_ptr,
                                          cy_desc_,
                                          cy_ptr,
                                          temp_space_.dptr,
                                          workspace_byte_));
      #endif
    }
    #endif

    if (ctx_.dev_type == kCPU) {
      if (ctx.is_train) {
        // allocate temp space
        const size_t work_cpu_space_size =
            GetRNNWorkspaceSize(param_.seq_length_, param_.batch_size_,
                              param_.state_size, direction, param_.mode);
        if (temp_init_space_ && temp_cpu_space_size_ < work_cpu_space_size) {
            Storage::Get()->Free(temp_cpu_space_);
            temp_init_space_ = false;
        }
        if (!temp_init_space_) {
          temp_cpu_space_ = Storage::Get()->Alloc
              (work_cpu_space_size * sizeof(DType), Context::CPU());
          temp_cpu_space_size_ = work_cpu_space_size;
          temp_init_space_ = true;
        }
        DType* work_cpu_space = static_cast<DType*>(temp_cpu_space_.dptr);

        const size_t r_size = GetRNNReserveSpaceSize(param_.num_layers, direction,
                                                     param_.seq_length_, param_.batch_size_,
                                                     param_.state_size, param_.mode);
        if (init_space_ && reserve_cpu_space_size_ < r_size) {
          Storage::Get()->Free(reserve_cpu_space_);
          init_space_ = false;
        }
        if (!init_space_) {
          reserve_cpu_space_ = Storage::Get()->Alloc(r_size * sizeof(DType), Context::CPU());
          reserve_cpu_space_size_ = r_size;
          init_space_ = true;
        }

        DType* reserve_space_ptr = static_cast<DType*>(reserve_cpu_space_.dptr);

        RNNForwardTraining<DType>(work_cpu_space,
                                  reserve_space_ptr,
                                  param_.state_outputs,
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
        #if MXNET_USE_MKLDNN == 1
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
                                         &has_cache,
                                         dtype,
                                         ctx.is_train,
                                         param_.mode);
        #else
        const size_t work_cpu_space_size =
            GetRNNWorkspaceSize(param_.seq_length_, param_.batch_size_,
                              param_.state_size, direction, param_.mode);
        if (temp_init_space_ && temp_cpu_space_size_ < work_cpu_space_size) {
            Storage::Get()->Free(temp_cpu_space_);
            temp_init_space_ = false;
        }
        if (!temp_init_space_) {
          temp_cpu_space_ = Storage::Get()->Alloc
              (work_cpu_space_size * sizeof(DType), Context::CPU());
          temp_cpu_space_size_ = work_cpu_space_size;
          temp_init_space_ = true;
        }
        DType* work_cpu_space = static_cast<DType*>(temp_cpu_space_.dptr);
        RNNForwardInference<DType>(work_cpu_space,
                                   param_.state_outputs,
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
                                   param_.mode);
        #endif
      }
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

    size_t num_inputs = (param_.mode == rnn_enum::kLstm) ? 4 : 3;
    //  kOut
    size_t num_outputs = 1;
    if (param_.state_outputs) {
      // kOut, kStateOut, kStateCellOut
      num_outputs = (param_.mode == rnn_enum::kLstm) ? 3 : 2;
    }

    CHECK_EQ(in_data.size(), num_inputs);
    CHECK_EQ(out_data.size(), num_outputs);
    CHECK_EQ(in_grad.size(), num_inputs);
    CHECK_EQ(out_grad.size(), num_outputs);
    CHECK_EQ(req.size(), num_inputs);
    CHECK_NE(req[rnn_enum::kData], kAddTo) << "AddTo is not supported for data";
    CHECK_NE(req[rnn_enum::kState], kAddTo) << "AddTo is not supported for state";
    Stream<xpu> *s = ctx.get_stream<xpu>();
    // get input + output tensors
    Tensor<xpu, 3, DType> x = in_data[rnn_enum::kData].get<xpu, 3, DType>(s);
    Tensor<xpu, 3, DType> dx = in_grad[rnn_enum::kData].get<xpu, 3, DType>(s);
    Tensor<xpu, 1, DType> w = in_data[rnn_enum::kParams].get<xpu, 1, DType>(s);
    Tensor<xpu, 1, DType> dw = in_grad[rnn_enum::kParams].get<xpu, 1, DType>(s);
    Tensor<xpu, 3, DType> hx = in_data[rnn_enum::kState].get<xpu, 3, DType>(s);
    Tensor<xpu, 3, DType> dhx = in_grad[rnn_enum::kState].get<xpu, 3, DType>(s);
    Tensor<xpu, 3, DType> y = out_data[rnn_enum::kOut].get<xpu, 3, DType>(s);
    Tensor<xpu, 3, DType> dy = out_grad[rnn_enum::kOut].get<xpu, 3, DType>(s);

    CHECK_EQ(x.CheckContiguous(), true);
    CHECK_EQ(w.CheckContiguous(), true);
    CHECK_EQ(dw.CheckContiguous(), true);
    CHECK_EQ(hx.CheckContiguous(), true);
    CHECK_EQ(dhx.CheckContiguous(), true);
    CHECK_EQ(y.CheckContiguous(), true);
    CHECK_EQ(dy.CheckContiguous(), true);
    CHECK_EQ(dx.CheckContiguous(), true);

    if (req[rnn_enum::kParams] != kAddTo) {
      dw = mshadow::expr::ScalarExp<DType>(0.0f);
    }

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

    DType* dcx_ptr = NULL;
    DType* dcy_ptr = NULL;
    DType* cx_ptr = NULL;

    if (param_.mode == rnn_enum::kLstm) {
      CHECK_NE(req[rnn_enum::kStateCell], kAddTo) << "AddTo is not supported for state cell";
      cx_ptr = (in_data[rnn_enum::kStateCell].get<xpu, 3, DType>(s)).dptr_;
      dcx_ptr = (in_grad[rnn_enum::kStateCell].get<xpu, 3, DType>(s)).dptr_;
    }
    if ((param_.mode == rnn_enum::kLstm) && param_.state_outputs)
        dcy_ptr = (out_grad[rnn_enum::kStateCellOut].get<xpu, 3, DType>(s)).dptr_;

    #if MXNET_USE_CUDNN_RNN && defined(__CUDACC__)
    if (!init_cudnn_) {
      Init(ctx, s, in_data, out_data);
    }

    #if USE_CUDNN_LSTM_PROJ
    CUDNN_CALL(cudnnRNNBackwardDataEx(s->dnn_handle_,
                                      rnn_desc_,
                                      y_data_desc_,
                                      y.dptr_,
                                      dy_data_desc_,
                                      dy.dptr_,
                                      nullptr,
                                      nullptr,
                                      dhy_desc_,
                                      dhy_ptr,
                                      dcy_desc_,
                                      dcy_ptr,
                                      w_desc_,
                                      w.dptr_,
                                      hx_desc_,
                                      hx.dptr_,
                                      cx_desc_,
                                      cx_ptr,
                                      dx_data_desc_,
                                      dx.dptr_,
                                      dhx_desc_,
                                      dhx.dptr_,
                                      dcx_desc_,
                                      dcx_ptr,
                                      nullptr,
                                      nullptr,
                                      temp_space_.dptr,
                                      workspace_byte_,
                                      reserve_space_.dptr,
                                      reserve_space_byte_));
    CUDNN_CALL(cudnnRNNBackwardWeightsEx(s->dnn_handle_,
                                         rnn_desc_,
                                         x_data_desc_,
                                         x.dptr_,
                                         hx_desc_,
                                         hx.dptr_,
                                         y_data_desc_,
                                         y.dptr_,
                                         temp_space_.dptr,
                                         workspace_byte_,
                                         dw_desc_,
                                         dw.dptr_,
                                         reserve_space_.dptr,
                                         reserve_space_byte_));
    #else
    CUDNN_CALL(cudnnRNNBackwardData(s->dnn_handle_,
                                    rnn_desc_,
                                    param_.seq_length_,
                                    y_desc_vec_.data(),
                                    y.dptr_,
                                    dy_desc_vec_.data(),
                                    dy.dptr_,
                                    dhy_desc_,
                                    dhy_ptr,
                                    dcy_desc_,
                                    dcy_ptr,
                                    w_desc_,
                                    w.dptr_,
                                    hx_desc_,
                                    hx.dptr_,
                                    cx_desc_,
                                    cx_ptr,
                                    dx_desc_vec_.data(),
                                    dx.dptr_,
                                    dhx_desc_,
                                    dhx.dptr_,
                                    dcx_desc_,
                                    dcx_ptr,
                                    temp_space_.dptr,
                                    workspace_byte_,
                                    reserve_space_.dptr,
                                    reserve_space_byte_));
    CUDNN_CALL(cudnnRNNBackwardWeights(s->dnn_handle_,
                                       rnn_desc_,
                                       param_.seq_length_,
                                       x_desc_vec_.data(),
                                       x.dptr_,
                                       hx_desc_,
                                       hx.dptr_,
                                       y_desc_vec_.data(),
                                       y.dptr_,
                                       temp_space_.dptr,
                                       workspace_byte_,
                                       dw_desc_,
                                       dw.dptr_,
                                       reserve_space_.dptr,
                                       reserve_space_byte_));
    #endif
    #endif

    if (ctx_.dev_type == kCPU) {
      // allocate temp space
      const size_t work_cpu_space_size =
          GetRNNWorkspaceSize(param_.seq_length_, param_.batch_size_,
                              param_.state_size, direction, param_.mode);
      if (!temp_init_space_ || temp_cpu_space_size_ != work_cpu_space_size) {
        LOG(FATAL) << "Check temp init error";
      }
      DType* work_cpu_space = static_cast<DType*>(temp_cpu_space_.dptr);
      size_t r_size = GetRNNReserveSpaceSize(param_.num_layers, direction,
                                             param_.seq_length_, param_.batch_size_,
                                             param_.state_size, param_.mode);

      if (!init_space_ || reserve_cpu_space_size_ != r_size) {
        LOG(FATAL) << "Check forward init error";
      }

      DType* reserve_space_ptr = static_cast<DType*>(reserve_cpu_space_.dptr);
      RNNBackward<DType>(work_cpu_space,
                         reserve_space_ptr,
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
                         // State cell should be present for LSTMs, but is absent for other RNNs.
                         param_.mode == rnn_enum::kLstm ? req[rnn_enum::kStateCell] : kNullOp,
                         param_.p,
                         param_.mode);
    }
  }


 private:
  inline void Init(const OpContext &ctx,
                   mshadow::Stream<xpu> *s,
                   const std::vector<TBlob> &in_data,
                   const std::vector<TBlob> &out_data) {
    using namespace mshadow;
    size_t num_inputs = (param_.mode == rnn_enum::kLstm) ? 4 : 3;
    //  kOut
    size_t num_outputs = 1;
    if (param_.state_outputs) {
      // kOut, kStateOut, kStateCellOut
      num_outputs = (param_.mode == rnn_enum::kLstm) ? 3 : 2;
    }

    CHECK_EQ(in_data.size(), num_inputs);
    CHECK_EQ(out_data.size(), num_outputs);

    #if MXNET_USE_CUDNN_RNN && defined(__CUDACC__)
    #if CUDNN_MAJOR >= 5
    format_ = CUDNN_TENSOR_NCHW;
    #endif

    if (!init_cudnn_) {
      init_cudnn_ = true;
      // get input + output tensors
      Tensor<xpu, 3, DType> x = in_data[rnn_enum::kData].get<xpu, 3, DType>(s);
      Tensor<xpu, 1, DType> w = in_data[rnn_enum::kParams].get<xpu, 1, DType>(s);
      param_.seq_length_ = x.shape_[0];
      param_.batch_size_ = x.shape_[1];
      param_.input_size_ = x.shape_[2];

      // Tensor Descriptors
      std::vector<cudnnTensorDescriptor_t> x_vec(param_.seq_length_);
      std::vector<cudnnTensorDescriptor_t> y_vec(param_.seq_length_);
      std::vector<cudnnTensorDescriptor_t> dx_vec(param_.seq_length_);
      std::vector<cudnnTensorDescriptor_t> dy_vec(param_.seq_length_);
      int dimA[3];
      int strideA[3];
      for (int i = 0; i < param_.seq_length_; i++) {
        CUDNN_CALL(cudnnCreateTensorDescriptor(&x_vec[i]));
        CUDNN_CALL(cudnnCreateTensorDescriptor(&y_vec[i]));
        CUDNN_CALL(cudnnCreateTensorDescriptor(&dx_vec[i]));
        CUDNN_CALL(cudnnCreateTensorDescriptor(&dy_vec[i]));

        dimA[0] = param_.batch_size_;
        dimA[1] = param_.input_size_;
        dimA[2] = 1;
        strideA[0] = dimA[2] * dimA[1];
        strideA[1] = dimA[2];
        strideA[2] = 1;

        CUDNN_CALL(cudnnSetTensorNdDescriptor(x_vec[i],
                                              dtype_,
                                              3,
                                              dimA,
                                              strideA));
        CUDNN_CALL(cudnnSetTensorNdDescriptor(dx_vec[i],
                                              dtype_,
                                              3,
                                              dimA,
                                              strideA));
        dimA[0] = param_.batch_size_;
        dimA[1] = param_.bidirectional ? param_.state_size * 2 : param_.state_size;
        dimA[2] = 1;
        strideA[0] = dimA[2] * dimA[1];
        strideA[1] = dimA[2];
        strideA[2] = 1;

        CUDNN_CALL(cudnnSetTensorNdDescriptor(y_vec[i],
                                              dtype_,
                                              3,
                                              dimA,
                                              strideA));
        CUDNN_CALL(cudnnSetTensorNdDescriptor(dy_vec[i],
                                              dtype_,
                                              3,
                                              dimA,
                                              strideA));
      }
      x_desc_vec_ = x_vec;
      y_desc_vec_ = y_vec;
      dx_desc_vec_ = dx_vec;
      dy_desc_vec_ = dy_vec;

      // set the state tensors
      dimA[0] = param_.num_layers * (param_.bidirectional ? 2 : 1);
      dimA[1] = param_.batch_size_;
      dimA[2] = param_.state_size;
      strideA[0] = dimA[2] * dimA[1];
      strideA[1] = dimA[2];
      strideA[2] = 1;
      #if USE_CUDNN_LSTM_PROJ
      int dimB[3];
      int strideB[3];
      dimB[0] = param_.num_layers * (param_.bidirectional ? 2 : 1);
      dimB[1] = param_.batch_size_;
      dimB[2] = param_.projection_size.has_value() ?
                param_.projection_size.value() : param_.state_size;
      strideB[0] = dimB[2] * dimB[1];
      strideB[1] = dimB[2];
      strideB[2] = 1;
      #endif
      #if USE_CUDNN_LSTM_PROJ
      CUDNN_CALL(cudnnSetTensorNdDescriptor(hx_desc_,
                                            dtype_,
                                            3,
                                            dimB,
                                            strideB));
      #else
      CUDNN_CALL(cudnnSetTensorNdDescriptor(hx_desc_,
                                            dtype_,
                                            3,
                                            dimA,
                                            strideA));
      #endif
      CUDNN_CALL(cudnnSetTensorNdDescriptor(cx_desc_,
                                            dtype_,
                                            3,
                                            dimA,
                                            strideA));
      #if USE_CUDNN_LSTM_PROJ
      CUDNN_CALL(cudnnSetTensorNdDescriptor(hy_desc_,
                                            dtype_,
                                            3,
                                            dimB,
                                            strideB));
      #else
      CUDNN_CALL(cudnnSetTensorNdDescriptor(hy_desc_,
                                            dtype_,
                                            3,
                                            dimA,
                                            strideA));
      #endif
      CUDNN_CALL(cudnnSetTensorNdDescriptor(cy_desc_,
                                            dtype_,
                                            3,
                                            dimA,
                                            strideA));
      #if USE_CUDNN_LSTM_PROJ
      CUDNN_CALL(cudnnSetTensorNdDescriptor(dhx_desc_,
                                            dtype_,
                                            3,
                                            dimB,
                                            strideB));
      #else
      CUDNN_CALL(cudnnSetTensorNdDescriptor(dhx_desc_,
                                            dtype_,
                                            3,
                                            dimA,
                                            strideA));
      #endif
      CUDNN_CALL(cudnnSetTensorNdDescriptor(dcx_desc_,
                                            dtype_,
                                            3,
                                            dimA,
                                            strideA));
      #if USE_CUDNN_LSTM_PROJ
      CUDNN_CALL(cudnnSetTensorNdDescriptor(dhy_desc_,
                                            dtype_,
                                            3,
                                            dimB,
                                            strideB));
      #else
      CUDNN_CALL(cudnnSetTensorNdDescriptor(dhy_desc_,
                                            dtype_,
                                            3,
                                            dimA,
                                            strideA));
      #endif
      CUDNN_CALL(cudnnSetTensorNdDescriptor(dcy_desc_,
                                            dtype_,
                                            3,
                                            dimA,
                                            strideA));

      // Create Dropout descriptors
      DType* dropout_states_ = NULL;
      if (param_.p > 0) {
         ctx.requested[rnn_enum::kCuDNNDropoutDescSpace].get_cudnn_dropout_desc
            (&dropout_desc_, s, 1.0f - param_.p, seed_);
      } else {
        dropout_byte_ = 0;
      }

      CUDNN_CALL(cudnnSetDropoutDescriptor(dropout_desc_, s->dnn_handle_,
                                           param_.p,  // discard probability
                                           dropout_states_, dropout_byte_,
                                           seed_));

      // RNN descriptors
      #if CUDNN_MAJOR >= 6
      cudnnRNNAlgo_t rnn_algo = CUDNN_RNN_ALGO_STANDARD;
      CUDNN_CALL(cudnnSetRNNDescriptor_v6(s->dnn_handle_,
                                          rnn_desc_,
                                          param_.state_size,
                                          param_.num_layers,
                                          dropout_desc_,
                                          input_mode_,
                                          direction_,
                                          mode_,
                                          rnn_algo,
                                          dtype_));
      #else
      CUDNN_CALL(cudnnSetRNNDescriptor(rnn_desc_,
                                       param_.state_size,
                                       param_.num_layers,
                                       dropout_desc_,
                                       input_mode_,
                                       direction_,
                                       mode_,
                                       dtype_));
      #endif
      #if CUDNN_MAJOR >= 7
        cudnnMathType_t math_type = CUDNN_DEFAULT_MATH;
        if (cudnn_tensor_core_ && rnn_algo == CUDNN_RNN_ALGO_STANDARD) {
          math_type = CUDNN_TENSOR_OP_MATH;
        }
      #if CUDNN_VERSION >= 7200
            if (GetEnvAllowTensorCore() && GetEnvAllowTensorCoreConversion() &&
                (DataType<DType>::kFlag != kFloat16))
              math_type = CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION;
      #endif
        CUDNN_CALL(cudnnSetRNNMatrixMathType(rnn_desc_, math_type));
      #endif
      #if USE_CUDNN_LSTM_PROJ
      if (param_.projection_size.has_value()) {
        CUDNN_CALL(cudnnSetRNNProjectionLayers(s->dnn_handle_,
                                               rnn_desc_,
                                               param_.projection_size.value(),
                                               0));
      }
      #endif
      // Get temp space sizes
      CUDNN_CALL(cudnnGetRNNWorkspaceSize(s->dnn_handle_,
                                          rnn_desc_,
                                          param_.seq_length_,
                                          x_desc_vec_.data(),
                                          &workspace_byte_));
      CUDNN_CALL(cudnnGetRNNTrainingReserveSize(s->dnn_handle_,
                                                rnn_desc_,
                                                param_.seq_length_,
                                                x_desc_vec_.data(),
                                                &reserve_space_byte_));
      workspace_size_ = workspace_byte_ / sizeof(DType);
      // Allocate the reserve space
      reserve_space_ = Storage::Get()->Alloc(reserve_space_byte_, Context::GPU(s->dev_id));
      // Allocate the temp space
      temp_space_ = Storage::Get()->Alloc(workspace_byte_, Context::GPU(s->dev_id));
      // Check that number of params are correct
      size_t cudnn_param_size;
      CUDNN_CALL(cudnnGetRNNParamsSize(s->dnn_handle_,
                                       rnn_desc_,
                                       x_desc_vec_[0],
                                       &cudnn_param_size,
                                       dtype_));
      CHECK_EQ(w.shape_[0] * sizeof(DType), cudnn_param_size);
      // Set param descriptors
      int dim_w[3] = {1, 1, 1};
      dim_w[0] = w.shape_[0];
      CUDNN_CALL(cudnnSetFilterNdDescriptor(w_desc_,
                                            dtype_,
                                            format_,
                                            3,
                                            dim_w));
      CUDNN_CALL(cudnnSetFilterNdDescriptor(dw_desc_,
                                            dtype_,
                                            format_,
                                            3,
                                            dim_w));

      // Query weight layout
      // cudnnFilterDescriptor_t m_desc;
      // CHECK_EQ(cudnnCreateFilterDescriptor(&m_desc), CUDNN_STATUS_SUCCESS);
      // DType *p;
      // int n = 2;
      // int64_t last = 0;
      // if (param_.mode == rnn_enum::kLstm) n = 8;
      // else if (param_.mode == rnn_enum::kGru) n = 6;

      // for (int i = 0; i < param_.num_layers*(param_.bidirectional?2:1); ++i) {
      //   for (int j = 0; j < n; ++j) {
      //     CHECK_EQ(cudnnGetRNNLinLayerMatrixParams(s->dnn_handle_, rnn_desc_,
      //       i, x_desc_vec_[0], w_desc_, 0, j, m_desc, (void**)&p), CUDNN_STATUS_SUCCESS);
      //     LOG(INFO) << ((int64_t)(p - NULL))/sizeof(DType) - last;
      //     last = ((int64_t)(p - NULL))/sizeof(DType);
      //     cudnnDataType_t t;
      //     cudnnTensorFormat_t f;
      //     int ndim = 5;
      //     int dims[5] = {0, 0, 0, 0, 0};
      //     CHECK_EQ(cudnnGetFilterNdDescriptor(m_desc, ndim, &t, &f, &ndim, &dims[0]),
      //       CUDNN_STATUS_SUCCESS);
      //     LOG(INFO) << "w: " <<  i << " " << j << " " << ((int64_t)(p - NULL))/sizeof(DType);
      //     for (int i = 0; i < ndim; ++i) LOG(INFO) << dims[i];
      //   }
      // }

      // for (int i = 0; i < param_.num_layers*(param_.bidirectional?2:1); ++i) {
      //   for (int j = 0; j < n; ++j) {
      //     CHECK_EQ(cudnnGetRNNLinLayerBiasParams(s->dnn_handle_, rnn_desc_, i, x_desc_vec_[0],
      //       w_desc_, 0, j, m_desc, (void**)&p), CUDNN_STATUS_SUCCESS);
      //     LOG(INFO) << ((int64_t)(p - NULL))/sizeof(DType) - last;
      //     last = ((int64_t)(p - NULL))/sizeof(DType);
      //     LOG(INFO) << "b: " << i << " " << j << " " << ((int64_t)(p - NULL))/sizeof(DType);
      //   }
      // }
    }
  #endif
  }
  #if MXNET_USE_CUDNN_RNN
  cudnnDataType_t dtype_;
  bool init_cudnn_;
  cudnnRNNDescriptor_t rnn_desc_;
  cudnnRNNMode_t mode_;
  cudnnDirectionMode_t direction_;
  cudnnRNNInputMode_t input_mode_;
  cudnnDropoutDescriptor_t dropout_desc_;
  Storage::Handle reserve_space_, temp_space_;
  uint64_t seed_ = 17 + rand() % 4096;  // NOLINT(runtime/threadsafe_fn)
  size_t workspace_byte_, reserve_space_byte_, dropout_byte_;
  int workspace_size_;
  std::vector<cudnnTensorDescriptor_t> x_desc_vec_, y_desc_vec_, dx_desc_vec_, dy_desc_vec_;
  #if USE_CUDNN_LSTM_PROJ
  cudnnRNNDataDescriptor_t x_data_desc_, y_data_desc_, dx_data_desc_, dy_data_desc_;
  #endif
  cudnnTensorDescriptor_t hx_desc_, cx_desc_;
  cudnnTensorDescriptor_t hy_desc_, cy_desc_;
  cudnnTensorDescriptor_t dhx_desc_, dcx_desc_;
  cudnnTensorDescriptor_t dhy_desc_, dcy_desc_;

  cudnnFilterDescriptor_t w_desc_, dw_desc_;
  // Allow TensorCore algo policy
  bool cudnn_tensor_core_;

  #if CUDNN_MAJOR >= 5
  cudnnTensorFormat_t format_;
  #endif
  #endif
  bool init_space_, temp_init_space_;
  size_t reserve_cpu_space_size_, temp_cpu_space_size_;
  Storage::Handle reserve_cpu_space_, temp_cpu_space_;
};  //  class RNNOp

static OpStatePtr CreateRNNState(const nnvm::NodeAttrs &attrs,
                                 const Context ctx,
                                 const mxnet::ShapeVector &in_shapes,
                                 const std::vector<int> &in_types) {
  const RNNParam& param = nnvm::get<RNNParam>(attrs.parsed);
  OpStatePtr state = OpStatePtr();
  MSHADOW_REAL_TYPE_SWITCH(in_types[rnn_enum::kData], DType, {
    if (ctx.dev_type == kGPU) {
      state = OpStatePtr::Create<RNNOp<gpu, DType>>(param, ctx);
    } else {
      state = OpStatePtr::Create<RNNOp<cpu, DType>>(param, ctx);
    }
  });
  return state;
}

template<typename xpu>
void RNNStatefulCompute(const OpStatePtr& state,
                        const OpContext& ctx,
                        const std::vector<TBlob>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<TBlob>& outputs) {
  int dtype = inputs[rnn_enum::kData].type_flag_;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    RNNOp<xpu, DType>& op = state.get_state<RNNOp<xpu, DType>>();
    op.Forward(ctx, inputs, req, outputs);
  });
}

#if MXNET_USE_MKLDNN == 1
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
    RNNOp<cpu, DType>& op = state_ptr.get_state<RNNOp<cpu, DType>>();
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
      op.has_cache = false;
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
        if (L > 1 && D == 1) {
          auto user_src_layer_md = mkldnn::memory::desc(
              { src_layer_tz }, mkldnn_dtype, mkldnn::memory::format::tnc);
          auto user_src_layer_memory = mkldnn::memory({ user_src_layer_md, cpu_engine });
          op.x_memory.push_back(user_src_layer_memory);

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

          auto user_src_layer_md = mkldnn::memory::desc(
              { src_layer_tz }, mkldnn_dtype, mkldnn::memory::format::tnc);
          auto user_src_layer_memory = mkldnn::memory({ user_src_layer_md, cpu_engine });
          op.x_memory.push_back(user_src_layer_memory);

          auto wx_md_n = mkldnn::memory::desc(
              { weights_layer_tz }, mkldnn_dtype, mkldnn::memory::format::ldgoi);
          auto wh_md_n = mkldnn::memory::desc(
              { weights_iter_tz }, mkldnn_dtype, mkldnn::memory::format::ldgoi);

          for (int l = 0; l < L; l++) {
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
            workptr = bias_n + D * ngates * H;
          }

          DType* wx_n = workptr;  //  D * ngates * (D * H) * H
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
#endif
/*
index description
0: x
1: w
2: hx
3: y
4: dy
5: hy
6: dhy
7: cx
8: cy
9: dcy
*/
template<typename xpu>
void RNNStatefulGradCompute(const OpStatePtr& state,
                            const OpContext& ctx,
                            const std::vector<TBlob>& inputs,
                            const std::vector<OpReqType>& req,
                            const std::vector<TBlob>& outputs) {
  std::vector<TBlob> in_data(inputs.begin(), inputs.begin() + 3);
  std::vector<TBlob> out_data{inputs[3]};
  std::vector<TBlob> out_grad{inputs[4]};
  const std::vector<TBlob> &in_grad = outputs;

  int dtype = inputs[rnn_enum::kData].type_flag_;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    RNNOp<xpu, DType>& op = state.get_state<RNNOp<xpu, DType>>();
    const RNNParam& param = op.param_;
    int index = 5;
    if (param.state_outputs) {
      out_data.push_back(inputs[index++]);
      out_grad.push_back(inputs[index++]);
    }

    if (param.mode == rnn_enum::kLstm) {
      in_data.push_back(inputs[index++]);
      if (param.state_outputs) {
        out_data.push_back(inputs[index++]);
        out_grad.push_back(inputs[index]);
      }
    }

    op.Backward(ctx, out_grad, in_data, out_data, req, in_grad);
  });
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_RNN_INL_H_
