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

#if MXNET_USE_MKLDNN == 1

#include "../../nn/mkldnn/mkldnn_base-inl.h"
#include "../quantization_utils.h"
#include "../../nn/fully_connected-inl.h"

namespace mxnet {
namespace op {

// value + bias_value * (range1 / limit_range1) * (limit_range2 / range2)
struct QuantizedBiasAddKernel {
  MSHADOW_XINLINE static void Map(int i, size_t k, int32_t *out,
                                  const int8_t *bias, const float *min_out,
                                  const float *max_out, const float *min_bias,
                                  const float *max_bias) {
    typedef int32_t T1;
    typedef int8_t  T2;
    using mshadow::red::limits::MinValue;
    using mshadow::red::limits::MaxValue;
    float float_for_one_out_quant  =
      MaxAbs(*min_out, *max_out) / static_cast<double>(MaxValue<T1>());
    float float_for_one_bias_quant =
      MaxAbs(*min_bias, *max_bias) / static_cast<double>(MaxValue<T2>());
    if (float_for_one_out_quant != 0) {
      out[i] = (out[i] * float_for_one_out_quant +
                bias[i%k] * float_for_one_bias_quant) /
               float_for_one_out_quant;
    } else {
      LOG(INFO) << "WARNING: QuantizedBiasAddKernel float_for_one_out_quant is 0 !";
    }
  }
};

template<typename SrcType, typename DstType, typename CmpType>
void MKLDNNQuantizedFullyConnectedForward(const nnvm::NodeAttrs& attrs,
                                       const OpContext &ctx,
                                       const std::vector<NDArray> &in_data,
                                       const std::vector<OpReqType> &req,
                                       const std::vector<NDArray> &out_data) {
  const FullyConnectedParam& param = nnvm::get<FullyConnectedParam>(attrs.parsed);
  using namespace mshadow;
  using namespace mxnet_op;
  size_t num_inputs = param.no_bias ? 2 : 3;
  CHECK_EQ(in_data.size(),  num_inputs * 3);
  CHECK_EQ(out_data.size(), 3U);
  const NDArray& data   =  in_data[0];
  const NDArray& weight =  in_data[1];
  const NDArray& out    = out_data[0];
  TShape dshape = data.shape();
  TShape wshape = weight.shape();
  TShape oshape = out.shape();

  CHECK(in_data[0].dtype() == mshadow::kInt8
    && in_data[1].dtype() == mshadow::kInt8)
    << "mkldnn_quantized_FullyConnected op only supports int8 as input type";

  const float alpha = 1.0f;
  const float beta  = 0.0f;
  const CBLAS_OFFSET offsetc = CblasFixOffset;
  const MKL_INT8 oa = -128;
  const MKL_INT8 ob = 0;
  MKL_INT32 oc = 0;
  const int m = dshape[0], n = wshape[0], k = dshape.ProdShape(1, dshape.ndim());
  const int omp_threads = mxnet::engine::OpenMP::Get()->GetRecommendedOMPThreadCount();
  uint8_t* pDataNewRange = reinterpret_cast<uint8_t*>(malloc(m*k*sizeof(uint8_t)));

  #pragma omp parallel for num_threads(omp_threads)
  for (int i = 0; i < m * k; i++) {
    pDataNewRange[i] = data.data().dptr<int8_t>()[i] + 128;
  }

  cblas_gemm_s8u8s32(CblasRowMajor,
                     CblasNoTrans,
                     CblasTrans,
                     offsetc,
                     m,
                     n,
                     k,
                     alpha,
                     pDataNewRange,
                     k,
                     oa,
                     weight.data().dptr<int8_t>(),
                     k,
                     ob,
                     beta,
                     out.data().dptr<int32_t>(),
                     n,
                     &oc);

  free(pDataNewRange);
  Stream<cpu> *s = ctx.get_stream<cpu>();
  Kernel<QuantizationRangeForMultiplicationStruct, cpu>::Launch(s, 1,
     out_data[1].data().dptr<float>(), out_data[2].data().dptr<float>(),
     in_data[num_inputs].data().dptr<float>(),   in_data[num_inputs+1].data().dptr<float>(),
     in_data[num_inputs+2].data().dptr<float>(), in_data[num_inputs+3].data().dptr<float>());

  if (!param.no_bias) {
    const NDArray& bias = in_data[2];
    Kernel<QuantizedBiasAddKernel, cpu>::Launch(s, out.shape().Size(),
        n, out.data().dptr<int32_t>(), bias.data().dptr<int8_t>(),
        out_data[1].data().dptr<float>(), out_data[2].data().dptr<float>(),
         in_data[7].data().dptr<float>(),  in_data[8].data().dptr<float>());
  }
}

NNVM_REGISTER_OP(_contrib_quantized_fully_connected)
.set_attr<FComputeEx>("FComputeEx<cpu>",
    MKLDNNQuantizedFullyConnectedForward<int8_t, int32_t, int32_t>);


}  // namespace op
}  // namespace mxnet
#endif

