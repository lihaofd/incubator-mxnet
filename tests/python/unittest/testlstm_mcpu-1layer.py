# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# pylint: skip-file
import numpy as np
import mxnet as mx
import math
import random
import itertools
from numpy.testing import assert_allclose, assert_array_equal
from mxnet.test_utils import *
from mxnet.base import py_str
import unittest

def check_rnn_consistency(cell1, T, N, I, H):
    xpu = mx.cpu()
    warmup = 3
    runtimes = 10

    dshape = (N, T, I)
    data = mx.sym.Variable('data')

    Y1, _ = cell1.unroll(T, data, layout='NTC', merge_outputs=True)
    mod1 = mx.mod.Module(Y1, label_names=None, context=xpu)
    mod1.bind(data_shapes=[('data', dshape)], label_shapes=None, inputs_need_grad=True)

    mod1.init_params()
 
    x = mx.random.uniform(shape=dshape)
    dy = mx.random.uniform(shape=(N, T, H))
    
    batch=mx.io.DataBatch(data=[x])
    fpa = open("out.txt", "w")
    
    # check inference
    
    for k in range(0 , warmup):        
        mod1.forward(batch, is_train=False)
        print >> fpa, mod1.get_outputs()[0][0][0][0]
 
    startTime = time.time()
    for k in range(0 , runtimes):        
        mod1.forward(batch, is_train=False)
        print >> fpa, mod1.get_outputs()[0][0][0][0]

    print ("-- use time %.8s seconds for %d Intelrnn infer---\n" % (time.time()-startTime, runtimes))

def test_lstm():
    T, N, I, H = 300, 20, 800, 800

    fused = mx.rnn.FusedRNNCell(H, num_layers=1, mode='lstm', get_next_state=True, prefix='')
    check_rnn_consistency(fused, T, N, I, H)

def test_lstm_bi():
    T, N, I, H = 300, 20, 800, 800
    fused = mx.rnn.FusedRNNCell(H, num_layers=1, mode='lstm',
                                bidirectional=True, get_next_state=True, prefix='')

    check_rnn_consistency(fused, T, N, I, H)

if __name__ == '__main__':
  test_lstm()
  test_lstm_bi()