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

def check_rnn_consistency_int8(cell1, cell2, T, N, I, H):
    xpu = mx.cpu()
    warmup = 3
    runtimes = 10

    dshape = (N, T, I)
    data = mx.sym.Variable('data')

    Y1, _ = cell1.unroll(T, data, layout='NTC', merge_outputs=True)
    mod1 = mx.mod.Module(Y1, label_names=None, context=xpu)
    mod1.bind(data_shapes=[('data', dshape)], label_shapes=None, inputs_need_grad=True)

    Y2, _ = cell2.unroll(T, data, layout='NTC', merge_outputs=True)
    mod2 = mx.mod.Module(Y2, label_names=None, context=xpu)
    mod2.bind(data_shapes=[('data', dshape)], label_shapes=None, inputs_need_grad=True)

    mod1.init_params()
    args, auxs = mod1.get_params()
    args = cell1.unpack_weights(args)
    args = cell2.pack_weights(args)
    mod2.set_params(args, auxs)
 
    x = mx.random.uniform(shape=dshape)
    
    batch=mx.io.DataBatch(data=[x])
    # check inference
    fpa = open("out.txt", "w")
    for k in range(0 , warmup):        
        mod1.forward(batch, is_train=False)
        print >> fpa, mod1.get_outputs()[0][0][0][0]
 
    startTime = time.time()
    for k in range(0 , runtimes):        
        mod1.forward(batch, is_train=False)
        print >> fpa, mod1.get_outputs()[0][0][0][0]

    print ("-- use time %.8s seconds for %d Intelrnn infer---\n" % (time.time()-startTime, runtimes))

    for k in range(0 , warmup):        
        mod2.forward(batch, is_train=False)
        print >> fpa, mod1.get_outputs()[0][0][0][0]
 
    startTime = time.time()
    for k in range(0 , runtimes):        
        mod2.forward(batch, is_train=False)
        print >> fpa, mod2.get_outputs()[0][0][0][0]

    print ("-- use time %.8s seconds for %d grucell infer---\n" % (time.time()-startTime, runtimes))

def test_multiplegru():
    T, N, I, H = 300, 20, 800, 800
    fused = mx.rnn.FusedRNNCell(H, num_layers=5, mode='gru', get_next_state=True, prefix='')
    stack = mx.rnn.SequentialRNNCell()
    stack.add(mx.rnn.GRUCell(H, prefix='l0_'))
    stack.add(mx.rnn.GRUCell(H, prefix='l1_'))
    
    stack.add(mx.rnn.GRUCell(H, prefix='l2_'))
    stack.add(mx.rnn.GRUCell(H, prefix='l3_'))
    stack.add(mx.rnn.GRUCell(H, prefix='l4_'))
    
    check_rnn_consistency_int8(fused, stack, T, N, I, H)



if __name__ == '__main__':
    test_multiplegru()
