# -*- coding: utf-8 -*-
from nn_sdk import *
import os
path_dir = os.path.dirname(__file__)
config = {
    "model_dir": r'./model1/model.ckpt',
    "aes":{
        "use":False,
        "key":bytes([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]),
        "iv":bytes([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]),
    },
    "log_level": 4,# fatal 1 , error 2 , info 4 , debug 8
    'engine':0, # 0 tensorflow,  1 onnx , 2  tensorrt , 3 fasttext
    "device_id": 0,
    'tf':{
        #tensorflow2 ConfigProto无效
        "ConfigProto": {
            "log_device_placement": False,
            "allow_soft_placement": True,
            "gpu_options": {
                "allow_growth": True
            },
            "graph_options":{
                "optimizer_options":{
                    "global_jit_level": 1
                }
            },
        },
        "engine_version": 1, # tensorflow版本
        "model_type": 1,# 0 pb , 1 ckpt
        "saved_model":{ # 当model_type为pb模型有效, 普通pb use=False ， 如果是saved_model冻结模型 , 则需启用use并且配置tags
            'use': False, # 是否启用saved_model
            'tags': ['serve'],
            'signature_key': 'serving_default',
        },
        "fastertransformer":{
            "use": False,
            "cuda_version":"11.3", #当前依赖 tf2pb,支持10.2 11.3 ,
        }
    },
    "graph": [
        {
            "input": [
                {"node":"x1:0"},{"node":"x2:0"}
            ],
            "output": [
                {"node":"pred_ids:0",},
            ],
        }
    ]}


import copy
config2 = copy.deepcopy(config)

config2["model_dir"] = os.path.join(path_dir,'model2/model.ckpt')




seq_length = 10
input_ids = [[1] * seq_length]
input_mask = [[1] * seq_length]
sdk_inf = csdk_object(config)
if sdk_inf.valid():
    net_stage = 0
    ret, out = sdk_inf.process(net_stage, input_ids,input_mask)
    print(out[0].shape)
    sdk_inf.close()

seq_length = 10
input_ids = [[1] * seq_length]
input_mask = [[1] * seq_length]
sdk_inf = csdk_object(config2)
if sdk_inf.valid():
    net_stage = 0
    ret, out = sdk_inf.process(net_stage, input_ids,input_mask)
    print(out[0].shape)
    sdk_inf.close()

