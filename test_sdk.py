# -*- coding: utf-8 -*-
from nn_sdk import *
config = {
    "model_dir": r'/root/model.ckpt',
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
    'onnx':{
        "engine_version": 1,# onnxruntime 版本
    },
    'trt':{
        "engine_version": 8,# tensorrt 版本
        "enable_graph": 0,
    },
    'fasttext': {
        "engine_version": 0,# fasttext主版本
        "threshold":0, # 预测k个标签的阈值
        "k":1, # 预测k个标签
        "dump_label": 1, #输出内部标签，用于上层解码
        "predict_label": 1, #获取预测标签 1  , 获取向量  0
    },
    "graph": [
        {
            # 对于Bert模型 shape [max_batch_size,max_seq_lenth],
            # 其中max_batch_size 用于c++ java开辟输入输出缓存,输入不得超过max_batch_size，对于python没有作用，取决于上层用户真实输入
            # python限制max_batch_size 在上层用户输入做
            # 对于fasttext node 对应name可以任意写，但不能少
            "input": [
                {"node":"input_ids:0", "data_type":"int64", "shape":[1, 256]},
                {"node":"input_mask:0", "data_type":"int64", "shape":[1, 256]}
            ],
            "output": [
                {"node":"pred_ids:0", "data_type":"int64", "shape":[1, 256]},
            ],
        }
    ]}

seq_length = 256
input_ids = [[1] * seq_length]
input_mask = [[1] * seq_length]
sdk_inf = csdk_object(config)
if sdk_inf.valid():
    net_stage = 0
    ret, out = sdk_inf.process(net_stage, input_ids,input_mask)
    print(ret)
    print(out)
    sdk_inf.close()
