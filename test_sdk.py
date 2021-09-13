# -*- coding: utf-8 -*-
from nn_sdk import *
'''
    前言:
        当前支持开发语言c/c++,python,java
        当前支持推理引擎tensorflow(v1,v2) onnxruntime tensorrt 注:tensorrt 7,8测试通过(建议8), 目前tensorrt只支持linux系统
        当前支持多子图,支持图多输入多输出, 支持pb [tensorflow 1,2] , ckpt [tensorflow] , trt [tensorrt]
        当前支持fastertransformer pb [32精度 相对于传统tf,加速1.9x] ,安装 pip install tf2pb  , 进行模型转换
        tf2pb pb模型转换参考: https://pypi.org/project/tf2pb
        模型加密参考test_aes.py,目前支持tensorflow 1 pb模型 , onnx模型 , tensorrt模型加密
        推荐环境ubuntu16 ubuntu18  ubuntu20 centos7 centos8 windows系列
        python (test_py.py) , c语言 (test.c) , java语言包 (nn_sdk.java)
        使用过程中遇到问题或者有建议可加qq group: 759163831
        更多使用参见: https://github.com/ssbuild

    python 推理demo
    config 字段介绍:
        aes: 加密参考test_aes.py,目前支持tensorflow 1 pb模型 , onnx模型 , tensorrt模型加密
        engine: 推理引擎 0: tensorflow , 1: onnx , 2: tensorrt
        log_level: 日志类型 0 fatal , 2 error , 4 info , 8 debug
        model_type: tensorflow 模型类型, 0 pb format , 1 ckpt format
        fastertransformer:  fastertransformer算子选项, 参考 https://pypi.org/project/tf2pb
        ConfigProto: tensorflow 显卡配置
        device_id: GPU id
        engine_version: 推理引擎主版本 tf 0,1  tensorrt 7 或者 8 , 需正确配置
        graph: 多子图配置 
            node: 例子: tensorflow 1 input_ids:0 ,  tensorflow 2: input_ids , onnx: input_ids
            data_type: 节点的类型根据模型配置，对于c++/java支持 int int64 long longlong float double ,对于python没有限制
            shape:  尺寸维度
'''
config = {
    "model_dir": r'/root/model.ckpt',
    "aes":{
        "use":False,
        "key":bytes([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]),
        "iv":bytes([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]),
    },
    "log_level": 4,# fatal 1 , error 2 , info 4 , debug 8
    'engine':0, # 0 tensorflow,  1 onnx , 2  tensorrt
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
    "graph": [
        {
            # 对于Bert模型 shape [max_batch_size,max_seq_lenth],
            # 其中max_batch_size 用于c++ java开辟输入输出缓存,输入不得超过max_batch_size，对于python没有作用，取决于上层用户真实输入
            # python限制max_batch_size 在上层用户输入做
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
