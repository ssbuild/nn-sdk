# -*- coding: utf-8 -*-
import numpy as np
from nn_sdk import csdk_object

'''
    前言: 
        当前支持开发语言c/c++,python,java
        当前支持推理引擎tensorflow(v1,v2) onnxruntime
        当前支持多子图,支持图多输入多输出，支持tensorflow 1 pb , tensorflow 2 pb , tensorflow ckpt
        当前支持tensorflow 1.x pb模型和onnx模型 aes加密 , 模型加密参考test_aes.py
        推荐环境ubuntu16 ubuntu18  ubuntu20 centos7 centos8 windows系列
        python (test_py.py) , c包 (test.c) , java包 (nn_sdk.java)
        qq group: 759163831
'''
'''
    python 推理demo
    config 字段介绍:
        aes: 模型加密配置，目前支持tensorflow 1 pb 模型
        engine: 推理引擎 0: tensorflow , 1: onnx
        log_level: 日志类型 0 fatal , 2 error , 4 info , 8 debug
        model_type: tensorflow时有效, 0 pb format   if 1 ckpt format
        ConfigProto: tensorflow时有效
        device_id: onnx 有效
        graph_inf_version: tensorflow version [0,1] or onnxruntime 1
        graph: 多子图配置 
            node: 例子: tensorflow 1 input_ids:0 ,  tensorflow 2: input_ids , onnx: input_ids
            data_type: 节点的类型根据模型配置，支持 int int64 long longlong float double 
            shape: 节点尺寸
            java 和 c 包不可缺少 data_type,shape字段

'''
config = {
    "model_dir": r'/data/finalmodel/2021/bert_ner_2021/model.onnx',
    "aes": {
        "use": False,
        "key": bytes([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
        "iv": bytes([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
    },
    "log_level": 8,
    'engine': 1,
    "device_id": 0,
    'onnx': {
        "engine_version": 1,
    },
    "graph": [
        {
            "input": [
                {"node": "input_ids", "data_type": "int", "shape": [1, 340]},
                {"node": "input_mask", "data_type": "int", "shape": [1, 340]},
            ],
            "output": [
                {"node": "pred_ids", "data_type": "int", "shape": [1, 340]},
            ],
        }
    ]}

seq_length = 340
input_ids = [[1] * seq_length]
input_mask = [[1] * seq_length]
sdk_inf = csdk_object(config)
if sdk_inf.valid():
    net_stage = 0
    ret, out = sdk_inf.process(net_stage, input_ids,input_mask)

    print(ret)
    print(out)
    if isinstance(out,np.ndarray):
        print(out.shape)
    sdk_inf.close()