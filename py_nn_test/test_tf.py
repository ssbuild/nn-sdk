# -*- coding: utf-8 -*-
import numpy as np
from nn_sdk import csdk_object

config = {
    "model_dir": r'/root/model.ckpt',
    "aes": {
        "use": False,
        "key": bytes([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
        "iv": bytes([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
    },
    "log_level": 8,
    'engine': 0,
    "device_id": 0,
    'tf': {
        "model_type": 1,
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
            }
        },
        "engine_version": 1,
    },
    'onnx': {
        "engine_version": 1,
    },
    'trt': {
        'enable_graph': 0,
        "engine_version": 8,
    },
    "graph": [
        {
            "input": [
                {"node": "input_ids:0", "data_type": "int32", "shape": [1, 10]},
            ],
            "output": [
                {"node": "pred_ids:0", "data_type": "float", "shape": [1, 10]},
            ],
        }
    ]}

seq_length = 340
input_ids = [[1] * seq_length]
input_mask = [[1] * seq_length]
sdk_inf = csdk_object(config)
if sdk_inf.valid():
    net_stage = 0
    ret, out = sdk_inf.process(net_stage, input_ids)

    print(ret)
    print(out)
    if isinstance(out,np.ndarray):
        print(out.shape)
    sdk_inf.close()