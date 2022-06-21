# nn_sdk

### nn_sdk 简介
    基于tensorflow(v1 ,v2),onnx,tensorrt,fasttext神经网络高性能推理接口
    深度学习模型推理 支持python  c/c++  java
    pip包地址:
    https://pypi.org/project/nn-sdk/#description

### 配置
```commandline
当前支持开发语言c/c++,python,java
当前支持推理引擎tensorflow(v1,v2) onnxruntime tensorrt,fasttext 注:tensorrt 7,8测试通过(建议8),目前tensorrt只支持linux系统
当前支持多子图,支持图多输入多输出, 支持pb [tensorflow 1,2] , ckpt [tensorflow] , trt [tensorrt] , fasttext
当前支持fastertransformer pb [32精度 相对于传统tf,加速1.9x] ,安装 pip install tf2pb  , 进行模型转换
tf2pb pb模型转换参考: https://pypi.org/project/tf2pb
模型加密参考test_aes.py,目前支持tensorflow 1 pb模型 , onnx模型 , tensorrt fasttext模型加密
推荐环境ubuntu16 ubuntu18 ubuntu20 centos7 centos8 windows系列
python (test_py.py) , c语言 (test.c) , java语言包 (nn_sdk.java)
使用过程中遇到问题或者有建议可加qq group: 759163831
更多使用参见: https://github.com/ssbuild

python 推理demo
config 字段介绍:
    aes: 加密参考test_aes.py
    engine: 推理引擎 0: tensorflow , 1: onnx , 2: tensorrt 3: fasttext
    log_level: 日志类型 0 fatal , 2 error , 4 info , 8 debug
    model_type: tensorflow 模型类型, 0 pb format , 1 ckpt format
    fastertransformer:  fastertransformer算子选项, 参考 https://pypi.org/project/tf2pb
    ConfigProto: tensorflow 显卡配置
    device_id: GPU id
    engine_version: 推理引擎主版本 tf 0,1  tensorrt 7 或者 8 , fasttext 0需正确配置
    graph: 多子图配置 
        node: 例子: tensorflow 1 input_ids:0 ,  tensorflow 2: input_ids , onnx: input_ids
        data_type: 节点的类型根据模型配置，对于c++/java支持 int int64 long longlong float double str
        shape:  尺寸维度
更新详情:
2021-10-16 优化 c++/java接口,可预测动态batch
2021-10-7 增加 fasttext 向量和标签推理

```

### 推理测试
    
    
```python
#例子: 推理 train 目录model.ckpt
import numpy as np
from nn_sdk import csdk_object

config = {
    "model_dir": r'./train/model.ckpt',
    "aes": {
        "use": False,
        "key": bytes([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
        "iv": bytes([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
    },
    "log_level": 8,
    'engine': 0,
    "device_id": 0,
     'tf':{
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
        "engine_version": 1, # tensorflow majar version , must be correct.
        "model_type": 1,# 0 pb , 1 ckpt
        #配置pb模型
        "saved_model":{
            # model_type为 1 pb , 模型有效,
            # 模型是否是是否启用saved_model冻结 , 如果是,则 use=True并且配置tags
            # 普通 freeze pb , use = False
            'use': False, # 是否启用saved_model
            'tags': ['serve'],
            'signature_key': 'serving_default',
        },
        "fastertransformer":{
            "use": False,
            "cuda_version":"11.3", #pip install tf2pb ,支持10.2 11.3 ,
        }
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
                {"node": "x1:0", "dtype": "float32"},
                {"node": "x2:0", "dtype": "float32"},
            ],
            "output": [
                {"node": "pred_ids:0", "dtype": "float32"},
            ],
        }
    ]
}

batch_size = 1
seq_length = 10
x1 = np.random.randint(1,10,size=(batch_size,seq_length))
x2 = np.random.randint(1,10,size=(batch_size,seq_length))

inputs = (x1,x2)

sdk_inf = csdk_object(config)
if sdk_inf.valid():
    net_stage = 0
    ret, out = sdk_inf.process(net_stage, *inputs)

    print(x1)
    print(x2)
    print(ret)
    print(out)
    sdk_inf.close()
```











