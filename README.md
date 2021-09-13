# nn_sdk

tensorflow 训练模型推理 支持python  c/c++  java推理模型 , tf pb模型支持aes加密！
pip包地址:
https://pypi.org/project/nn-sdk/#description


        当前支持开发语言c/c++,python,java
        当前支持推理引擎tensorflow(v1,v2) onnxruntime tensorrt 注:tensorrt 7,8测试通过(建议8), 目前tensorrt只支持linux系统
        当前支持多子图,支持图多输入多输出, 支持pb [tensorflow 1,2] , ckpt [tensorflow] , trt [tensorrt]
        当前支持fastertransformer pb [32精度 相对于传统tf,加速1.9x] ,安装 pip install tf2pb  , 进行模型转换
        tf2pb pb模型转换参考: https://pypi.org/project/tf2pb
        模型加密参考test_aes.py,目前支持tensorflow 1 pb模型 , onnx模型 , tensorrt模型加密
        推荐环境ubuntu16 ubuntu18  ubuntu20 centos7 centos8 windows系列
        python (test_py.py) , c语言 (test.c) , java语言包 (nn_sdk.java)

