## nn-sdk 
    前言:
        支持开发语言c/c++,python,java
        支持推理引擎tensorflow(v1,v2) onnxruntime tensorrt,fasttext 注:tensorrt 7,8测试通过(建议8),目前tensorrt只支持linux系统
        支持多子图,支持图多输入多输出, 支持pb [tensorflow 1,2] , ckpt [tensorflow] , trt [tensorrt] , fasttext
        支持fastertransformer pb [32精度 相对于传统tf,加速1.9x] 
        pip install tf2pb  , 进行模型转换,tf2pb pb模型转换参考: https://pypi.org/project/tf2pb
        模型加密参考test_aes.py,目前支持tensorflow 1 pb模型 , onnx模型 , tensorrt fasttext模型加密
        推荐环境ubuntu系列 centos7 centos8 windows系列
        python (test_py.py) , c语言 (test.c) , java语言包 (nn_sdk.java)
        更多使用参见: https://github.com/ssbuild/nn-sdk
    instructions:
        Support development languages c/c++, python, java
        Support inference engine tensorflow (v1, v2) onnxruntime tensorrt, fasttext Note: tensorrt 7, 8 passed the test (recommended 8), currently tensorrt only supports linux system
        Support multiple subgraphs, support multiple input and multiple output graphs, support pb [tensorflow 1,2] , ckpt [tensorflow] , trt [tensorrt] , fasttext
        Support fastertransformer pb [32 precision compared to traditional tf, speed up 1.9x]
        pip install tf2pb , model conversion, tf2pb pb model conversion reference: https://pypi.org/project/tf2pb
        Model encryption reference test_aes.py, currently supports tensorflow 1 pb model, onnx model, tensorrt fasttext model encryption
        Recommended environmentubuntu series centos7 centos8 windows series
        python (test_py.py) , c language (test.c) , java language package (nn_sdk.java)
        For more usage see: https://github.com/ssbuild/nn-sdk

    config:
        aes: 加密参考test_aes.py
        engine: 
            0: tensorflow 
            1: onnx 
            2: tensorrt 
            3: fasttext
        log_level: 
            0: fatal 
            2: error 
            4: warn
            8: info 
            16: debug
        model_type: tensorflow model type
                0: pb format 
                1: ckpt format
        fastertransformer:
            fastertransformer算子,模型转换参考tf2pb, 参考 https://pypi.org/project/tf2pb
        ConfigProto: tensorflow 显卡配置
        device_id: GPU id
        engine_major: 推理引擎主版本 tf 0,1  tensorrt 7 或者 8 , fasttext 0
        engine_minor: 推理引擎次版本
        graph: 多子图配置 
            node: 例子: tensorflow 1 input_ids:0 ,  tensorflow 2: input_ids , onnx: input_ids
            dtype: 节点的类型根据模型配置，对于c++/java支持 int int64 long longlong float double str
            shape:  尺寸维度
    更新详情:
    2022-07-28 enable tf1 reset_default_graph
    2022-06-23 split tensorrt to trt_sdk , optimize onnx engine and modify onnx engine reload bug.
    2022-01-21 modify define graph shape contain none and modity demo note,modity a tensorflow 2 infer dtype bug,
               remove a deprecationWarning in py>=3.8
    2021-12-09 graph data_type 改名 dtype , 除fatal info err debug 增加warn
    2021-11-25 修复nn-sdk非主动close, close小bug.
    2021-10-21 修复fastext推理向量维度bug
    2021-10-16 优化 c++/java接口,可预测动态batch
    2021-10-07 增加 fasttext 向量和标签推理

## python demo


```python

# -*- coding: utf-8 -*-
from nn_sdk import *
config = {
    "model_dir": r'/root/model.pb',
    "aes":{
        "use":False,
        "key":bytes([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]),
        "iv":bytes([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]),
    },
    "log_level": 8,# 0 fatal , 2 error , 4 warn, 8 info , 16 debug
    'engine':0, # 0 tensorflow,  1 onnx , 2  tensorrt , 3 fasttext
    "device_id": 0,
    'tf':{
        "ConfigProto": {
            "log_device_placement": False,
            "allow_soft_placement": True,
            "gpu_options": {"allow_growth": True},
            "graph_options":{
                "optimizer_options":{"global_jit_level": 1}
            },
        },
        "engine_major": 1, # tensorflow engine majar version
        "is_reset_graph": 1, # 1 reset_default_graph , 0 do nothing
        "model_type": 0,# 0 pb , 1 ckpt
        #配置pb模型
        "saved_model":{
            # model_type为 1 pb , 模型有效,
            # 模型是否是是否启用saved_model冻结 , 如果是,则 use=True并且配置tags
            # 普通 freeze pb , use = False
            'enable': False, # 是否启用saved_model
            'tags': ['serve'],
            'signature_key': 'serving_default',
        },
        "fastertransformer":{"enable": False}
    },
    'onnx':{
        'tensorrt': True, #是否启用tensorrt算子
    },
    'trt':{
        #pip install trt-sdk , support tensorrt 7.2 8.0 8.2 8.4 or more new
        "engine_major": 8,# 7 or 8
        "engine_minor": 0,
        "enable_graph": 0,
    },
    'fasttext': {
        "engine_major": 0,
        "threshold":0, # 预测k个标签的阈值
        "k":1, # 预测k个标签 score >= threshold
        "dump_label": 1, #输出内部标签，用于上层解码
        "predict_label": 1, #获取预测标签 1  , 获取向量  0
    },
    "graph": [
        {
            # 对于Bert模型 shape [max_batch_size,max_seq_lenth],
            # 其中max_batch_size 用于c++ java开辟输入输出缓存,输入不得超过max_batch_size，对于python没有作用，取决于上层用户真实输入
            # python 限制max_batch_size 在上层用户输入做 , dtype and shape are not necessary for python
            # 对于fasttext node 对应name可以任意写，但不能少
            # dtype must be in [int int32 int64 long longlong uint uint32 uint64 ulong ulonglong float float32 float64 double str]
            "input": [
                {
                    "node":"input_ids:0",
                    #"dtype":"int64",
                    #"shape":[1, 256] #Python may be empty, c/c++ java must exist , it will be used to alloc mem
                },
                {
                    "node":"input_mask:0",
                    #"dtype":"int64",
                    #"shape":[1, 256] #Python may be empty , c/c++ java must exist , it will be used to alloc mem
                }
            ],
            "output": [
                {
                    "node":"pred_ids:0",
                    #"dtype":"int64",
                    #"shape":[1, 256] #Python may be empty , c/c++ java must exist , it will be used to alloc mem
                },
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

```





## java demo


```java
    package nn_sdk;

//输入缓冲区 自定义 可自定义改
class nn_buffer_batch{
	  //输入 输出内存节点，名字跟图配置一样，根据图对象修改。
	public float [] input_ids = null;//推理图的输入,
	public float[] pred_ids =   null;//推理的结果保存

	public int batch_size = 1;
	public nn_buffer_batch(int batch_size_){
		this.input_ids = new float[batch_size_ * 10];
		this.pred_ids =  new float[batch_size_ * 10];
		this.batch_size = batch_size_;
		for(int i =0;i<1 * 10;i++) {
			this.input_ids[i] = 1;
			this.pred_ids[i] = 0;
		}
	}
}


//包名必须是nn_sdk
public class nn_sdk {
	//推理函数
	public native static int  sdk_init_cc();
	public native static int  sdk_uninit_cc();
	public native static long sdk_new_cc(String json);
	public native static int  sdk_delete_cc(long handle);
	//nn_buffer_batch 类
	public native static int sdk_process_cc(long handle, int net_state,int batch_size, nn_buffer_batch buffer);

	static {
		//动态库的绝对路径windows是engine_csdk.pyd , linux是 engine_csdk.so
		System.load("engine_csdk.pyd");
	}

	public static void main(String[] args){
		System.out.println("java main...........");

	   nn_sdk instance = new nn_sdk();

	   nn_buffer_batch buf = new nn_buffer_batch(2);
	   sdk_init_cc();

	   String json = "{\r\n"
	   + "    \"model_dir\": r'model.ckpt',\r\n"
	   + "    \"aes\":{\r\n"
	   + "        \"enable\":False,\r\n"
	   + "        \"key\":bytes([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]),\r\n"
	   + "        \"iv\":bytes([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]),\r\n"
	   + "    },\r\n"
	   + "    \"log_level\": 4,# fatal 1 , error 2 , info 4 , debug 8\r\n"
	   + "    'engine':0, # 0 tensorflow,  1 onnx , 2  tensorrt , 3 fasttext\r\n"
	   + "    \"device_id\": 0,\r\n"
	   + "    'tf':{\r\n"
	   + "        #tensorflow2 ConfigProto无效\r\n"
	   + "        \"ConfigProto\": {\r\n"
	   + "            \"log_device_placement\": False,\r\n"
	   + "            \"allow_soft_placement\": True,\r\n"
	   + "            \"gpu_options\": {\r\n"
	   + "                \"allow_growth\": True\r\n"
	   + "            },\r\n"
	   + "            \"graph_options\":{\r\n"
	   + "                \"optimizer_options\":{\r\n"
	   + "                    \"global_jit_level\": 1\r\n"
	   + "                }\r\n"
	   + "            },\r\n"
	   + "        },\r\n"
	   + "        \"engine_version\": 1, # tensorflow版本\r\n"
	   + "        \"model_type\": 1,# 0 pb , 1 ckpt\r\n"
	   + "        \"saved_model\":{ # 当model_type为pb模型有效, 普通pb enable=False ， 如果是saved_model冻结模型 , 则需启用enable并且配置tags\r\n"
	   + "            'enable': False, # 是否启用saved_model\r\n"
	   + "            'tags': ['serve'],\r\n"
	   + "            'signature_key': 'serving_default',\r\n"
	   + "        },\r\n"
	   + "        \"fastertransformer\":{\r\n"
	   + "            \"enable\": False,\r\n"
	   + "        }\r\n"
	   + "    },\r\n"
	   + "    'onnx':{\r\n"
	   + "        \"engine_version\": 1,# onnxruntime 版本\r\n"
	   + "    },\r\n"
	   + "    'trt':{\r\n"
	   + "        \"engine_version\": 8,# tensorrt 版本\r\n"
	   + "        \"enable_graph\": 0,\r\n"
	   + "    },\r\n"
	   + "    'fasttext': {\r\n"
	   + "        \"engine_version\": 0,# fasttext主版本\r\n"
	   + "        \"threshold\":0, # 预测k个标签的阈值\r\n"
	   + "        \"k\":1, # 预测k个标签\r\n"
	   + "        \"dump_label\": 1, #输出内部标签，用于上层解码\r\n"
	   + "        \"predict_label\": 1, #获取预测标签 1  , 获取向量  0\r\n"
	   + "    },\r\n"
	   + "    \"graph\": [\r\n"
	   + "        {\r\n"
	   + "            # 对于Bert模型 shape [max_batch_size,max_seq_lenth],\r\n"
	   + "            # 其中max_batch_size 用于c++ java开辟输入输出缓存,输入不得超过max_batch_size，对于python没有作用，取决于上层用户真实输入\r\n"
	   + "            # python限制max_batch_size 在上层用户输入做\r\n"
	   + "            # 对于fasttext node 对应name可以任意写，但不能少\r\n"
	   + "            \"input\": [\r\n"
	   + "                {\"node\":\"input_ids:0\", \"data_type\":\"float\", \"shape\":[1, 10]},\r\n"
	   + "            ],\r\n"
	   + "            \"output\": [\r\n"
	   + "                {\"node\":\"pred_ids:0\", \"data_type\":\"float\", \"shape\":[1, 10]},\r\n"
	   + "            ],\r\n"
	   + "        }\r\n"
	   + "    ]}";



	  System.out.println(json);

	  long handle = sdk_new_cc(json);
	  System.out.printf("handle: %d\n",handle);

	  int code = sdk_process_cc(handle,0,buf.batch_size,buf);
	  System.out.printf("sdk_process_cc %d \n" ,code);
	  if(code == 0) {
		  for(int i = 0;i<20 ; i++) {
			  System.out.printf("%f ",buf.pred_ids[i]);
		  }
		  System.out.println();
	  }
	  sdk_delete_cc(handle);
	   sdk_uninit_cc();
	   System.out.println("end");
	}
}
```



## c/c++  demo


```commandline

#include <stdio.h>
#include "nn_sdk.h"

int main(){
    if (0 != sdk_init_cc()) {
		return -1;
	}
    printf("配置参考 python.........\n");
	const char* json_data = "{\n\
    \"model_dir\": \"/root/model.ckpt\",\n\
    \"log_level\":8, \n\
     \"device_id\":0, \n\
    \"tf\":{ \n\
         \"ConfigProto\": {\n\
            \"log_device_placement\":0,\n\
            \"allow_soft_placement\":1,\n\
            \"gpu_options\":{\"allow_growth\": 1}\n\
        },\n\
        \"engine_version\": 1,\n\
        \"model_type\":1 ,\n\
    },\n\
    \"graph\": [\n\
        {\n\
            \"input\": [{\"node\":\"input_ids:0\", \"data_type\":\"float\", \"shape\":[1, 10]}],\n\
            \"output\" : [{\"node\":\"pred_ids:0\", \"data_type\":\"float\", \"shape\":[1, 10]}]\n\
        }\n\
    ]\n\
}";
	printf("%s\n", json_data);
	auto handle = sdk_new_cc(json_data);
	const int INPUT_NUM = 1;
	const int OUTPUT_NUM = 1;
	const int M = 1;
	const int N = 10;
	int *input[INPUT_NUM] = { 0 };
	float* result[OUTPUT_NUM] = { 0 };
	int element_input_size = sizeof(int);
	int element_output_size = sizeof(float);
	for (int i = 0; i < OUTPUT_NUM; ++i) {
		result[i] = (float*)malloc(M * N * element_output_size);
		memset(result[i], 0, M * N * element_output_size);
	}
	for(int i =0;i<INPUT_NUM;++i){
		input[i] = (int*)malloc(M * N * element_input_size);
		memset(input[i], 0, M * N * element_input_size);
		for (int j = 0; j < N; ++j) {
			input[i][j] = i;
		}
	}

    int batch_size = 1;
	int code = sdk_process_cc(handle,  0 , batch_size, (void**)input,(void**)result);
	if (code == 0) {
		printf("result\n");
		for (int i = 0; i < N; ++i) {
			printf("%f ", result[0][i]);
		}
		printf("\n");
	}
	for (int i = 0; i < INPUT_NUM; ++i) {
		free(input[i]);
	}
	for (int i = 0; i < OUTPUT_NUM; ++i) {
		free(result[i]);
	}
	sdk_delete_cc(handle);
	sdk_uninit_cc();
	return 0;
}
```



## 模型加密模块

```commandline
# -*- coding: UTF-8 -*-

import sys
from nn_sdk.engine_csdk import sdk_aes_encode_decode

def test_string():
    data1 = {
        "mode":0,# 0 加密 ， 1 解密
        "key": bytes([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]),
        "iv": bytes([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]),
        "data": bytes([1,2,3,5,255])
    }

    code,encrypt = sdk_aes_encode_decode(data1)
    print(code,encrypt)

    data2 = {
        "mode":1,
        "key": bytes([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]),
        "iv": bytes([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]),
        "data": encrypt
    }

    code,plain = sdk_aes_encode_decode(data2)
    print(code,plain)

def test_encode_file(in_filename,out_filename):

    with open(in_filename,mode='rb') as f:
        data = f.read()
    if len(data) == 0 :
        return -1
    data1 = {
        "mode": 0,  # 0 加密 ， 1 解密
        "key": bytes([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
        "iv": bytes([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
        "data": bytes(data)
    }

    code, encrypt = sdk_aes_encode_decode(data1)
    if code != 0:
        return code
    with open(out_filename, mode='wb') as f:
        f.write(encrypt)
    return code
def test_decode_file(in_filename,out_filename):
    with open(in_filename, mode='rb') as f:
        data = f.read()
    if len(data) == 0:
        return -1
    data1 = {
        "mode": 1,  # 0 加密 ， 1 解密
        "key": bytes([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
        "iv": bytes([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
        "data": bytes(data)
    }

    code, plain = sdk_aes_encode_decode(data1)
    if code != 0:
        return code
    with open(out_filename, mode='wb') as f:
        f.write(plain)
    return code

test_encode_file(r'C:\Users\acer\Desktop\img\a.txt',r'C:\Users\acer\Desktop\img\a.txt.encode')
test_decode_file(r'C:\Users\acer\Desktop\img\a.txt.encode',r'C:\Users\acer\Desktop\img\a.txt.decode')

```