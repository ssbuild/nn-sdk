import sys
from tf_csdk import sdk_new,sdk_delete,sdk_version,sdk_process
import numpy as np

class C_test:
    def __init__(self):
        conf = {
                "model_dir":r'D:\pytorch_model\bert_ner\dhcc_model_2.3.0-rc2',
                #"model_dir": r'E:\algo_text\nn_csdk\nn_csdk\py_test_ckpt\model.ckpt',
                "device_id":0,
                "tf":{
                    "ConfigProto": {
                        "log_device_placement": False,
                        "allow_soft_placement": True,
                        "gpu_options": {
                            "allow_growth": True
                        },
                    },
                    "model_type": 0,  # 0 pb , 1 ckpt
                    "engine_version": 1,
                },
                 "graph":[
                     {
                         "input":["input_ids:0"],
                         "output":["pred_ids:0"]
                     }
                 ]}


        self.handle = sdk_new(conf)
        print("sdk_new ", self.handle)

    def close(self):
        if self.handle:
            print("sdk_delete create ", self.handle)
            sdk_delete(self.handle)
            self.handle = None

    def test(self):

        input_ids = [
            [1.] * 256
        ]
        input_mask = [
            [1] * 256
        ]

        print(np.asarray(input_ids).shape)
        print(np.asarray(input_mask).shape)
        net_stage = 0
        #input_mask
        ret,out = sdk_process(self.handle,net_stage,input_ids,input_mask)
        print(ret)
        print(out)

        # for i in range(100):
        #     ret, out = sdk_process(self.handle, net_stage, input_ids, input_mask)
        #     print(ret)
    def __del__(self):
        self.close()



d = C_test()

d.test()

d.close()