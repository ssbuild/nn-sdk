# -*- coding: utf-8 -*-
import numpy as np
from nn_sdk import csdk_object

config = {
    "model_dir": r"D:\tf_model\model_unsub.bin",
    "aes": {
        "use": False,
        "key": bytes([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
        "iv": bytes([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
    },
    "log_level": 8,
    'engine': 3,
    "device_id": 0,
    'fasttext': {
        "engine_version": 0,
        "threshold": 0,
        "k":1,
        "predict_label": 0,
    },
    "graph": [
        {
            "input": [
                {"node": "input_ids", "data_type": "str", "shape": [1, 10]},
            ],
            "output": [
                {"node": "pred_ids", "data_type": "float", "shape": [1,128]},
            ],
        }
    ]}

input_ids = ['2016 年 4 月 起 , 被告人 蓝 某某 为 获取 投注额 10% 的 提成 , 在 自己 经营 的 位于 佛山市 禅城区 张 槎 街道 上朗 村委会 长塘 大街 七巷 2 号 的 无牌 维修 店内 , 接受 他人 投注 “ 香港 六合彩 ” 聚众赌博 , 累计 接受 34 个 以上 的 参赌 人员 的 投注 。 2017 年 5 月 25 日 21 时许 , 被告人 蓝 某某 正在 该 维修 店内 接受 他人 投注 时 , 被 民警 当场 抓获 。 民警 从 现场 缴获 投注单 据 21 张 、 赌资 人民币 1100 元 和 被告人 蓝 某某 用于 微信 投注 的 手机 一台 等 。',
             '2014 年 7 月 19 日 12 时许 、 7 月 20 日 12 时许 , 被告人 谢 某某 在 其屋 的 二楼 , 容留 吸毒 人员 黎 某某 吸食毒品 海洛因 。 2014 年 7 月 22 日 21 时许 , 被告人 谢 某某 在 其屋 的 二楼 , 容留 吸毒 人员 黎 某某 和 “ 叠记 ” 吸食 海洛因 时 , 被 民警 抓获 并 当场 缴获 可疑 毒品 0.001 克 白色 粉末 、 吸筒 一个 、 刀片 一张 。 经 鉴定 , 该 可疑 毒品 0.001 克 白色 粉末 检出 海洛因 成分 。']

input_ids = ['2016 年 4 月 起']
sdk_inf = csdk_object(config)

if sdk_inf.valid():
    net_stage = 0
    ret, out = sdk_inf.process(net_stage, input_ids)
    print(ret)
    print(out)
    if isinstance(out[0],np.ndarray):
        print(out[0].shape)
    sdk_inf.close()