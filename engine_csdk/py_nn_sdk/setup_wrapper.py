# -*- coding: utf-8 -*-
# @Time    : 2021/11/17 14:38
# @Author  : wyw

import os
import platform
import shutil
import version_config

def get_desc():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # long_description_str = title
    # long_description_str += '\n\n'

    long_description_str = ''

    def load_file(demo_file):
        with open(demo_file, mode='r', encoding='utf-8') as f:
            data_string = str(f.read())
        return data_string

    data_string = load_file(os.path.join(current_dir, 'readme.md'))
    long_description_str += data_string + '\n'

    return long_description_str


def copy_dep_lib():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    if platform.system().lower() == 'windows':
        file_map = {
            'src': [
                os.path.join(current_dir, '../Release/engine_csdk.pyd'),
                os.path.join(current_dir, '../../cxx_module/Release/fasttext_inf.dll')
            ],
            'dst': [
                os.path.join(current_dir, 'engine_csdk.pyd'),
                os.path.join(current_dir, 'fasttext_inf.dll')
            ]
        }

    elif platform.system().lower() == 'linux':

        file_map = {
            'src': [
                os.path.join(current_dir, '../engine_csdk.so'),
                os.path.join(current_dir, '../../{}/fasttext_inf.so'.format( 'cxx_module_cross_compile' if version_config.BUILD_CROSSCOMPILING else 'cxx_module'))
            ],
            'dst': [
                os.path.join(current_dir, 'engine_csdk.so'),
                os.path.join(current_dir, 'fasttext_inf.so')
            ],
        }
    src_lst = file_map['src']
    dst_lst = file_map['dst']
    for i in range(len(src_lst)):
        src = src_lst[i]
        dst = dst_lst[i]
        if os.path.exists(src):
            shutil.copyfile(src, dst)

def get_package_version():

    package_version = str(version_config.NN_SDK_VERSION_MAJOR) + '.' + \
                      str(version_config.NN_SDK_VERSION_MINOR) + '.' + \
                      str(version_config.NN_SDK_VERSION_PATCH)
    return package_version

def get_is_cross_compile():
    return version_config.BUILD_CROSSCOMPILING