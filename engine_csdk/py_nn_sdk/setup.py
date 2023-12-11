# -*- coding:utf-8 -*-
import setuptools
import sys
sys.path.append('.')
import platform
import os
import shutil

from setuptools import Extension
from setuptools.command.build_ext import build_ext as _build_ext
from setuptools.command.build_py import build_py as _build_py

from setup_wrapper import get_desc,copy_dep_lib,get_package_version,get_is_cross_compile

package_name = 'nn-sdk'


current_dir = os.path.dirname(os.path.abspath(__file__))
title = 'nn-sdk tensorflow(v1 ,v2),onnx,tensorrt,fasttext model infer engine'

project_description_str = get_desc()
project_description_str = title + '\n' + project_description_str


platforms_name = sys.platform + '_' + platform.machine()
class PrecompiledExtesion(Extension):
    def __init__(self, name):
        super().__init__(name, sources=[])

class build_ext(_build_ext):
    def build_extension(self, ext):
        if not isinstance(ext, PrecompiledExtesion):
            return super().build_extension(ext)


exclude = ['setup','setup_wrapper']

class build_py(_build_py):
    def find_package_modules(self, package, package_dir):
        modules = super().find_package_modules(package, package_dir)
        return [(pkg, mod, file, )  for (pkg, mod, file,) in modules if mod not in exclude]


if __name__ == '__main__':
    copy_dep_lib()

    setuptools.setup(
        platforms=platforms_name,
        name=package_name,
        version=get_package_version(),
        author="ssbuild",
        author_email="9727464@qq.com",
        description=title,
        long_description_content_type='text/markdown',
        long_description=project_description_str,
        url="https://github.com/ssbuild",
        #packages=setuptools.find_packages(exclude=['setup.py']),
        packages=['nn_sdk'],   # 指定需要安装的模块
        include_package_data=True,
        package_dir={'nn_sdk': './py_nn_sdk'},
        package_data={'': ['*.pyd','*.dll','*.so','*.h','*.c','*.java']},
        ext_modules=[PrecompiledExtesion('nn_sdk')],
        cmdclass={'build_ext': build_ext,'build_py': build_py},
        # data_files =[('',["nn_sdk/easy_tokenizer.so","nn_sdk/engine_csdk.so"])],
        # packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
        # py_modules=["six"], # 剔除不属于包的单文件Python模块
        # install_requires=['peppercorn'], # 指定项目最低限度需要运行的依赖项
        python_requires='>=3, <4', # python的依赖关系
        #install_requires=['tf2pb ~= 0.2.0'],
        classifiers=[
            'Development Status :: 5 - Production/Stable',
            'Intended Audience :: Developers',
            'Intended Audience :: Education',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: Apache Software License',
            'Programming Language :: C++',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'Programming Language :: Python :: 3.10',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Mathematics',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'Topic :: Software Development',
            'Topic :: Software Development :: Libraries',
            'Topic :: Software Development :: Libraries :: Python Modules',
        ],
        license='Apache 2.0',
        keywords=[package_name,"nn_sdk",'tensorflow','tf','onnx','tensorrt','trt','onnxruntime','inference','pb'],
    )

    if platform.system().lower() == 'linux':
        src = ''
        dst = ''

        is_cross_compile = get_is_cross_compile()
        for path,dirs,filenames in os.walk(os.path.join(current_dir,'../dist')):
            for filename in filenames:
                if filename.find('.whl'):
                    src = os.path.join(path,filename)
                    if not is_cross_compile:
                        dst = os.path.join(path,filename.replace('linux','manylinux2014'))
                    else:
                        dst = os.path.join(path, filename.replace('linux_x86_64', 'manylinux2014_aarch64'))
                    break
        if src and dst:
            shutil.move(src,dst)
