#!/usr/bin python
#-*- coding:utf-8 -*-

# Copyright (C) 2018, Nudt, JingshengTang, All Rights Reserved
# Author: Jingsheng Tang
# Email: mrtang@nudt.edu.cn

from setuptools import setup, find_packages

setup(
    name = "bplib",
    version = "1.0",
    python_requires='>=3.7',
    author = 'mrtang',
    author_email = 'mrtang_cs@163.com',
    description = 'bplib',
    packages = find_packages(),
    install_requires = ['paho-mqtt'], #依赖包
)
