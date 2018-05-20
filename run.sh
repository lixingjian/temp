#!/bin/sh

#!/bin/bash
envdir=/mnt/home/lixingjian/mqy/conda/miniconda2/envs/tf1.4_py2.7
libdir=/mnt/home/lixingjian/tool

export PATH=$envdir/bin:$PATH
export PYTHONPATH=$envdir//lib/python2.7/site-packages/tensorflow/python:$PYTHONPATH
export LD_LIBRARY_PATH=$envdir/lib:$libdir/cuda-8.0/lib64:$libdir/cudnn_v6.0/cuda/lib64:$LD_LIBRARY_PATH
python -u $*
