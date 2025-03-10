#!/bin/bash
#
##############################################################################
# Authors:
# - Alessio Xompero, alessio.xompero@gmail.com
#
#  Created Date: 2024/12/02
# Modified Date: 2024/12/02
#
# MIT License

# Copyright (c) 2024 Alessio Xompero

# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
# -----------------------------------------------------------------------------
#
##############################################################################
#
CUDA_DEVICE=0
##############################################################################
# PATHS
# Directory of the repository in the current machine/server
ROOT_DIR=$PWD

CONFIG_FILE=$ROOT_DIR/configs/bim_attack.json

IMG=resources/dog.jpg
TARGET=baboon
##############################################################################
#
# Activate the conda environment
conda activate viadvattack

# Testing
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python srcs/main.py                \
    --config                $CONFIG_FILE        \
    --image                 $IMG                \
    --target_class          $TARGET

conda deactivate
