#!/bin/bash
#
# Script to automatically format python files according to PEP 8 format style.
# The script requires the pip package 'black' installed.
# 	```pip3 install black```
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
# PARAMETERS
LINE_LEN=79
#
##############################################################################
#
python -m black --line-length=$LINE_LEN srcs/main.py
python -m black --line-length=$LINE_LEN srcs/utils.py
python -m black --line-length=$LINE_LEN srcs/advattack.py
