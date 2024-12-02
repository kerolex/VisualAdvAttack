#!/usr/bin/env python
#
# 
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

import argparse
import inspect
import json
import os
import sys

# Package modules
current_path = os.path.abspath(inspect.getfile(inspect.currentframe()))

dirlevel1 = os.path.dirname(current_path)
dirlevel0 = os.path.dirname(dirlevel1)

print(dirlevel0)

sys.path.insert(0, dirlevel0)

from srcs.advattack import VisualAdversarialAttack

from srcs.utils import (
    device,
    set_seed
)

#############################################################################

def GetParser(desc=""):
    """ """
    parser = argparse.ArgumentParser(
        description=desc,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--seed", 
        type=int
        )
   

    parser.add_argument(
        "--config", 
        required=True, 
        help="Please provide a config.json file"
    )

    parser.add_argument(
        "--image", 
        type=str, 
        required=True,
        help="Provide input image to misclassify"
    )

    parser.add_argument(
        "--target_class", 
        type=str, 
        required=True,
        help="Please provide a target class from the list of classes in ImageNet (see file resources/imagenet.names)"
    )

    return parser


#############################################################################
#
if __name__ == "__main__":
    print("Initialising:")
    print("Python {}.{}".format(sys.version_info[0], sys.version_info[1]))
    # print("PyTorch {}".format(torch.__version__))
    print("Using {}".format(device))

    # Arguments
    parser = GetParser()
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    set_seed(config["params"]["seed"])

    bim_attack = VisualAdversarialAttack(config, args.target_class)
    bim_attack.run_attack(args.image)
