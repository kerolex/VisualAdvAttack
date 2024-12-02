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

import os
import random

import numpy as np
import torch

# ----------------------------------------------------------------------------
# CONSTANTS
#
device = "cuda" if torch.cuda.is_available() else "cpu"

n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0


# ----------------------------------------------------------------------------
def set_seed(seed_val) -> None:
    """ """
    np.random.seed(seed_val)
    random.seed(seed_val)

    torch.manual_seed(seed_val)

    if device == "cuda":
        torch.cuda.manual_seed(seed_val)

        # When running on the CuDNN backend, two further options must be set
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed_val)
    print(f"Random seed set as {seed_val}")
