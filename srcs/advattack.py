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

# Image processing, vision libraries
from PIL import Image

# PyTorch libraries
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from srcs.utils import (
    device
)

from pdb import set_trace as bp  # This is only for debugging

##########################################################################################
# The class could be extended to different attacks. Names captured by this global list.

ATTACKS=["BIM"] 

class VisualAdversarialAttack(nn.Module):
    """ Class for visual adversarial attacks on images.
    """
    def __init__(
        self,
        config,
        target_label="gibbon",
    ):
        self.config = config

        self.repo_dir = config["paths"]["root_dir"]

        # FGSM/BIM parameters
        self.delta = 1/255. # We are going to use the image normalised to [0,1] (standardisation)
        self.eps = config["bim"]["eps"]
        self.img_size = config["bim"]["img_size"]
        
        # Load ImageNet classes only once and re-use across the class.
        self.imagenet_classes = self.load_vocabulary()

        if target_label not in self.imagenet_classes:
            raise Exception("The provided target label is not part of the vocabulary of the pre-trained model. Re-run with a label selected from resources/imagenet.class.")
        
        self.target_label = target_label
    

    def load_vocabulary(self):
        """ Load the list of classes from ImageNet.
        """
        fp = open(
            os.path.join(
                self.repo_dir,
                "resources",
                "imagenet.classes",
            ),
            "r",
        )
        names = fp.read().split("\n")[:-1]
        fp.close()

        return names

    def load_model(self):
        """ Load a pre-trained model to predict the class of an input image.

        We use a PyTorch model, ResNet-18, pre-trained on ImageNet.
        Ref: https://pytorch.org/hub/pytorch_vision_resnet/
        """
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

        for param in model.parameters():
            param.requires_grad = False

        self.model = model.to(device)
        self.model.eval()
    
    def set_img_transform(self):
        """Function to standardise the image with ImageNet values from training set.

        This is a function that resizes an image to the required resolution by the chosen model
        and normalises the values between 0 and 1 with the pre-computed values from ImageNet dataset.
        """
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )  # imagenet values

        self.full_im_transform = transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                normalize,
            ]
        )

    def load_image(self, image_fn):
        """Load the image from the filename into PIL format.

        The image is also resized and normalised with ImageNet weights.
        Both the PIL and transformed image are returned by the function.
        """
        img_pil = Image.open(image_fn).convert("RGB")

        full_im = self.full_im_transform(img_pil)

        return img_pil, full_im

    def add_aversarial_perturbation(self, image):
        """
        """


    def basic_iterative_method_attack(self, image):
        """ Attack the model with the basic iterative method.

        This is a targeted attack: the user provides the label to make 
        the model misclassifies the input image.

        Reference:
        ADVERSARIAL EXAMPLES IN THE PHYSICAL WORLD
        Kurakin, ICLR 2017
        https://arxiv.org/pdf/1607.02533
        Sec.2.2

        Arguments:
            - image
        """
        # Number of iterations for the BIM attack (see paper)
        n_iters = int(min(self.eps + 4, self.eps * 1.25))

        loss = nn.CrossEntropyLoss()

        for n in range(n_iters):
            self.add_aversarial_perturbation(image)



    def predict_image_class(self, image):
        """ Predict the most likely class of the input image.

        The model predicts the logits of the input image (1000 classes for imagenet). 
        Logits are converted into probabilities using softmax. 
        We get the class (string) corresponding to the class with the top confidence.
        """
        outputs = self.model(image)

        # convert logits into probabilities
        out_probs = torch.softmax(outputs)

        max_prob_ind = torch.max(out_probs)

        top_class = self.imagenet_classes[max_prob_ind[0][1]]

        print("Predicted class for input image: " + top_class)
        print("Confidence of the class: {:.2f}".format(max_prob_ind[0][0]))


    def save_image(self):
        """ Save the attacked image to file.
        """
        
    
    def run_attack(self, image_fn, attack="BIM"):
        """ Main function of the class to run the adversarial attack.
        """

        assert(attack in ATTACKS) # Check that the passed attack is valid, if any

        self.load_model()

        # Load and normalise the image with mean and std from ImageNet
        img_pil, img_norm = self.load_image(image_fn)

        # Evaluate the original image and its prediction
        self.predict_image_class(img_norm)

        # Run the attack
        if attack == "BIM":
            self.basic_iterative_method_attack(img_norm)

        print()