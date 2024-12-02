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

import numpy as np

# from tqdm import tqdm

import cv2

# PyTorch libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable

import torch.optim as optim

# import torch.optim.lr_scheduler as lr_scheduler

from srcs.utils import device

from pdb import set_trace as bp  # This is only for debugging

##########################################################################################
# The class could be extended to different attacks. Names captured by this global list.

ATTACKS = ["BIM"]


class VisualAdversarialAttack(nn.Module):
    """Class for visual adversarial attacks on images."""

    def __init__(
        self,
        config,
        target_label="gibbon",
    ):
        super(VisualAdversarialAttack, self).__init__()

        self.config = config

        self.repo_dir = config["paths"]["root_dir"]

        # FGSM/BIM parameters
        self.delta = (
            1 / 255.0
        )  # We are going to use the image normalised to [0,1] (standardisation)
        self.eps = config["bim"]["eps"]
        self.img_size = config["bim"]["img_size"]

        # Load ImageNet classes only once and re-use across the class.
        self.imagenet_classes = self.load_vocabulary()

        if target_label not in self.imagenet_classes:
            raise Exception(
                "The provided target label is not part of the vocabulary of the pre-trained model. Re-run with a label selected from resources/imagenet.class."
            )

        self.target_label = target_label

        self.load_model()

        self.configure_optimizer()

        # Prepare image transformations
        self.set_img_transform()

    def load_vocabulary(self):
        """Load the list of classes from ImageNet."""
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
        """Load a pre-trained model to predict the class of an input image.

        We use a PyTorch model, ResNet-18, pre-trained on ImageNet.
        Ref: https://pytorch.org/hub/pytorch_vision_resnet/
        """
        model = torch.hub.load(
            "pytorch/vision:v0.10.0", "resnet18", weights="IMAGENET1K_V1"
        )

        # for param in model.parameters():
        #     param.requires_grad = False

        model.eval()

        self.model = model

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

    def configure_optimizer(self):
        """ """
        params = self.config["params"]

        if params["optimizer"] == "Adam":
            self.optimizer = getattr(optim, params["optimizer"])(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=params["init_lr"],
                weight_decay=params["weight_decay"],
            )
            self.optimizer_name = "Adam"

    def add_aversarial_perturbation(self, adv_example, img_grad):
        """ """
        # Compute adversarial noise as a small perturbation with respect to the image gradients
        adv_noise = self.delta * img_grad.sign()

        # Compute adversarial example
        adv_example -= adv_noise

        # Clip image at the top and bottom
        # adv_example_clipped = torch.clamp(adv_example, min=0, max=1)

        return adv_example

    def basic_iterative_method_attack(self, image):
        """Attack the model with the basic iterative method.

        This is a targeted attack: the user provides the label to make
        the model misclassifies the input image.

        Reference:
        ADVERSARIAL EXAMPLES IN THE PHYSICAL WORLD
        Kurakin, ICLR 2017
        https://arxiv.org/pdf/1607.02533
        Sec.2.2

        TODO: Unnoticeable part still to accomplish based on this implemented algorithm.

        Arguments:
            - image
        """
        # Number of iterations for the BIM attack (see paper)
        n_iters = int(min(self.eps + 4, self.eps * 1.25))

        loss = nn.CrossEntropyLoss()

        target_class_var = Variable(
            torch.from_numpy(np.zeros(len(self.imagenet_classes)))
        )
        idx_target = self.imagenet_classes.index(self.target_label)
        target_class_var[idx_target] = 1
        target_class_var = target_class_var.unsqueeze(0).to(device)

        adv_img = image.clone()
        adv_img = Variable(adv_img, requires_grad=True)
        adv_img.to(device)

        print("Number of iterations for the attack: {:d}".format(n_iters))

        for n in range(n_iters):
            print("Iteration #{:d}/{:d}".format(n + 1, n_iters))

            tmp_img = adv_img.clone()
            tmp_img.grad = None

            outputs, _, _ = self.predict_image_class(adv_img.unsqueeze(0))

            target_loss = loss(outputs.to(device), target_class_var)
            target_loss.backward()

            adv_img = self.add_aversarial_perturbation(tmp_img, adv_img.grad)

            adv_img = Variable(adv_img, requires_grad=True)

        return adv_img.clone()

    def predict_image_class(self, image):
        """Predict the most likely class of the input image.

        The model predicts the logits of the input image (1000 classes for imagenet).
        Logits are converted into probabilities using softmax.
        We get the class (string) corresponding to the class with the top confidence.
        """
        if len(image.shape) == 3:
            outputs = self.model(image.unsqueeze(0))
        else:
            outputs = self.model(image)

        # convert logits into probabilities
        out_probs = F.softmax(outputs)

        max_prob = torch.max(out_probs)
        max_prob_ind = torch.argmax(out_probs)

        top_class = self.imagenet_classes[max_prob_ind]

        print("Predicted class for input image: " + top_class)
        print("Confidence of the class: {:.2f}".format(max_prob))

        return outputs, top_class, max_prob

    def save_image(self, adv_img, out_filename):
        """Save the attacked image to file."""
        print("Saving image to file ...")
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        for c in range(3):
            adv_img[c, :, :] *= std[c]
            adv_img[c, :, :] += mean[c]

            adv_img = torch.clamp(adv_img, min=0, max=1)

        img_save = (adv_img * 255).round()

        img_save = img_save.cpu().detach().numpy()
        img_save = np.uint8(img_save).transpose(1, 2, 0)
        img_save = img_save[..., ::-1]

        cv2.imwrite(out_filename, img_save)

        print("Saved!")

    def run_attack(self, image_fn, attack="BIM"):
        """Main function of the class to run the adversarial attack."""

        assert (
            attack in ATTACKS
        )  # Check that the passed attack is valid, if any

        # Load and normalise the image with mean and std from ImageNet
        img_pil, img_norm = self.load_image(image_fn)

        # Evaluate the original image and its prediction
        self.predict_image_class(img_norm)

        # Run the attack
        if attack == "BIM":
            adv_img = self.basic_iterative_method_attack(img_norm)

            _, top_class, max_prob = self.predict_image_class(adv_img)
            self.save_image(
                adv_img,
                os.path.join(self.repo_dir, "resources", "adv_example.jpg"),
            )
