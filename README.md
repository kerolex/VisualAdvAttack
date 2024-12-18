# VisualAdvAttack

Library for Visual Adversarial Attacks. 

This is a simple software that performs the Basic Iterative Method as visual adversarial attack on 
a image-based classifier (ResNet-18) pre-trained on ImageNet. This is a targeted attack to add a noise to the input image in an iterative way until predicting the target label, with a predefined number of iterations based on the formula provided in the paper (see reference at the end of the readme).


## Requirements

* Python
* PyTorch
* OpenCV (for saving images)
* Numpy
* PIL

You can follow the following commands to install the conda environment. 
Please ensure to update according to your version.

```
conda create -n viadvattack
conda activate viadvattack

conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=10.2 pyg -c pytorch -c pyg

python -m pip install -U pip
python -m pip install -U black Pillow tqdm


# Vision libraries
python -m pip install -U opencv-python
```

## Instructions/demo

To run the software, you can do it in 2 ways: use the provided bash script or running from command line.
Make sure to install the conda environment first or have a conda environment with the corresponding packages already installed.


Mode 1:
1. Open terminal at the repository directory
2. Type: ``source scripts/run_bim.sh``

Mode 2:
1. Open terminal at the repository directory
2. Activate your conda environment: ``conda activate viadvattack``
3. Type:
```
python srcs/main.py --config  ./configs/bim_attack.json --image resources/dog.jpg --target_class baboon
```
4. Deactivate your conda environment: ``conda deactivate``

The software has been tested with the target labels (single word): baboon and gibbon. 
The demo use the already provided image ``resources/dog.jpg``, available in PyTorch: https://pytorch.org/hub/pytorch_vision_resnet/ 

Predicted class and confidence are outputted at the screen.

The output of the available script should correspond to the following:
```
Predicted class for input image: Samoyed
Confidence of the class: 0.84
Number of iterations for the attack: 6
Iteration #1/6
Predicted class for input image: Samoyed
Confidence of the class: 0.84
Iteration #2/6
Predicted class for input image: Samoyed
Confidence of the class: 0.47
Iteration #3/6
Predicted class for input image: Samoyed
Confidence of the class: 0.21
Iteration #4/6
Predicted class for input image: baboon
Confidence of the class: 0.20
Iteration #5/6
Predicted class for input image: baboon
Confidence of the class: 0.83
Iteration #6/6
Predicted class for input image: baboon
Confidence of the class: 0.99
Predicted class for input image: baboon
Confidence of the class: 1.00
```

The outputted image is available at ``resources/adv_example_bak.jpg``

Input image
![Input image (dog)](/resources/dog.jpg)

Output image
![Adversarial example](/resources/adv_example_bak.jpg)

## Known issues
* The output image is not at the original resolution
* The image has noticeable changes that could be due to the saving (to verify) or the algorithm itself
* The input does not handle multi-words (e.g., giant panda). 

## TODO
* Modifying the implemented algorithm to make the adversarial image unnoticeable (e.g., adding as an auxiliary loss the L2-norm between the input image and the output adversarial image)
* Fixing the output resolution
* Fixig the handling of multi-words
* Save predicted class and confidence (txt file, or on the image, or else)


## Reference
* Kurakin, et al., ADVERSARIAL EXAMPLES IN THE PHYSICAL WORLD, ICLR 2017, https://arxiv.org/pdf/1607.02533 (Sec.2.2)