# Polyp-Segmentation

## Objective
This repository aims to provide an efficient and accurate solution for the automatic segmentation of polyps in medical images. The goal is to improve the detection and delineation of polyps, which is crucial for early diagnosis and treatment planning in colorectal cancer. This project leverages state-of-the-art deep learning techniques, including advanced segmentation models like UNET, UNET++, PraNet,...  to achieve high performance on benchmark datasets.

## Approach
I approach polyp segmentation by leveraging UNet and PraNet. UNet’s encoder-decoder structure with skip connections effectively captures both global context and fine details, making it ideal for precise medical image segmentation. PraNet, designed specifically for polyp segmentation, focuses on highlighting polyp regions while minimizing background noise, resulting in improved accuracy. To further enhance model performance, I apply extensive data augmentation techniques, including rotations, flips, shifts, and varying image sizes, which help improve generalization and robustness. 

## Dataset 

This project uses the dataset for colonoscopy polyp segmentation and neoplasm characterization which belong to the competition [BKAI-IGH NeoPolyp](https://www.kaggle.com/competitions/bkai-igh-neopolyp) on Kaggle. 

## Trainning Model

We have trained our model with two backbones: **UNET** and **UNET++**

We have set up all requirement in enviroment on **Kaggle**. All the log and results during the training and testing process have been saved on [Wandb.ai](https://wandb.ai/hung123ka5/polypsegmentation?nw=nwuserhung123ka5)

You can follow this link to get the source code:

[UNET_Polyp_Segmentation](https://www.kaggle.com/code/hunghoang31/unet-polyp-segmentaion#Training)

## Accuracy 
The model achives approximate **80\%** and reach the high postition of the competition.

## Set up Enviroment
You need to install libraries in **requirement.txt** and get the path data files. Or you can access the [Kaggle_link](https://www.kaggle.com/code/hunghoang31/unet-polyp-segmentaion#Training) where I have set up all the enviroment. 

## Project Structure
```
├── Training
│  ├── ...                 
├── Data
│  ├── test   
│  ├── train
│  ├── train_gt   
```