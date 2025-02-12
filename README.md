# Polyp-Segmentation

## Objective
This repository focuses on achieving high-precision and reliable segmentation of polyps in colonoscopy images, a critical step in early colorectal cancer detection and treatment planning. The project explores state-of-the-art deep learning models, including **DeepLabV3+**, **EMCAD**, and **PraNet**, along with an innovative **post-processing methodology**. The post-processing technique employs **adaptive morphological operations** and a **region-aware refinement network**, which consistently enhances segmentation accuracy by 3-4% across multiple architectures.

## Approach
Our approach integrates **DeepLabV3+**, **PraNet**, and **EMCAD**, each optimized for colon polyp segmentation:
- **DeepLabV3+** is utilized for its superior multi-class segmentation performance, leveraging atrous convolutions and Atrous Spatial Pyramid Pooling (ASPP) for multi-scale feature extraction.
- **PraNet** incorporates a **Reverse Attention Module** to enhance the focus on polyp regions while minimizing background noise.
- **EMCAD** (Efficient Multi-scale Convolutional Attention Decoder) introduces a computationally efficient approach to segmentation by leveraging large-kernel grouped attention and multi-scale convolutional attention mechanisms.

Additionally, a **post-processing method** is implemented to refine the segmentation results, ensuring higher consistency and reducing noise in predicted masks. This is achieved through **connected component analysis** and **region-wise color consistency enforcement**, leading to improved Dice scores across all tested models.

## Dataset
The dataset used in this project is the **[BKAI-IGH NeoPolyp](https://www.kaggle.com/competitions/bkai-igh-neopolyp)** dataset from Kaggle, which contains images annotated with detailed polyp segmentation masks. It includes 1,200 colonoscopy images, providing a diverse and challenging dataset for model training and evaluation.

## Training Model
The models are trained using **DeepLabV3+ (RegNetx320, ResNet50)**, **EMCAD (PVT2-b2)**, and **PraNet** on **Kaggle**. Training is performed with extensive **data augmentation**, including **geometric transformations** (rotations, flips, shifts) and **image quality adjustments** (random gamma, Gaussian blur, coarse dropout) to enhance generalization.

The training process involves:
- **Loss Function**: A combination of **Cross-Entropy Loss** and **Dice Loss** to balance pixel-wise classification and region-level similarity.
- **Optimization**: Adam optimizer with weight decay (1e-4) and an **ExponentialLR scheduler** (gamma=0.6 every 5 epochs) for stable convergence.
- **Batch Size & Learning Rate**: Training is conducted with a batch size of **16** and an initial learning rate of **1e-4**, progressively reduced over **50 epochs**.

All logs and results, including training and validation metrics, are tracked using **[Wandb.ai](https://wandb.ai/hung123ka5/polypsegmentation?nw=nwuserhung123ka5)**.

You can follow this link to get the source code:


[Polyp_Segmentation](https://www.kaggle.com/code/hunghoang31/unet-polyp-segmentaion#Training)

## DICE Score 
The models achieve state-of-the-art **segmentation score**, consistently improving with **post-processing techniques**. The **Dice Score** comparison for different models is presented below:

| Model                  | Dice Score (Without Processing) | Dice Score (With Processing) |
|------------------------|--------------------------------|-----------------------------|
| **PraNet**            | 80.02%                         | 82.94%                      |
| **DeepLabV3+ (RegNetx320)** | **82.56%**                  | **86.33%**                      |
| **DeepLabV3+ (ResNet50)**  | 81.12%                  | 85.62%                      |
| **EMCAD (PVT2-b2)**   | 80.93%                         | 83.76%                      |
| **UNet**              | 75.95%                         | -                           |
| **UNet++**            | 77.54%                         | -                           |

These results highlight the effectiveness of **DeepLabV3+ with RegNetx320**, achieving the highest **Dice Score of 86.33%** after post-processing, demonstrating superior multi-class segmentation capability.

## Results & Achievements
- Ranked **Top 10** in the **[BKAI-IGH NeoPolyp](https://www.kaggle.com/competitions/bkai-igh-neopolyp)** competition.
- Achieved a **Dice Score of 86.33%** with **DeepLabV3+ (RegNetx320)** after post-processing.
- Demonstrated a **3-4% improvement** in segmentation accuracy across multiple architectures using our **post-processing technique**.


## Set up Enviroment
You need to install libraries in **requirement.txt** and get the path data files. Or you can access the [Kaggle_link](https://www.kaggle.com/code/hunghoang31/unet-polyp-segmentaion#Training) where I have set up all the enviroment. 

## Project Structure
```
├── Training
│  ├── ...                 
│  ├── test   
│  ├── train
│  ├── train_gt   
```