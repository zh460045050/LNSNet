# LNSNet
## Overview

Official implementation of [Learning the Superpixel in a Non-iterative and Lifelong Manner][arxiv] (CVPR'21)

<center>
<img src="pics/strategy.png" width="80%" />
  
</center>


The proposed LNSNet views superpixel segmentation process of each image as **an independent pixel-level clustering task** and use **lifelong learning strategy** to train the superpixel segmentation network for a a series of images.

[arxiv]: https://arxiv.org/abs/2103.10681

The structure of proposed LNS-Net shown in Fig. 3 contains three parts: 

1) **Feature Embedder Module (FEM)** that embeds the original feature into a cluster-friendly space; 
2) **Non-iteratively Clustering Module (NCM)** that assigns the label for pixels with the help of a seed estimation module, which automatically estimates
the indexes of seed nodes;
3) **Gradient Rescaling Module (GRM)** that adaptively rescales the gradient for each weight parameter based on the channel and spatial context to avoid catastrophic forgetting
for the sequential learning.

<center>
<img src="pics/structures.png" width="80%" />
  
</center>


## Getting Started

Here we only release the model trained on BSDS dataset and corresponding code to utilizes it for superpixel segmentation. The whole training code will be coming soon.

To uese the given model for generate superpixel:

`git clone https://github.com/zh460045050/LNSNet`

`cd LNSNet`

`sh runDemo.sh`

or

`python demo.py --n_spix $num_superpixel --img_path $input_img_path --check_path lnsnet_BSDS_checkpoint.pth`

The performance and complexity of methods for generating 100 superpixel on BSDS test dataset with image size 481*321:

<center>
<img src="pics/results.png" width="50%" />

</center>


## Citation

If you find our work useful in your research, please cite:

@InProceedings\{Lei_2021_CVPR,
title = \{Learning the Superpixel in a Non-iterative and Lifelong Manner\},
author = \{Zhu, Lei and She, Qi and Zhang, Bin and Lu, Yanye and Lu, Zhilin and Li, Duo and Hu, Jie\},
booktitle = \{IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)\},
month = \{June\},
year = \{2021\}
\}
