# Revisiting Point Cloud Classification: A New Benchmark Dataset and Classification Model on Real-World Data
**[Revisiting Point Cloud Classification: A New Benchmark Dataset and Classification Model on Real-World Data](https://hkust-vgd.github.io/scanobjectnn/)** 

Mikaela Angelina Uy, Quang-Hieu Pham, Binh-Son Hua, Duc Thanh Nguyen and Sai-Kit Yeung

ICCV 2019 Oral Presentation

![pic-network](objects_teaser.png)

## Introduction
This work revisits the problem of point cloud classification but on real world scans as opposed to synthetic models such as ModelNet40 that were studied in other recent works. We introduce **ScanObjectNN**, a new benchmark dataset containing ~15,000 object that are categorized into 15 categories with 2902 unique object instances. The raw objects are represented by a list of points with global and local coordinates, normals, colors attributes and semantic labels. We also provide part annotations, which to the best of our knowledge is the first on real-world data. From our comprehensive benchmark, we show that our dataset poses great challenges to existing point cloud classification techniques as objects from real-world scans are often cluttered with background and/or are partial due to occlusions. Our project page can be found [here](https://hkust-vgd.github.io/scanobjectnn/), and the arXiv version of our paper can be found [here](https://arxiv.org/abs/1908.04616).
```
@inproceedings{uy-scanobjectnn-iccv19,
      title = {Revisiting Point Cloud Classification: A New Benchmark Dataset and Classification Model on Real-World Data},
      author = {Mikaela Angelina Uy and Quang-Hieu Pham and Binh-Son Hua and Duc Thanh Nguyen and Sai-Kit Yeung},
      booktitle = {International Conference on Computer Vision (ICCV)},
      year = {2019}
  }
```

## ScanObjectNN Dataset
Our ScanObjectNN Dataset can be downloaded from here: [HKUST OneDrive](https://gohkust-my.sharepoint.com/:f:/g/personal/saikit_ust_hk/EqRFLP5XEihCt_PFIHyPNO8BsKb7r8S5V5ELaCqk7UdDTQ?e=FX2OPF) or here: [SUTD server](http://103.24.77.34:8080/scanobjectnn/). We provide different variants of our scan dataset namely: OBJ_BG, PB_T25, PB_T25_R, PB_T50_R and PB_T50_RS as described in our paper. We released both the processed .h5 files and the raw .bin objects as described below.

### h5 files
* Download the **h5_files.zipped** to obtained all the h5 files. Main split was used for the experiments in the [main paper](https://arxiv.org/pdf/1908.04616.pdf), while splits 1-4 are the additional training/test splits reported in our [supplementary material](https://hkust-vgd.github.io/scanobjectnn/assets/iccv19_supp.pdf).
* The pre-processed h5 files can be directly used by deep learning frameworks, containing fields: 
   1. **data**: Nx3 point cloud
   2. **label**: class label
   3. **mask**: indicator whether each point is part of the object instance or the background.
* Each object contained 2048 points, where each point is represented by its x, y, z coordinates. 
* We first ensured that a data sample had at least 2048 object instance points (excluding the background) before 2048 points were randomly selected (including the background points) and included into the h5 file. For the \*_nobg h5 files, background points were first filtered out before the random selection 
* Naming convention: Prefixes are *training_** and *test_** for training set and test set, respectively.
    * **OBJ_BG** / **OBJ_ONLY**: *\*objectdataset.h5*
    * **PB_T25**: *\*objectdataset_augmented25_norot.h5*
    * **PB_T25_R**: *\*objectdataset_augmented25rot.h5*
    * **PB_T50_R**: *\*objectdataset_augmentedrot.h5*
    * **PB_T50_RS**: *\*objectdataset_augmentedrot_scale75.h5*

### Raw files
[TODO]

## Code
Documentation on code structure coming soon!

## Pre-trained Models
Pre-trained models can be downloaded [here](https://drive.google.com/open?id=1somhNuzwEnJB7J6ESGuW_6ZryW8emW6u).

## References
Our released code heavily based on each methods original repositories as cited below:
* <a href="https://github.com/charlesq34/pointnet" target="_blank">PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation</a> by Qi et al. (CVPR 2017).
* <a href="https://github.com/charlesq34/pointnet2" target="_black">PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space</a> by Qi et al. (NIPS 2017).
* <a href="https://github.com/WangYueFt/dgcnn" target="_black"> Dynamic Graph CNN for Learning on Point Clouds</a> by Wang et al. (TOG 2019).
* <a href="https://github.com/yangyanli/PointCNN" target="_black">PointCNN: Convolution On X-Transformed Points</a> by Li et al. (NIPS 2018).
* <a href="https://github.com/xyf513/SpiderCNN" target="_black">SpiderCNN: Deep Learning on Point Sets with Parameterized Convolutional Filters</a> by Xu et al. (ECCV 2018).
* <a href="https://github.com/sitzikbs/3DmFV-Net" target="_black">3DmFV : Three-Dimensional Point Cloud Classification in Real-Time using Convolutional Neural Networks</a> by Ben-Shabat et al. (RA-L 2018).  
