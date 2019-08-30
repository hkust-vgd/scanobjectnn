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
Documentation on dataset coming soon!

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
