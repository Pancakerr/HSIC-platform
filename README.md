# Hyperspectral-Classification-Platform
This is a deep learning platform for hyperspectral classification by pytorch, integrated several classic HSIC models including **[CNN1D](https://www.hindawi.com/journals/js/2015/258619/),[A2S2KResNet](https://ieeexplore.ieee.org/document/9306920),[HybridSN](https://ieeexplore.ieee.org/document/8736016),[M3DCNN](https://ieeexplore.ieee.org/document/8297014/),[SSUN](https://ieeexplore.ieee.org/document/8356713),[SSFCN](https://ieeexplore.ieee.org/document/8737729) [SpectralFormer](https://ieeexplore.ieee.org/document/9627165),VIT,GhostNet,[FullyContNet](https://ieeexplore.ieee.org/document/9347487),MobileNet,UNet**.
This is also an official implement of ["SSRNet: A Lightweight Successive Spatial Rectified Network With Noncentral Positional Sampling Strategy for Hyperspectral Images Classification"](https://ieeexplore.ieee.org/document/10203001) and ["A Multi-scale Convolutional Neural Network Based on Multilevel Wavelet Decomposition for Hyperspectral Image Classification"](https://link.springer.com/chapter/10.1007/978-3-031-18913-5_38)

## Install

Firstly 
    conda create -n myenv python==3.7
    conda activate myenv
    conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
    conda install --file requirements.txt

## Usage
We used *visdom* for visualizing experiments, so you should run *visdom.server* firstly.

    python -m visdom.server

You also can just run the experiment without *visdom* by command line parameter **--VISDOM 0**

    python main.py --MODEL SSRNet --VISDOM 0

### Examples

You can easily run different hyperspectral models by command lines. For example:

    python main.py --MODEL SSRNet       --DATASET  IP  --RUNS 1 --DEVICE 0 --VISDOM 1
                           A2S2KResNet             UP
                           CNN1D                   UH
                           FullyContNet
                           GhostNet
                           HybridSN
                           M3DCNN
                           MLWBDN
                           MobileNet
                           SSFCN
                           SSUN
                           UNet
                           SFormer_px
                           SFormer_pt
                           VIT

We also provide many basic parameters for training and test such as *batchsize, epoch, learning rate and so on*. You can see details in **param file**.

## Acknowledge

[DotWang/FullyContNet](https://github.com/DotWang/FullyContNet)
[eecn/Hyperspectral-Classification](https://github.com/eecn/Hyperspectral-Classification)
[danfenghong/IEEE_TGRS_SpectralFormer](https://github.com/danfenghong/IEEE_TGRS_SpectralFormer)
[YonghaoXu/SSFCN](https://github.com/YonghaoXu/SSFCN)
[suvojit-0x55aa/A2S2K-ResNet](https://github.com/suvojit-0x55aa/A2S2K-ResNet)
