# CSF
### Enhancing Sandstorm Images via Color-Guided Spatial-Frequency Fusion Network

The official PyTorch Implementation of CSF for Image Enhancement


#### Zhengwei Guo</sup>, Bo Wang </sup>



## Latest
- `09/29/2024`: Release code and datesets.
- `09/27/2024`: Repository is created. 


## Method
<br>
<details>
  <summary>
  <font size="+1">Abstract</font>
  </summary>
Sandstorm images are often plagued by color distortions, reduced contrast, and blurred details, posing challenges for advanced vision tasks. To address these issues, we propose a novel Color-guided Spatial-Frequency Fusion Network (CSF) for sandstorm image enhancement. The CSF framework comprises a Color Guided Module (CGM) for correcting color distortions and a Dual-domain Feature Fusion Module (DFFM) that integrates spatial and frequency domain features. CGM dynamically adjusts color information across RGB channels, facilitating preliminary color correction. DFFM extracts multi-scale spatial features and global frequency components, enabling the network to learn robust and discriminative representations. Experimental results on both synthetic and real-world sandstorm datasets demonstrate that CSF outperforms state-of-the-art methods in terms of qualitative and quantitative performance, highlighting its effectiveness for enhancing sandstorm images. Our code will be available at https://github.com/cvandpr/CSF.
</details>


## Environments

The project is built with Python 3.10, Pytorch 1.13.0, CUDA 11.7

## Datasets

Datasets can be found in: https://pan.baidu.com/s/1WrJFPswjfVuCeAfvunNmiw?pwd=ccfz

## Acknowledgements

This project is mainly based on [Selective frequency network for image restoration(ICLR2023)]

