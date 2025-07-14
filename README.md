# FSGNet: A Frequency-Aware and Semantic Guidance Network for Infrared Small Target Detection

**We are preparing to submit the paper for review and will provide code after publication.**

## Recommended Environment
 - [ ] python  3.11.7
 - [ ] pytorch 2.2.1
 - [ ] torchvision 0.17.1

## Datasets
**Our project has the following structure:**
  ```
  ├───dataset/
  │    ├── NUAA-SIRST
  │    │    ├── image
  │    │    │    ├── Misc_1.png
  │    │    │    ├── Misc_2.png
  │    │    │    ├── ...
  │    │    ├── mask
  │    │    │    ├── Misc_1.png
  │    │    │    ├── Misc_2.png
  │    │    │    ├── ...
  │    │    ├── train_NUAA-SIRST.txt
  │    │    │── train_NUAA-SIRST.txt
  │    ├── IRSTD-1K
  │    │    ├── image
  │    │    │    ├── XDU0.png
  │    │    │    ├── XDU1.png
  │    │    │    ├── ...
  │    │    ├── mask
  │    │    │    ├── XDU0.png
  │    │    │    ├── XDU1.png
  │    │    │    ├── ...
  │    │    ├── train_IRSTD-1K.txt
  │    │    ├── train_IRSTD-1K.txt
  │    ├── ...  
  ```
<be>

## Results
#### Visualization results
![outline](image/visual.emf)
#### 3D visualization results
![outline](image/3D.emf)

#### Quantitative Results on NUAA-SIRST, IRSTD-1K, NUDT-SIRST and SIRSTAUG

| Dataset         | IoU (x10(-2)) | nIoU (x10(-2)) | Pd(x10(-2))| Fa (x10(-6))|
| ------------- |:-------------:|:-------------:|:-----:|:-----:|
| NUAA-SIRST    | 78.48  | 79.09  |  96.58 | 21.95 |
| IRSTD-1K      | 72.45  | 68.04  |  92.93 | 5.43 |
| NUDT-SIRST    | 93.78  | 94.14  |  99.26 | 4.89  |
| SIRSTAUG      | 75.69  | 71.70  |  98.76 | 14.36  |
| [[weights]](https://drive.google.com/file/d/1d-RAINGfIoPxOoxcM2moCqVkM9lEanzo/view?usp=sharing)|

*This code is highly borrowed from [SCTransNet](https://github.com/xdFai/SCTransNet). Thanks to Shuai Yuan.

*The overall repository style is highly borrowed from [DNANet](https://github.com/YeRen123455/Infrared-Small-Target-Detection). Thanks to Boyang Li.








