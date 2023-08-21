# CoT
Code for the paper **Hierarchical Visual Primitive Experts for Compositional Zero-Shot Learning**, **ICCV 2023**.

Hanjae Kim, Jiyoung Lee, Seongheon Park, Kwanghoon Sohn


## Installation
1. Please create conda environment and install dependencies following the below steps.
```
conda env create --file environment.yml
conda activate cot
```
2. Download C-GQA and VAW-CZSL dataset.
* [C-GQA](https://s3.mlcloud.uni-tuebingen.de/czsl/cgqa-updated.zip)
* [VAW-CZSL](https://drive.google.com/drive/folders/1CalwDXkkGALxz0e-aCFg9xBmf7Pu4eXL?usp=sharing)
3. Download Glove word embedding.
* [Glove](https://drive.google.com/drive/folders/1BE2X70eNMIMkGYwhe01HA4c5jixUQdWd?usp=sharing)
4. Unzip all downloaded files and place it to the dataset folder following the below structures:
```
dataset
└─cgqa
│    └─compositional-split-natural
│    └─images
└─vaw-czsl
│    └─...
└─glove
     └─glove.6B.300d.txt
     └─glove_vocab.txt
``` 
## Training & Testing
1. Update dataset directory in config/*.yml files. 
2. To run training code, type
```
python train.py --cfg config/vaw-czsl.yml
```
3. For testing, type
```
python test.py --cfg config/vaw-czsl.yml --load vaw-czsl.pth
```
## Acknowledgement
Our code is based on the following excellent projects; 
* [ExplanableML/czsl](https://github.com/ExplainableML/czsl)
* [nirat1606/OADis](https://github.com/nirat1606/OADis)
* [lukemelas/PyTorch-Pretrained-ViT](https://github.com/lukemelas/PyTorch-Pretrained-ViT)

## References
This section will be updated after the final version is published.
