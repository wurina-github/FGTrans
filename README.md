<p align="center">
    <h2 align="center">Functional Grasp Transfer across a Category of
Objects from Only One Labeled Instance</h2>
    <p align="center">Rina Wu, Tianqiang Zhu, Wanli Peng, Jinglue Hang, Yi Sun &dagger; ；<br />
    Dalian University of Technology<br />
    &dagger; corresponding author<br />
    <a href='hhttps://ieeexplore.ieee.org/document/10079105'>IEEE Xplore</a>
</p>
Our initial research has been accepted by [IEEE Robotics and Automation Letters(RAL)](https://ieeexplore.ieee.org/document/10079105).

## Contents

1. [Overview](#overview)

2. [Requirements](#requirements)

3. [Data Preprocess](#data-preprocess)

4. [Training--Learning Categorical Dense Correspondence](#training--learning-categorical-dense-correspondence)

5. [Post processing of generated data](#post-processing-of-generated-data)

6. [Acknowledgment](#acknowledgment)

7. [Citation](#citation)

8. [License](#license)

## Overview

To assist or replace human beings in completing various tasks, research on the functional grasp synthesis of dexterous hands with high degree-of-freedom (DoF) is necessary and challenging. The dexterous functional grasp requires not only that the grasp is stable but more importantly facilitates the functional manipulation after grasping. Such work still relies on manual annotation when collecting data. To this end, we propose a category-level multi-fingered functional grasp transfer framework, in which we only need to label the hand-object contact relationship on functional parts of one object, and then transfer the contact information through the dense correspondence of functional parts between objects, so as to achieve the functional grasp synthesis for new objects based on the transferred hand-object contact information. We verify this method on three categories of representative objects through simulation experiments and achieve successful functional grasps by labeling only one instance in each category.


## Requirements

- Ubuntu 18.04 or 20.04

- Python 3.7

- PyTorch 1.8.1

- [PyTorch3D](https://github.com/facebookresearch/pytorch3d)

- torchmeta 1.8.0

- NVIDIA driver version >= 460.32

- CUDA 10.2

## Data Preprocess

- Download source meshes and grasp labels for knife, bottle categories from [ShapeNetCore](https://shapenet.org/download/shapenetcore) dataset.

- Copy all folders in '02876657'(bottle) to datasets, and file '03624134'(knife) as above. And arrange the files as follows:

```
|-- FGtrans
    |-- datasets
        |-- obj
            |-- knife
            |-- bottle
                |-- files
```

    (*Here we take **bottle** as an example.*)

    ```shell
    # Preprocess data, run:
    python processdata/preprocess_ShapeNet.py --category bottle
    
    # Generate sdf values, run:
    python processdata/generate_sdf.py --category bottle --mode train
    python processdata/generate_sdf.py --category bottle --mode eval
    ```

## Training--Learning Categorical Dense Correspondence

- To train **DIF**, run the following commands.

    ```shell
    python DIF/train.py \
    --config DIF/configs/train/bottle.yml
    ```
    
- To transfer touch code, run the following commands. Please modify the path of the split in the config file, such as "split/eval/batch. txt" to "split/train/batch. txt" to generate transfer data for all objects.

    ```shell
    python DIF/evaluate.py \
    --config DIF/configs/eval/bottle.yml
    ```
    
## Post processing of generated data

- Convert the touch code to the original data.

     ```shell
    python processdata/change_to_real_obj.py --category bottle --mode train
    python processdata/change_to_real_obj.py --category bottle --mode eval
    
    python processdata/remove_redundancy.py --category bottle
    
    ```
    
## Acknowledgment

This repo is based on [DIF-Net](https://github.com/microsoft/DIF-Net). Many thanks for their excellent works.

## Citation

```BibTeX
@article{wu2023functional,
  title={Functional Grasp Transfer Across a Category of Objects From Only one Labeled Instance},
  author={Wu, Rina and Zhu, Tianqiang and Peng, Wanli and Hang, Jinglue and Sun, Yi},
  journal={IEEE Robotics and Automation Letters},
  volume={8},
  number={5},
  pages={2748--2755},
  year={2023},
  publisher={IEEE}
}
```
## License

Our code is released under [MIT License](./LICENSE).
