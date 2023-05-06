# [ECCV 2022] Meta-Sampler: Almost Universal yet Task-Oriented Sampling for Point Clouds
This is the PyTorch implementation of the paper Meta-Sampler: Almost Universal yet Task-Oriented Sampling for Point Clouds which will appear in ECCV-2022 Conference. ** The readability of the code will continue to be polished. **

![Overview](https://github.com/ttchengab/MetaSampler/blob/main/overview.png)

## Cite this work

```
@inproceedings{metasampler,
  title={Meta-Sampler: Almost Universal yet Task-Oriented Sampling for Point Clouds},
  author={Cheng, Ta-Ying and 
          Hu, Qingyong and 
          Xie, Qian and 
          Trigoni, Niki and 
          Markham, Andrew},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2022}
}
```

## Preliminaries
The meta-sampler was built on top of the official [PyTorch SampleNet implementation](https://github.com/itailang/SampleNet) and the training algorithm is performed on pretrained point cloud networks: [PointNet/PointNet2](https://github.com/yanx27/Pointnet_Pointnet2_pytorch), [Point Completion Network (PCN)](https://github.com/vinits5/learning3d/tree/master/models), and [PCRNet](https://github.com/vinits5/pcrnet_pytorch). Please cite them accordingly when using their code. The essential components from SampleNet and PointNet/PCN are contained in this repository. To test on PCRNet, clone the pcrnet implementation into this github repository.

The code uses standard [ModelNet40 dataset](https://modelnet.cs.princeton.edu) that can also be obtained [here](https://github.com/yanx27/Pointnet_Pointnet2_pytorch).

## Training models for using joint training

To perform the proposed joint-training on a particular network task using unsampled point clouds:

```shell
python train_samplenet_networktask_ensemble.py
```

and replace ```networktask``` with the designated task (e.g., classification/pcn/retrieval).


## Training meta-sampler

To perform meta-sampler training:

```shell
python train_samplenet_meta.py
```

Checkpoints can be found [here](https://drive.google.com/drive/folders/1EIhRHAsyS6EVSBs75X0J8Va30psudwjp?usp=sharing).
