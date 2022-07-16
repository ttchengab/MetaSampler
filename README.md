# Meta-Sampler: Almost Universal yet Task-Oriented Sampling for Point Clouds
This is PyTorch implementation of the paper Meta-Sampler: Almost Universal yet Task-Oriented Sampling for Point Clouds which will appear in ECCV-2022 Conference.

## Cite this work

```
@inproceedings{metasampler,
  title={Meta-Sampler: Almost Universal yet Task-Oriented Sampling for Point Clouds},
  author={Cheng, Ta-Ying and 
          Hu, Qingyong and 
          Xie, Qian and 
          Trigoni, Niki and 
          Markham, Andrew},
  booktitle={ECCV},
  year={2022}
}
```

## Preliminaries
The meta-sampler was built on top of the official [PyTorch SampleNet implementation](https://github.com/itailang/SampleNet) and the training algorithm is performed on pretrained point cloud networks: [PointNet/PointNet2](https://github.com/yanx27/Pointnet_Pointnet2_pytorch), [Point Completion Network (PCN)](https://github.com/vinits5/learning3d/tree/master/models), and [PCRNet](https://github.com/vinits5/pcrnet_pytorch). The essential components from SampleNet and PointNet/PCN are contained in this repository. To test on PCRNet, clone the pcrnet implementation into this github repository.

## Training models for using joint training

To perform the proposed joint-training on a particular network task using unsampled point clouds:

```shell
python train_networktask_ensemble.py
```

and replace ```networktask``` with the designated task (e.g., classification).


## Training meta-sampler

To perform meta-sampler training:
```shell
python train_samplenet_meta.py
```

