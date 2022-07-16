# Universal Sampling

## Current Steps
Train a pointnet network:

```shell
python train_classification.py
```

Train a SampleNet based on the frozen pointnet:

Add a '.' to ```models/pointnet_cls.py```'s ```from pointnet_utils import PointNetEncoder, feature_transform_reguliarzer``` for this step (sorry!)

```shell
python train_samplenet_badtask.py
```

Test the samplenet:

```shell
python test_samplenet_cls.py
```
