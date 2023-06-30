# shapenet_part_experiments

This repository contains part segmentation experiments conducted on the ShapeNetPart dataset.

## Dataset

The ShapeNetPart dataset is annotated for 3D object part segmentation. It consists of 16,880 models
from 16 shape categories, with 14,006 3D models for training and 2,874 for testing. The number of parts for each
category is between 2 and 6, with 50 different parts in total.

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Experiments

The experiments are conducted on the following models:

- [DGCNN](https://github.com/kentechx/x-dgcnn)
- [PointNet](https://github.com/kentechx/pointnet)
- [PointNet2](https://github.com/kentechx/pointnet)

Training models by running the corresponding scripts in the `code` folder. For example, to train the DGCNN model, run
the following command:

```bash
python code/train_dgcnn.py
```

The processed dataset will be downloaded automatically when running the training scripts.

## Results

The table below presents the classification accuracy of the models on the ShapeNetPart dataset with 2048 points.
The experiments are conducted on a single Nvidia RTX 3090 GPU.

| Model        | input | ins. mIoU | cls. mIoU |
|--------------|-------|-----------|-----------|
| PointNet2SSG | xyz   | 84.8%     | 82.0%     |
| PointNet2MSG | xyz   | 85.2%     | 82.5%     |
| DGCNN        | xyz   | 85.4%     | 83.1%     |

You can reproduce the results by running the corresponding scripts in the `code` folder with default configurations.
For example, to train the PointNet model, run the following command

```bash
python code/train_pointnet.py
```

