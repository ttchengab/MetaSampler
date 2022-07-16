import os
import sys
import numpy as np
from datetime import datetime

import logging
import provider
import importlib
import shutil
import argparse

from pathlib import Path
import torch
import torch.nn as nn
from models import pointnet_onet
from models import samplenet_onet as samplenet
from data_utils.ModelNetSampleLoader import ModelNetDataLoader
from tqdm import tqdm

"""This code bit assumes a defined and pretrained task_model(),
data, and an optimizer."""

"""Get SampleNet parsing options and add your own."""
def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--model', default='pointnet_onet', help='model name [default: pointnet_cls]')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    return parser.parse_args()


"""Create a data loader."""
args = parse_args()
data_path = 'data/modelnet40_normal_resampled/'
train_dataset = ModelNetDataLoader(root=data_path, args=args, split='train', process_data=args.process_data)
test_dataset = ModelNetDataLoader(root=data_path, args=args, split='test', process_data=args.process_data)
trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)

"""
Hyperparameters
"""

out_points = 64
bottleneck_size = 128
group_size = 10

print("Target Sample Number: {}".format(out_points))
"""Create an ONet and a SampleNet sampler instance."""

# onet_model = pointnet_onet
# onet = onet_model.get_model(normal_channel=args.use_normals)

# checkpoint = torch.load('log/onet/sample128_2021-11-19_05-29/checkpoints/best_model.pth')
# onet.load_state_dict(checkpoint['model_state_dict'])

# for param in onet.parameters():
#     param.requires_grad = False
# onet = onet.cuda()
# onet.eval()

# samplenet_model = importlib.import_module('samplenet')
sampler = samplenet.SampleNet(
    num_out_points=out_points,
    bottleneck_size=bottleneck_size,
    group_size=group_size,
    initial_temperature=1.0,
    input_shape="bnc",
    output_shape="bnc",
)
sampler = sampler.cuda()

"""For inference time behavior, set sampler.training = False."""
sampler.training = True


"""distill bce loss"""
def bceloss(pred, target):
    return -1*torch.mean(target * torch.log(pred) + (1 - target) * torch.log(1 - pred + 1e-6))

"""optimizer"""
optimizer = torch.optim.Adam(
    sampler.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-08,
)
# onet_optimizer = torch.optim.Adam(onet.parameters(), lr=0.0001)

"""Create Ckpt dir"""
saving_dir = 'log/samplenet/'+str(out_points)+'withbce'+datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+'/'
print('With BCE Loss')
# model_dir = 'ckpts/MixupVis'+datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+'/'
os.makedirs(saving_dir)


"""Training routine."""

EPOCHS = 100
best_loss = 100000
best_acc = 0.
for epoch in range(EPOCHS):
    sampler.training = True
    sampler.train()
    # onet.train()
    for idx, (points, sample_orig_points, query, target) in enumerate(tqdm(trainDataLoader)):

        points = points.data.numpy()
        points = provider.random_point_dropout(points)
        points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
        points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
        points = torch.Tensor(points)
        # points = points.transpose(2, 1)


"""Only Use sample_orig_points when you want to have an additional ONet"""
        sample_orig_points = sample_orig_points.data.numpy()
        # sample_orig_points = provider.random_point_dropout(sample_orig_points)
        sample_orig_points[:, :, 0:3] = provider.random_scale_point_cloud(sample_orig_points[:, :, 0:3])
        sample_orig_points[:, :, 0:3] = provider.shift_point_cloud(sample_orig_points[:, :, 0:3])
        sample_orig_points = torch.Tensor(sample_orig_points)
        sample_orig_points = sample_orig_points.transpose(2, 1)

        points, sample_orig_points, target, query = points.cuda(), sample_orig_points.cuda(), target.cuda(), query.cuda()

        # Sample and predict
        # Samplenet
        # simp_pc, proj_pc= sampler(points)

        #Samplenet + onet
        simp_pc, proj_pc, onet_pred= sampler(points, query)

        # onet_rep, _ = onet(sample_orig_points, query)
        # proj_pc = proj_pc.transpose(2, 1)
        # onet_pred, _ = onet(proj_pc, query)

        # print(points.shape)
        # print(simp_pc.shape)

        # Compute losses
        simplification_loss = sampler.get_simplification_loss(
                points, simp_pc, out_points
        )
        projection_loss = sampler.get_projection_loss()
        samplenet_loss = simplification_loss + projection_loss

        reconstruction_loss = nn.BCELoss()(onet_pred, target)
        # reconstruction_loss = 0.

        # reconstruction_loss = bceloss(onet_pred, onet_rep) + bce_loss_gt
        # reconstruction_loss = 0

        # Equation (1) in SampleNet paper
        loss = reconstruction_loss + samplenet_loss
        if idx % 20 == 0:
            print("Epoch {}/{} Current Loss is {}, bce is {}, simplification_loss is {}, projection_loss is {}"\
                                    .format(epoch, EPOCHS, loss, reconstruction_loss, simplification_loss, projection_loss))

        # Backward + Optimize
        optimizer.zero_grad()
        # onet_optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # onet_optimizer.step()

    with torch.no_grad():
        sampler.training = False
        sampler.eval()
        onet.eval()
        val_loss = 0
        total = 0
        correct = 0
        for points, sample_orig_points, query, target in tqdm(testDataLoader):

            points = points.data.numpy()
            points = provider.random_point_dropout(points)
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            # points = points.transpose(2, 1)


"""Only Use sample_orig_points when you want to have an additional ONet"""
            sample_orig_points = sample_orig_points.data.numpy()
            # sample_orig_points = provider.random_point_dropout(sample_orig_points)
            sample_orig_points[:, :, 0:3] = provider.random_scale_point_cloud(sample_orig_points[:, :, 0:3])
            sample_orig_points[:, :, 0:3] = provider.shift_point_cloud(sample_orig_points[:, :, 0:3])
            sample_orig_points = torch.Tensor(sample_orig_points)
            sample_orig_points = sample_orig_points.transpose(2, 1)

            points, sample_orig_points, target, query = points.cuda(), sample_orig_points.cuda(), target.cuda(), query.cuda()

            # Sample and predict
            # simp_pc, proj_pc= sampler(points)

            #Samplenet + onet
            simp_pc, proj_pc, onet_pred = sampler(points, query)
            # onet_rep, _ = onet(sample_orig_points, query)
            # proj_pc = proj_pc.transpose(2, 1)
            # onet_pred, _ = onet(proj_pc, query)



            # Compute losses
            simplification_loss = sampler.get_simplification_loss(
                    points, simp_pc, out_points
            )
            projection_loss = sampler.get_projection_loss()
            samplenet_loss = simplification_loss + projection_loss

            # print(onet_pred.shape)
            # print(target.shape)
            # bce_loss_gt = nn.BCELoss()(onet_pred, target)

            # reconstruction_loss = bceloss(onet_pred, onet_rep) + bce_loss_gt
            reconstruction_loss = nn.BCELoss()(onet_pred, target)
            onet_pred = torch.ge(onet_pred, 0.5).float()
            correct += torch.sum(onet_pred == target)
            # reconstruction_loss = bceloss(onet_pred, onet_rep)

            # Equation (1) in SampleNet paper
            val_loss += reconstruction_loss + samplenet_loss
            total += len(points)

        print('Current Validation Loss: {}'.format(val_loss/total))
        cur_acc = correct/total
        print('Current Accuracy: {}, Prev Accuracy: {}'.format(cur_acc, best_acc))
        if val_loss <= best_loss:
            print("Saved best checkpoint")
            best_loss = val_loss
            checkpoint = {'epoch': epoch, 'model_state_dict': sampler.state_dict()}
            torch.save(checkpoint, saving_dir+'/best.pth')

        if best_acc < cur_acc:
            best_acc = cur_acc
