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
from models import pcn
from data_utils.ModelNetDataLoader import ModelNetDataLoader
from tqdm import tqdm

"""This code bit assumes a defined and pretrained task_model(),
data, and an optimizer."""

"""Get SampleNet parsing options and add your own."""
def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size in training')
    parser.add_argument('--model', default='pointnet_cls', help='model name [default: pointnet_cls]')
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

"""Create a PCN instance."""
pcnet = pcn.PCN()
pcnet = pcnet.cuda()

"""optimizer"""
optimizer = torch.optim.Adam(
    pcnet.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-08,
)
# onet_optimizer = torch.optim.Adam(onet.parameters(), lr=0.0001)

"""Loss Definition"""
chamfer_loss = pcn.ChamferDistanceLoss()

"""Create Ckpt dir"""
saving_dir = 'log/samplenet_meta/'+'pcn'+datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+'/'
# model_dir = 'ckpts/MixupVis'+datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+'/'
os.makedirs(saving_dir)

"""Training routine."""

EPOCHS = 100
best_chamfer = 10000
lr = 1e-3
for epoch in range(EPOCHS):
    pcnet.train()
    for idx, (gt_points, _) in enumerate(tqdm(trainDataLoader)):

        gt_points = gt_points.data.numpy()
        gt_points = provider.random_point_dropout(gt_points)
        gt_points[:, :, 0:3] = provider.random_scale_point_cloud(gt_points[:, :, 0:3])
        gt_points[:, :, 0:3] = provider.shift_point_cloud(gt_points[:, :, 0:3])
        gt_points = torch.Tensor(gt_points)
        # points = points.transpose(2, 1)
        gt_points = gt_points.cuda()

        """
        PCN Training
        """
        # print('hiiii')
        # print(gt_points.device)
        # print(pcnet.device)
        # print('whattt')
        predicted_points = pcnet(gt_points)
        loss = chamfer_loss(predicted_points['coarse_output'], gt_points)

        if idx % 20 == 0:
            print("Epoch {}/{} chamfer_distance is {}"\
                                    .format(epoch, EPOCHS, loss))

        # update sampler
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    with torch.no_grad():
        pcnet.eval()
        val_loss = 0
        total = 0
        for (gt_points, _) in tqdm(testDataLoader):
            if not args.use_cpu:
                gt_points = gt_points.cuda()
            predicted_points = pcnet(gt_points)
            val_loss += chamfer_loss(predicted_points['coarse_output'], gt_points)
            # points = points.transpose(2, 1)
            total += gt_points.shape[0]

        avg_chamfer = val_loss/total
        print('Current Average Chamfer Distance: {}'.format(avg_chamfer))
        print('Best Chamfer: {}'.format(best_chamfer))
        if best_chamfer > avg_chamfer:
            print("Saved best checkpoint")
            best_chamfer = avg_chamfer
            checkpoint = {'epoch': epoch, 'best_chamfer': best_chamfer,'model_state_dict': pcnet.state_dict()}
            torch.save(checkpoint, saving_dir+'/best.pth')
