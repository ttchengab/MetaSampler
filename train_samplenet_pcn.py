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
from models import pointnet_cls
from models import pcn
from models import samplenet as samplenet
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
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
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

"""
Hyperparameters
"""

out_points = 128
bottleneck_size = 128
group_size = 10

print("Target Sample Number: {}".format(out_points))

"""Create a SampleNet sampler instance."""
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

num_class = args.num_category
model = pointnet_cls

# Testing on the original network
test_network = pcn.PCN()
for param in test_network.parameters():
    param.requires_grad = False
if not args.use_cpu:
    test_network = test_network.cuda()
checkpoint = torch.load('log/samplenet_meta/pcn_test/best.pth')
test_network.load_state_dict(checkpoint['model_state_dict'])
test_network.eval()

pcnet = pcn.PCN()
for param in pcnet.parameters():
    param.requires_grad = False
if not args.use_cpu:
    pcnet = pcnet.cuda()
checkpoint = torch.load('log/samplenet_meta/pcn_1/best.pth')
pcnet.load_state_dict(checkpoint['model_state_dict'])
pcnet.eval()
"""Loss Definition"""
chamfer_loss = pcn.ChamferDistanceLoss()

"""For inference time behavior, set sampler.training = False."""
sampler.training = True

"""optimizer"""
optimizer = torch.optim.Adam(
    sampler.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-08,
)
# onet_optimizer = torch.optim.Adam(onet.parameters(), lr=0.0001)

"""Create Ckpt dir"""
saving_dir = 'log/samplenet_meta/'+str(out_points)+'pcn_singletask_'+datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+'/'
# model_dir = 'ckpts/MixupVis'+datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+'/'
os.makedirs(saving_dir)

"""Training routine."""

EPOCHS = 100
best_chamfer = 10000
for epoch in range(EPOCHS):
    sampler.training = True
    sampler.train()
    # onet.train()
    for idx, (points, _) in enumerate(tqdm(trainDataLoader)):

        points = points.data.numpy()
        points = provider.random_point_dropout(points)
        points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
        points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
        points = torch.Tensor(points)
        # points = points.transpose(2, 1)
        points = points.cuda()

        simp_pc, proj_pc = sampler(points)
        # proj_pc = proj_pc.transpose(2, 1)
        predicted_points = pcnet(proj_pc)
        c_loss = chamfer_loss(predicted_points['coarse_output'], points)
        simplification_loss = sampler.get_simplification_loss(
                points, simp_pc, out_points
        )
        projection_loss = sampler.get_projection_loss()
        samplenet_loss = simplification_loss + projection_loss
        loss = c_loss + samplenet_loss

        if idx % 20 == 0:
            print("Epoch {}/{} Current Loss is {}, chamfer is {}, simplification_loss is {}, projection_loss is {}"\
                                    .format(epoch, EPOCHS, loss, c_loss, simplification_loss, projection_loss))

        # Backward + Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        test_network.eval()
        val_loss = 0
        total = 0
        for (gt_points, _) in tqdm(testDataLoader):
            if not args.use_cpu:
                gt_points = gt_points.cuda()
            simp_pc, proj_pc = sampler(gt_points)
            predicted_points = test_network(proj_pc)
            val_loss += chamfer_loss(predicted_points['coarse_output'], gt_points)
            # points = points.transpose(2, 1)
            total += gt_points.shape[0]

        avg_chamfer = val_loss/total
        print('Current Average Chamfer Distance: {}'.format(avg_chamfer))
        print('Best Chamfer: {}'.format(best_chamfer))
        if best_chamfer > avg_chamfer:
            print("Saved best checkpoint")
            best_chamfer = avg_chamfer
            checkpoint = {'epoch': epoch, 'best_chamfer': best_chamfer,'model_state_dict': sampler.state_dict()}
            torch.save(checkpoint, saving_dir+'/best.pth')
