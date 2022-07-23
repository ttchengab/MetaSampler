"""
Author: Benny
Date: Nov 2019
"""

import os
import sys
import torch
import numpy as np

import datetime
import logging
import provider
import importlib
import shutil
import argparse

from pathlib import Path
from tqdm import tqdm
from data_utils.ModelNetRetrievalLoader import ModelNetDataLoader, ModelNetTestLoader

import torch.nn as nn

from models import samplenet as samplenet

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


"""
Hyperparameters
"""

out_points = 16
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


checkpoint = torch.load('log/shape_retrieval_sample/retrievalnet16_2022-02-11_18-53/checkpoints/best_model.pth') # Put checkpoint directory here.
sampler.load_state_dict(checkpoint['model_state_dict'])
sampler = sampler.cuda()


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--model', default='pointnet_retrieval', help='model name [default: pointnet_cls]')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--epoch', default=60, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    return parser.parse_args()


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True


def test(model, sampler, loader):
    mean_correct = []
    print("Using the test network")
    classifier = model.eval()
    correct = 0
    total = 0
    print('nway is now 4')
    sampler.eval()
    sampler.training=False
    for j, (points, target_sets, labels) in tqdm(enumerate(loader), total=len(loader)):
        if not args.use_cpu:
            points, labels = points.cuda(), labels.cuda()
        best_pred = -1
        best_pred_val = 0
        simp_pc, proj_pc = sampler(points)
        proj_pc = proj_pc.transpose(2, 1)
        for i, target_cloud in enumerate(target_sets):
            target_cloud = target_cloud.cuda()
            target_cloud = target_cloud.transpose(2, 1)
            pred_val = classifier(proj_pc, target_cloud)
            if pred_val > best_pred_val:
                best_pred_val = pred_val
                best_pred = i
        if best_pred == labels:
            correct += 1
        total += 1

    acc = correct/total

    return acc


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


    '''LOG'''
    args = parse_args()

    '''DATA LOADING'''
    print('Loading dataset')
    data_path = 'data/modelnet40_normal_resampled/'

    test_dataset = ModelNetTestLoader(root=data_path, args=args, split='test', process_data=args.process_data)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=10)


    '''Initialize testing network for shape retrieval'''
    print('Loading the test model')
    model = importlib.import_module(args.model)
    test_network = model.get_model(normal_channel=args.use_normals)
    checkpoint = torch.load('log/shape_retrieval/test_net/checkpoints/best_model.pth')
    test_network.load_state_dict(checkpoint['model_state_dict'])
    for param in test_network.parameters():
        param.requires_grad = False
    if not args.use_cpu:
        test_network = test_network.cuda()

    with torch.no_grad():
        acc = test(test_network.eval(), sampler.eval(), testDataLoader)
        print('Test Accuracy is {}'.format(acc))


if __name__ == '__main__':
    args = parse_args()
    main(args)
