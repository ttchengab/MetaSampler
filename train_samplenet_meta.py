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
from models import pointnet_retrieval
from data_utils.ModelNetMetaLoader import ModelNetDataLoader
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
# test_network = model.get_model(num_class, normal_channel=args.use_normals)
# for param in test_network.parameters():
#     param.requires_grad = False
# if not args.use_cpu:
#     test_network = test_network.cuda()
# checkpoint = torch.load('log/classification/2021-11-19_07-05/checkpoints/best_model.pth')
# print('testing network is suboptimal')
# test_network.load_state_dict(checkpoint['model_state_dict'])
# test_network.eval()
#
# classifier = model.get_model(num_class, normal_channel=args.use_normals)
# # classifier.apply(inplace_relu)
# criterion = model.get_loss()
# checkpoint = torch.load('log/classification/2021-11-19_07-05/checkpoints/best_model.pth')
# # checkpoint = torch.load('log/classification/2021-11-19_07-05/checkpoints/best_model.pth')
# print('Loading the best model')
# classifier.load_state_dict(checkpoint['model_state_dict'])
# for param in classifier.parameters():
#     param.requires_grad = False
# if not args.use_cpu:
#     classifier = classifier.cuda()
# classifier.eval()

# pcnet = pcn.PCN()
# pcnet = pcnet.cuda()
# checkpoint = torch.load('pcn_dir')
# print('Loading the best pcn model')
# pcnet.load_state_dict(checkpoint['model_state_dict'])
# for param in pcnet.parameters():
#     param.requires_grad = False
# pcnet.eval()

"""Initialize meta learning models"""
checkpoints_dir = ['ptnet_cls_1','ptnet_cls_2','ptnet_cls_3']
num_tasks = len(checkpoints_dir)
task_models = []
for i in range(num_tasks):
    classifier = model.get_model(num_class, normal_channel=args.use_normals)
    criterion = model.get_loss()
    checkpoint = torch.load('log/classification/' + checkpoints_dir[i] + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])
    for param in classifier.parameters():
        param.requires_grad = False
    if not args.use_cpu:
        classifier = classifier.cuda()
    classifier.eval()
    task_models.append(classifier)
    # task_losses.append(criterion)

"""ADD EVERYTHING INTO ONE TASK MODEL LIST WITH CORRESPONDING LOSSES"""
# TODO: Finish Loading All PCNs
checkpoints_dir = ['pcn_1','pcn_2','pcn_3']
num_tasks = len(checkpoints_dir)
for i in range(num_tasks):
    pcn_joint = pcn.PCN()
    checkpoint = torch.load('log/samplenet_meta/' + checkpoints_dir[i] + '/best.pth')
    pcn_joint.load_state_dict(checkpoint['model_state_dict'])
    for param in pcn_joint.parameters():
        param.requires_grad = False
    if not args.use_cpu:
        pcn_joint = pcn_joint.cuda()
    pcn_joint.eval()
    task_models.append(pcn_joint)
    # task_losses.append()
chamfer_loss = pcn.ChamferDistanceLoss()
#
"""Add all the shape retrieval models"""
checkpoints_dir = ['shape_ret1','shape_ret2','shape_ret3']
num_tasks = len(checkpoints_dir)
for i in range(num_tasks):
    retrievalnet_joint = pointnet_retrieval.get_model(normal_channel=args.use_normals)
    checkpoint = torch.load('log/shape_retrieval/' + checkpoints_dir[i] + '/checkpoints/best_model.pth')
    retrievalnet_joint.load_state_dict(checkpoint['model_state_dict'])
    for param in retrievalnet_joint.parameters():
        param.requires_grad = False
    if not args.use_cpu:
        retrievalnet_joint = retrievalnet_joint.cuda()
    retrievalnet_joint.eval()
    task_models.append(retrievalnet_joint)
    # task_losses.append()
bceloss = nn.BCELoss()
#
# # TODO: Add training with shape retrieval




num_tasks = len(task_models)

maml_samplers = []
for i in range(num_tasks):
    maml_sampler = samplenet.SampleNet(
        num_out_points=out_points,
        bottleneck_size=bottleneck_size,
        group_size=group_size,
        initial_temperature=1.0,
        input_shape="bnc",
        output_shape="bnc",
    )
    if not args.use_cpu:
        maml_sampler = maml_sampler.cuda()
    maml_samplers.append(maml_sampler)


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
saving_dir = 'log/samplenet_meta_all/'+str(out_points)+'_'+str(num_tasks)+'all_24_'+datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+'/'
# model_dir = 'ckpts/MixupVis'+datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+'/'
os.makedirs(saving_dir)

"""Training routine."""

EPOCHS = 100
lr = 1e-3
num_steps = 5
for epoch in range(EPOCHS):
    sampler.training = True
    sampler.train()
    # onet.train()
    if epoch > 50:
        lr *= 0.1
    for idx, (points, diff_points, target) in enumerate(tqdm(trainDataLoader)):

        orig_points = points.clone().detach()
        orig_points = orig_points.transpose(2, 1)
        diff_points = diff_points.transpose(2,1)
        points = points.data.numpy()
        points = provider.random_point_dropout(points)
        points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
        points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
        points = torch.Tensor(points)
        # points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        orig_points = orig_points.cuda()
        diff_points = diff_points.cuda()

        """
        Perform Meta Learning
        """
        # Add all the sample networks from samplenets and train
        for i in range(len(task_models)):
            maml_samplers[i].load_state_dict(sampler.state_dict())
            maml_samplers[i].training = True
            maml_optimizer = torch.optim.SGD(
                maml_sampler.parameters(),
                lr=0.001
            )
            for j in range(num_steps):
                simp_pc, proj_pc = maml_samplers[i](points)

                if i < 3:
                    proj_pc = proj_pc.transpose(2, 1)
                    pred, trans_feat = task_models[i](proj_pc)
                    loss = criterion(pred, target.long(), trans_feat)
                elif i < 6:
                    predicted_points = task_models[i](proj_pc)
                    loss = chamfer_loss(predicted_points['coarse_output'], points)
                else:
                    proj_pc = proj_pc.transpose(2, 1)
                    one_pred = task_models[i](proj_pc, orig_points)
                    ones = torch.ones(points.shape[0], 1)
                    zero_pred = task_models[i](proj_pc, diff_points)
                    zeros = torch.zeros(points.shape[0], 1)
                    preds = torch.cat((one_pred, zero_pred)).cuda()
                    labels = torch.cat((ones, zeros)).cuda()
                    loss = bceloss(preds, labels)

                maml_optimizer.zero_grad()
                loss.backward()
                maml_optimizer.step()

        # do the outside update
        grads = ''
        sum_losses = 0.
        for i in range(len(task_models)):
            simp_pc, proj_pc = maml_samplers[i](points)

            if i < 3:
                proj_pc = proj_pc.transpose(2, 1)
                pred, trans_feat = task_models[i](proj_pc)
                sum_losses += criterion(pred, target.long(), trans_feat)
            elif i < 6:
                predicted_points = task_models[i](proj_pc)
                sum_losses += chamfer_loss(predicted_points['coarse_output'], points)
            else:
                proj_pc = proj_pc.transpose(2, 1)
                one_pred = task_models[i](proj_pc, orig_points)
                ones = torch.ones(points.shape[0], 1)
                zero_pred = task_models[i](proj_pc, diff_points)
                zeros = torch.zeros(points.shape[0], 1)
                preds = torch.cat((one_pred, zero_pred)).cuda()
                labels = torch.cat((ones, zeros)).cuda()
                sum_losses += bceloss(preds, labels)

            if grads == '':
                grads = torch.autograd.grad(sum_losses, maml_samplers[i].parameters())
            else:
                grads += torch.autograd.grad(sum_losses, maml_samplers[i].parameters())


        # add in projection and simplification_loss
        simp_pc, proj_pc = sampler(points)
        # bceloss = criterion(pred, target.long(), trans_feat)
        simplification_loss = sampler.get_simplification_loss(
                points, simp_pc, out_points
        )
        projection_loss = sampler.get_projection_loss()
        samplenet_loss = simplification_loss + projection_loss

        if idx % 20 == 0:
            print("Epoch {}/{} Sum_loss is {}, simplification_loss is {}, projection_loss is {}"\
                                    .format(epoch, EPOCHS, sum_losses, simplification_loss, projection_loss))

        # update sampler
        for p, grad in zip(sampler.parameters() ,grads):
            p.data.sub_(grad * lr)
        optimizer.zero_grad()
        samplenet_loss.backward()
        optimizer.step()

        """
        End Meta Learning
        """
    if epoch % 10 == 0:
            print("Saving current checkpoint")
            checkpoint = {'epoch': epoch, 'model_state_dict': sampler.state_dict()}
            torch.save(checkpoint, saving_dir+'/'+str(epoch)+'.pth')
