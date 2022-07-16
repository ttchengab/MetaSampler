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
# checkpoint = torch.load('log/samplenet_meta/16_3ensemble2021-12-14_04-37-42/best.pth')
# sampler.load_state_dict(checkpoint['model_state_dict'])
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

classifier = model.get_model(num_class, normal_channel=args.use_normals)
# classifier.apply(inplace_relu)
criterion = model.get_loss()
checkpoint = torch.load('log/classification/2021-11-19_07-05/checkpoints/best_model.pth')
# checkpoint = torch.load('log/classification/2021-11-19_07-05/checkpoints/best_model.pth')
print('Loading the best model')
classifier.load_state_dict(checkpoint['model_state_dict'])
for param in classifier.parameters():
    param.requires_grad = False
if not args.use_cpu:
    classifier = classifier.cuda()
classifier.eval()


"""Initialize meta learning models"""
checkpoints_dir = ['ptnet_cls_1','ptnet_cls_2','ptnet_cls_3']
num_tasks = len(checkpoints_dir)
task_models = []
for i in range(num_tasks):
    classifier_joint = model.get_model(num_class, normal_channel=args.use_normals)
    criterion = model.get_loss()
    checkpoint = torch.load('log/classification/' + checkpoints_dir[i] + '/checkpoints/best_model.pth')
    classifier_joint.load_state_dict(checkpoint['model_state_dict'])
    for param in classifier_joint.parameters():
        param.requires_grad = False
    if not args.use_cpu:
        classifier_joint = classifier_joint.cuda()
    classifier_joint.eval()
    task_models.append(classifier_joint)

# maml_samplers = []
# for i in range(num_tasks):
#     maml_sampler = samplenet.SampleNet(
#         num_out_points=out_points,
#         bottleneck_size=bottleneck_size,
#         group_size=group_size,
#         initial_temperature=1.0,
#         input_shape="bnc",
#         output_shape="bnc",
#     )
#     if not args.use_cpu:
#         maml_sampler = maml_sampler.cuda()
#     maml_samplers.append(maml_sampler)


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
saving_dir = 'log/samplenet_meta/'+str(out_points)+'_'+str(num_tasks)+'ensemble'+datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+'/'
# model_dir = 'ckpts/MixupVis'+datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+'/'
os.makedirs(saving_dir)

"""Training routine."""

EPOCHS = 100
best_loss = 100000
best_acc = 0.
vote_num = 3
lr = 1e-3
num_steps = 5
for epoch in range(EPOCHS):
    sampler.training = True
    sampler.train()
    if epoch > 50:
        lr *= 0.1
    for idx, (points, target) in enumerate(tqdm(trainDataLoader)):

        points = points.data.numpy()
        points = provider.random_point_dropout(points)
        points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
        points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
        points = torch.Tensor(points)
        # points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        simp_pc, proj_pc = sampler(points)
        proj_pc = proj_pc.transpose(2, 1)

        # Add all task models to train
        bceloss = 0.
        for i in range(len(task_models)):
            pred, trans_feat = task_models[i](proj_pc)
            bceloss += criterion(pred, target.long(), trans_feat)

        # add in projection and simplification_loss
        simplification_loss = sampler.get_simplification_loss(
                points, simp_pc, out_points
        )
        projection_loss = sampler.get_projection_loss()
        samplenet_loss = simplification_loss + projection_loss + bceloss

        if idx % 20 == 0:
            print("Epoch {}/{} Sum_bce is {}, simplification_loss is {}, projection_loss is {}"\
                                    .format(epoch, EPOCHS, bceloss, simplification_loss, projection_loss))

        # update sampler
        optimizer.zero_grad()
        samplenet_loss.backward()
        optimizer.step()



    with torch.no_grad():
        sampler.training = False
        sampler.eval()

        mean_correct = []
        class_acc = np.zeros((num_class, 3))
        for points, target in tqdm(testDataLoader):
            if not args.use_cpu:
                points, target = points.cuda(), target.cuda()

            # points = points.transpose(2, 1)
            simp_pc, proj_pc = sampler(points)
            proj_pc = proj_pc.transpose(2, 1)
            vote_pool = torch.zeros(target.size()[0], num_class).cuda()
            for _ in range(vote_num):
                pred, _ = classifier(proj_pc)
                vote_pool += pred
            pred = vote_pool / vote_num
            pred_choice = pred.data.max(1)[1]

            for cat in np.unique(target.cpu()):
                classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
                class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
                class_acc[cat, 1] += 1
            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))

        class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
        class_acc = np.mean(class_acc[:, 2])
        instance_acc = np.mean(mean_correct)

        print('Current Instance Accuracy: {}, Class Accuracy: {}'.format(instance_acc, class_acc))
        print('Best Accuracy: {}'.format(best_acc))
        if best_acc <= instance_acc:
            print("Saved best checkpoint")
            best_acc = instance_acc
            checkpoint = {'epoch': epoch, 'best_acc': instance_acc,'model_state_dict': sampler.state_dict()}
            torch.save(checkpoint, saving_dir+'/best.pth')
