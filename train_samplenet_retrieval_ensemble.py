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


def test(model, sampler, loader, num_class=40):
    mean_correct = []
    class_acc = np.zeros((num_class, 3))
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
        # points = points.transpose(2,1)
        best_pred = -1
        best_pred_val = 0
        # print(labels)
        simp_pc, proj_pc = sampler(points)
        proj_pc = proj_pc.transpose(2, 1)
        for i, target_cloud in enumerate(target_sets):
            # TODO, cuda and transpose
            target_cloud = target_cloud.cuda()
            target_cloud = target_cloud.transpose(2, 1)
            # print(target_cloud.shape)
            pred_val = classifier(proj_pc, target_cloud)
            if pred_val > best_pred_val:
                best_pred_val = pred_val
                best_pred = i
        if best_pred == labels:
            correct += 1
        # print("labels {}, predictions {}".format(best_pred, labels))
        total += 1

        # if total % 50 == 0:
        #     print('Accuracy {}'.format(correct/total))
    acc = correct/total

    return acc


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = 'retrievalnet_ensemble' + str(out_points) + '_' + str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('shape_retrieval_sample')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    data_path = 'data/modelnet40_normal_resampled/'

    train_dataset = ModelNetDataLoader(root=data_path, args=args, split='train', process_data=args.process_data)
    test_dataset = ModelNetTestLoader(root=data_path, args=args, split='test', process_data=args.process_data)
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=10)

    '''MODEL LOADING'''
    num_class = args.num_category
    model = importlib.import_module(args.model)
    shutil.copy('./models/%s.py' % args.model, str(exp_dir))
    shutil.copy('models/pointnet2_utils.py', str(exp_dir))
    shutil.copy('./train_classification.py', str(exp_dir))

    '''Initialize training network for shape retrieval'''
    # classifier = model.get_model(normal_channel=args.use_normals)
    # checkpoint = torch.load('log/shape_retrieval/shape_ret1/checkpoints/best_model.pth')
    # checkpoint = torch.load('log/classification/2021-11-19_07-05/checkpoints/best_model.pth')
    # print('Loading the best model')

    checkpoints_dir_ensem = ['shape_ret1','shape_ret2','shape_ret3']
    num_tasks = len(checkpoints_dir_ensem)
    task_models = []
    for i in range(num_tasks):
        retrievalnet_joint = model.get_model(normal_channel=args.use_normals)
        checkpoint = torch.load('log/shape_retrieval/' + checkpoints_dir_ensem[i] + '/checkpoints/best_model.pth')
        retrievalnet_joint.load_state_dict(checkpoint['model_state_dict'])
        for param in retrievalnet_joint.parameters():
            param.requires_grad = False
        if not args.use_cpu:
            retrievalnet_joint = retrievalnet_joint.cuda()
        retrievalnet_joint.eval()
        task_models.append(retrievalnet_joint)


    '''Initialize testing network for shape retrieval'''
    print('Loading the test model')
    test_network = model.get_model(normal_channel=args.use_normals)
    checkpoint = torch.load('log/shape_retrieval/test_net/checkpoints/best_model.pth')
    # checkpoint = torch.load('log/classification/2021-11-19_07-05/checkpoints/best_model.pth')
    test_network.load_state_dict(checkpoint['model_state_dict'])
    for param in test_network.parameters():
        param.requires_grad = False
    if not args.use_cpu:
        test_network = test_network.cuda()

    criterion = nn.BCELoss()
    if not args.use_cpu:
        criterion = criterion.cuda()


    print('change optimizer to sampler')
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            sampler.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(sampler.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    global_epoch = 0
    global_step = 0
    best_acc = 0.0
    best_epoch = 0

    '''TRAINING'''
    logger.info('Start training...')
    start_epoch = 0
    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        mean_correct = []

        scheduler.step()
        sampler.train()
        sampler.training=True
        for batch_id, (orig_points, diff_points) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()

            points = orig_points.clone().detach()
            points = points.data.numpy()
            points = provider.random_point_dropout(points)
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            points = points.transpose(2, 1)
            # orig_points = orig_points.transpose(2,1)
            diff_points = diff_points.transpose(2,1)
            # print("Shapes orig points {}, points {}, diff_points {}".format(orig_points.shape, points.shape, diff_points.shape))

            if not args.use_cpu:
                orig_points, points, diff_points = orig_points.cuda(), points.cuda(), diff_points.cuda()
            # print(points.shape)
            simp_pc, proj_pc = sampler(orig_points)
            proj_pc = proj_pc.transpose(2, 1)

            loss = 0.

            for i in range(num_tasks):
                one_pred = task_models[i](proj_pc, points)
                ones = torch.ones(orig_points.shape[0], 1)
                zero_pred = task_models[i](proj_pc, diff_points)
                zeros = torch.zeros(orig_points.shape[0], 1)
                preds = torch.cat((one_pred, zero_pred)).cuda()
                labels = torch.cat((ones, zeros)).cuda()
                loss += criterion(preds, labels)

            simplification_loss = sampler.get_simplification_loss(
                    orig_points, simp_pc, out_points
            )
            projection_loss = sampler.get_projection_loss()
            samplenet_loss = simplification_loss + projection_loss

            loss += samplenet_loss
            loss.backward()
            optimizer.step()
            global_step += 1

        log_string('Train Instance Loss: %f' % loss)

        with torch.no_grad():
            acc = test(test_network.eval(), sampler.eval(), testDataLoader, num_class=num_class)

            if (acc >= best_acc):
                best_acc = acc
                best_epoch = epoch

            log_string('Test Instance Accuracy: %f' % (acc))
            log_string('Best Instance Accuracy: %f' % (best_acc))

            if (acc >= best_acc):
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': acc,
                    'model_state_dict': sampler.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            global_epoch += 1

    logger.info('End of training...')


if __name__ == '__main__':
    args = parse_args()
    main(args)
