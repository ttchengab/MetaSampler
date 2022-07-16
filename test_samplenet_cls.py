"""
Author: Benny
Date: Nov 2019
"""
from data_utils.ModelNetDataLoader import ModelNetDataLoader
import argparse
import numpy as np
import os
import torch
import logging
from tqdm import tqdm
import sys
import importlib
from models import samplenet

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Testing')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--log_dir', type=str, default='ptnet_cls_4', help='Experiment root')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--num_votes', type=int, default=3, help='Aggregate classification scores with voting')
    return parser.parse_args()


def test(model, sampler, loader, num_class=40, vote_num=1):
    mean_correct = []
    classifier = model.eval()
    class_acc = np.zeros((num_class, 3))

    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):
        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()

        # points = points.transpose(2, 1)
        vote_pool = torch.zeros(target.size()[0], num_class).cuda()
        simp_pc, proj_pc = sampler(points)
        proj_pc = proj_pc.transpose(2, 1)

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
    return instance_acc, class_acc


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    experiment_dir = 'log/classification/' + args.log_dir

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    data_path = 'data/modelnet40_normal_resampled/'

    test_dataset = ModelNetDataLoader(root=data_path, args=args, split='test', process_data=False)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)

    '''MODEL LOADING'''
    num_class = args.num_category
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    model = importlib.import_module(model_name)

    classifier = model.get_model(num_class, normal_channel=args.use_normals)
    if not args.use_cpu:
        classifier = classifier.cuda()

    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])


    '''Samplenet Loading'''
    out_points = 16
    bottleneck_size = 128
    group_size = 10
    sampler = samplenet.SampleNet(
        num_out_points=out_points,
        bottleneck_size=bottleneck_size,
        group_size=group_size,
        initial_temperature=1.0,
        input_shape="bnc",
        output_shape="bnc",
    )
    # checkpoint = torch.load('log/samplenet/64withbce2021-11-19_09-26-07/best.pth')
    checkpoint = torch.load('log/samplenet_meta/16_3ensemble2021-12-14_04-37-42/best.pth')
    """
    16_3ensemble2021-12-14_04-35-38  16_3tasks_1282021-12-14_00-33-10  16_3tasks_SGD_1282021-12-13_18-38-20
    16_3ensemble2021-12-14_04-37-42  16_3tasks_1282021-12-14_00-49-23  16_3tasks_sgd_2021-12-13_07-07-14
    16_3tasks2021-12-12_21-45-08     16_3tasks_SGD2021-12-13_17-36-39  16no_task2021-12-13_04-50-15
    16_3tasks2021-12-12_22-46-05     16_3tasks_SGD2021-12-13_17-41-15  16single_task2021-12-12_22-41-22
    16_3tasks2021-12-13_08-14-26     16_3tasks_SGD2021-12-13_17-53-11
    """
    #64badtask2021-11-24_19-06-23
    # print('bad task 64')
    sampler.load_state_dict(checkpoint['model_state_dict'])
    print(checkpoint['best_acc'])
    sampler = sampler.cuda()
    sampler.training=False

    with torch.no_grad():
        instance_acc, class_acc = test(classifier.eval(), sampler.eval(), testDataLoader, vote_num=args.num_votes, num_class=num_class)
        log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))


if __name__ == '__main__':
    args = parse_args()
    main(args)
