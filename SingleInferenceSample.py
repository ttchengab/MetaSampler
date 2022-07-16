from models import samplenet
import argparse
import numpy as np
import os
import torch
import logging
from tqdm import tqdm
import sys
import importlib

out_points = 32
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

"""
PRETRAINED META SAMPLERS
128_9all_24_2022-02-19_07-58-57                32_9all_24_2022-02-15_07-51-03
16_9all_24_2022-02-13_18-11-02                 64_9all_24_2022-02-18_08-40-02
32_3ensemble_cls_2022-02-17_07-33-41            32_3pcn_recon2022-02-22_03-04-04
retrievalnet32_2022-02-11_07-16   retrievalnet_ensemble32_2022-02-12_19-06
"""
sampler_ckpt = torch.load('log/shape_retrieval_sample/retrievalnet32_2022-02-11_07-16/checkpoints/best_model.pth')
sampler.load_state_dict(sampler_ckpt['model_state_dict'])
sampler = sampler.cuda()
sampler.training=False
sampler.eval()

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc
model_cat = 'piano'
model_id = '70'
print('shape_retrieval')
point_set = np.loadtxt('data/modelnet40_normal_resampled/'+model_cat+'/'+model_cat+'_00'+model_id+'.txt', delimiter=',').astype(np.float32)
point_set = point_set[0:1024, :]
point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
point_set = point_set[:, 0:3]
points = torch.Tensor(point_set).unsqueeze(0).cuda()
# points = torch.cat((points,points)).cuda()
simp_pc, proj_pc = sampler(points)
np_proj = proj_pc.squeeze(0).cpu().detach().numpy()
np.savetxt(model_cat + model_id + 'singlesr.txt', np_proj)
