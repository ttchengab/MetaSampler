import argparse
import os
import sys
import logging
import numpy
import numpy as np
import torch
import torch.utils.data
import torchvision
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm

# Only if the files are in example folder.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR[-8:] == 'examples':
	sys.path.append(os.path.join(BASE_DIR, os.pardir))
	os.chdir(os.path.join(BASE_DIR, os.pardir))

from pcrnet.models import PointNet
from pcrnet.models import iPCRNet
from pcrnet.losses import ChamferDistanceLoss
from pcrnet.data_utils import RegistrationData, ModelNet40Data

def _init_(args):
	if not os.path.exists('checkpoints'):
		os.makedirs('checkpoints')
	if not os.path.exists('checkpoints/' + args.exp_name):
		os.makedirs('checkpoints/' + args.exp_name)
	if not os.path.exists('checkpoints/' + args.exp_name + '/' + 'models'):
		os.makedirs('checkpoints/' + args.exp_name + '/' + 'models')
	os.system('cp main.py checkpoints' + '/' + args.exp_name + '/' + 'main.py.backup')
	os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')


class IOStream:
	def __init__(self, path):
		self.f = open(path, 'a')

	def cprint(self, text):
		print(text)
		self.f.write(text + '\n')
		self.f.flush()

	def close(self):
		self.f.close()

def test_one_epoch(device, model, test_loader):
	model.eval()
	test_loss = 0.0
	pred  = 0.0
	count = 0
	for i, data in enumerate(tqdm(test_loader)):
		template, source, igt = data

		template = template.to(device)
		source = source.to(device)
		igt = igt.to(device)

		# mean substraction
		source = source - torch.mean(source, dim=1, keepdim=True)
		template = template - torch.mean(template, dim=1, keepdim=True)

		output = model(template, source)
		loss_val = ChamferDistanceLoss()(template, output['transformed_source'])

		test_loss += loss_val.item()
		count += 1

	test_loss = float(test_loss)/count
	return test_loss

def test(args, model, test_loader, textio):
	test_loss = test_one_epoch(args.device, model, test_loader)
	textio.cprint('Validation Loss: %f'%(test_loss))

def train_one_epoch(device, model, train_loader, optimizer):
	model.train()
	train_loss = 0.0
	pred  = 0.0
	count = 0
	for i, data in enumerate(tqdm(train_loader)):
		template, source, igt = data

		template = template.to(device)
		source = source.to(device)
		igt = igt.to(device)

		# mean substraction
		source = source - torch.mean(source, dim=1, keepdim=True)
		template = template - torch.mean(template, dim=1, keepdim=True)

		output = model(template, source)
		loss_val = ChamferDistanceLoss()(template, output['transformed_source'])
		# if i %10 == 0:
		# 	print(loss_val.item())

		# forward + backward + optimize
		optimizer.zero_grad()
		loss_val.backward()
		optimizer.step()

		train_loss += loss_val.item()
		count += 1

	train_loss = float(train_loss)/count
	return train_loss

def train(args, model, train_loader, test_loader, boardio, textio, checkpoint):
	learnable_params = filter(lambda p: p.requires_grad, model.parameters())
	if args.optimizer == 'Adam':
		optimizer = torch.optim.Adam(learnable_params, lr = 0.001)
	else:
		optimizer = torch.optim.SGD(learnable_params, lr=0.1)

	if checkpoint is not None:
		min_loss = checkpoint['min_loss']
		optimizer.load_state_dict(checkpoint['optimizer'])

	best_test_loss = np.inf

	for epoch in range(args.start_epoch, args.epochs):
		train_loss = train_one_epoch(args.device, model, train_loader, optimizer)
		test_loss = test_one_epoch(args.device, model, test_loader)

		if test_loss<best_test_loss:
			best_test_loss = test_loss
			snap = {'epoch': epoch + 1,
					'model': model.state_dict(),
					'min_loss': best_test_loss,
					'optimizer' : optimizer.state_dict(),}
			torch.save(snap, 'checkpoints/%s/models/best_model_snap.t7' % (args.exp_name))
			torch.save(model.state_dict(), 'checkpoints/%s/models/best_model.t7' % (args.exp_name))
			torch.save(model.feature_model.state_dict(), 'checkpoints/%s/models/best_ptnet_model.t7' % (args.exp_name))

		torch.save(snap, 'checkpoints/%s/models/model_snap.t7' % (args.exp_name))
		torch.save(model.state_dict(), 'checkpoints/%s/models/model.t7' % (args.exp_name))
		torch.save(model.feature_model.state_dict(), 'checkpoints/%s/models/ptnet_model.t7' % (args.exp_name))

		boardio.add_scalar('Train Loss', train_loss, epoch+1)
		boardio.add_scalar('Test Loss', test_loss, epoch+1)
		boardio.add_scalar('Best Test Loss', best_test_loss, epoch+1)

		textio.cprint('EPOCH:: %d, Training Loss: %f, Testing Loss: %f, Best Loss: %f'%(epoch+1, train_loss, test_loss, best_test_loss))

def options():
	parser = argparse.ArgumentParser(description='Point Cloud Registration')
	parser.add_argument('--exp_name', type=str, default='exp_ipcrnet3', metavar='N',
						help='Name of the experiment')
	parser.add_argument('--eval', type=bool, default=False, help='Train or Evaluate the network.')

	# settings for input data
	parser.add_argument('--dataset_type', default='modelnet', choices=['modelnet', 'shapenet2'],
						metavar='DATASET', help='dataset type (default: modelnet)')
	parser.add_argument('--num_points', default=1024, type=int,
						metavar='N', help='points in point-cloud (default: 1024)')

	# settings for PointNet
	parser.add_argument('--pointnet', default='tune', type=str, choices=['fixed', 'tune'],
						help='train pointnet (default: tune)')
	parser.add_argument('--emb_dims', default=1024, type=int,
						metavar='K', help='dim. of the feature vector (default: 1024)')
	parser.add_argument('--symfn', default='max', choices=['max', 'avg'],
						help='symmetric function (default: max)')

	# settings for on training
	parser.add_argument('--seed', type=int, default=1234)
	parser.add_argument('-j', '--workers', default=4, type=int,
						metavar='N', help='number of data loading workers (default: 4)')
	parser.add_argument('-b', '--batch_size', default=32, type=int,
						metavar='N', help='mini-batch size (default: 32)')
	parser.add_argument('--epochs', default=200, type=int,
						metavar='N', help='number of total epochs to run')
	parser.add_argument('--start_epoch', default=0, type=int,
						metavar='N', help='manual epoch number (useful on restarts)')
	parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD'],
						metavar='METHOD', help='name of an optimizer (default: Adam)')
	parser.add_argument('--resume', default='', type=str,
						metavar='PATH', help='path to latest checkpoint (default: null (no-use))')
	parser.add_argument('--pretrained', default='', type=str,
						metavar='PATH', help='path to pretrained model file (default: null (no-use))')
	parser.add_argument('--device', default='cuda:0', type=str,
						metavar='DEVICE', help='use CUDA if available')

	args = parser.parse_args()
	return args

def main():
	args = options()

	torch.backends.cudnn.deterministic = True
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)
	np.random.seed(args.seed)

	boardio = SummaryWriter(log_dir='checkpoints/' + args.exp_name)
	_init_(args)

	textio = IOStream('checkpoints/' + args.exp_name + '/run.log')
	textio.cprint(str(args))


	trainset = RegistrationData('PCRNet', ModelNet40Data(train=True, download=True))
	testset = RegistrationData('PCRNet', ModelNet40Data(train=False, download=True))
	train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.workers)
	test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.workers)

	if not torch.cuda.is_available():
		args.device = 'cpu'
	args.device = torch.device(args.device)

	# Create PointNet Model.
	ptnet = PointNet(emb_dims=args.emb_dims)
	model = iPCRNet(feature_model=ptnet)
	model = model.to(args.device)

	checkpoint = None
	if args.resume:
		assert os.path.isfile(args.resume)
		checkpoint = torch.load(args.resume)
		args.start_epoch = checkpoint['epoch']
		model.load_state_dict(checkpoint['model'])

	if args.pretrained:
		assert os.path.isfile(args.pretrained)
		model.load_state_dict(torch.load(args.pretrained, map_location='cpu'))
	model.to(args.device)

	if args.eval:
		test(args, model, test_loader, textio)
	else:
		train(args, model, train_loader, test_loader, boardio, textio, checkpoint)

if __name__ == '__main__':
	main()
