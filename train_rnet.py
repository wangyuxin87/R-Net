import torch
from torch.utils import data
from torch import nn
from torch.optim import lr_scheduler
from data_ import dataset_load
from model import R_Net
from loss import Loss,Loss_val
import os
import time


def train(train_img_path, train_gt_path, pths_path, batch_size, lr, num_workers, epoch_iter, interval, val_img_path,val_gt_path):
	file_num = len(os.listdir(train_img_path))
	trainset = dataset_load(train_img_path, train_gt_path)

	train_loader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

	valset = dataset_load(val_img_path, val_gt_path)
	val_loader = data.DataLoader( valset ,batch_size=1, shuffle=False,num_workers=1,pin_memory=True)
	criterion = Loss()
	criterion_new = Loss_val()
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model = R_Net()

	data_parallel = False
	if torch.cuda.device_count() > 1:
		model = nn.DataParallel(model)
		data_parallel = True
	model.to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[(epoch_iter)//3,(epoch_iter)*2//3], gamma=0.1)

	best = 100

	for epoch in range(epoch_iter):	
		model.train()
		scheduler.step()
		epoch_loss = 0
		epoch_time = time.time()
		for i, (img, gt_score, gt_geo, ignored_map) in enumerate(train_loader):
			start_time = time.time()
			img, gt_score, gt_geo, ignored_map = img.to(device), gt_score.to(device), gt_geo.to(device), ignored_map.to(device)
			pred_score, pred_geo = model(img)
			loss = criterion(gt_score, pred_score, gt_geo, pred_geo, ignored_map)
			
			epoch_loss += loss.item()
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			print('Epoch is [{}/{}], mini-batch is [{}/{}], time consumption is {:.8f}, batch_loss is {:.8f}'.format(\
              epoch+1, epoch_iter, i+1, int(file_num/batch_size), time.time()-start_time, loss.item()))
		
		print('epoch_loss is {:.8f}, epoch_time is {:.8f}'.format(epoch_loss/int(file_num/batch_size), time.time()-epoch_time))
		print(time.asctime(time.localtime(time.time())))
		print('='*50)
		if (epoch + 1) % interval == 0:
			state_dict = model.module.state_dict() if data_parallel else model.state_dict()
			torch.save(state_dict, os.path.join(pths_path, 'model_epoch_{}.pth'.format(epoch+1)))
		if (epoch + 1) % 5 == 0:
			epoch_loss_new = 0
			num = 1
			for num, (img, gt_score, gt_geo, ignored_map) in enumerate(val_loader):
				start_time = time.time()
				img, gt_score, gt_geo, ignored_map = img.to(device), gt_score.to(device), gt_geo.to(
					device), ignored_map.to(device)
				pred_score_new, pred_geo_new = model(img)
				loss = criterion_new(gt_score, pred_score_new, gt_geo, pred_geo_new, ignored_map)

				epoch_loss_new += loss.item()

			epoch_loss_new = epoch_loss_new/num
			print('loss is {}-----best is {} ------num is {}'.format(epoch_loss_new,best,num))
			if epoch_loss_new < best and epoch > 600:
				best = epoch_loss_new
				state_dict = model.module.state_dict() if data_parallel else model.state_dict()
				torch.save(state_dict, os.path.join(pths_path, 'model_best.pth'))

if __name__ == '__main__':
	train_path_img = './dataset/train/img'
	train_path_gt  = './dataset/train/txt'
	val_path_img = './dataset/test/image'
	val_path_gt  = './dataset/test/gt'
	pths_path      = './output/checkpoints'
	batch_size     = 8  # 16 for 2 GPUs is default
	learning_rate  = 1e-4
	num_workers    = 16
	epoch_iter     = 600
	save_interval  = 200
	train(train_path_img, train_path_gt, pths_path, batch_size, learning_rate, num_workers, epoch_iter, save_interval, val_path_img, val_path_gt)
	
