import torch
from torchvision import transforms
from PIL import Image, ImageDraw
from model import R_Net
import os
from dataset import get_rotate_mat
import numpy as np
import lanms
import time
import cv2
import math

def resize_img(img):
	'''resize image to be divisible by 32
	'''
	w, h = img.size
	resize_w = w
	resize_h = h

	resize_h = resize_h if resize_h % 32 == 0 else int(resize_h / 32) * 32
	resize_w = resize_w if resize_w % 32 == 0 else int(resize_w / 32) * 32
	img = img.resize((resize_w, resize_h), Image.BILINEAR)
	ratio_h = resize_h / h
	ratio_w = resize_w / w

	return img, ratio_h, ratio_w


def load_pil(img):
	'''convert PIL Image to torch.Tensor
	'''
	t = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))])
	return t(img).unsqueeze(0)


def is_valid_poly(res, score_shape, scale):
	'''check if the poly in image scope
	Input:
		res        : restored poly in original image
		score_shape: score map shape
		scale      : feature map -> image
	Output:
		True if valid
	'''
	cnt = 0
	for i in range(res.shape[1]):
		if res[0,i] < 0 or res[0,i] >= score_shape[1] * scale or \
           res[1,i] < 0 or res[1,i] >= score_shape[0] * scale:
			cnt += 1
	return True if cnt <= 1 else False


def restore_polys(valid_pos, valid_geo, score_shape, scale=4):
	'''restore polys from feature maps in given positions
	Input:
		valid_pos  : potential text positions <numpy.ndarray, (n,2)>
		valid_geo  : geometry in valid_pos <numpy.ndarray, (5,n)>
		score_shape: shape of score map
		scale      : image / feature map
	Output:
		restored polys <numpy.ndarray, (n,8)>, index
	'''
	polys = []
	index = []
	valid_pos *= scale
	d = valid_geo[:4, :] # 4 x N
	angle = valid_geo[4, :] # N,

	for i in range(valid_pos.shape[0]):
		x = valid_pos[i, 0]
		y = valid_pos[i, 1]
		y_min = y - d[0, i]
		y_max = y + d[1, i]
		x_min = x - d[2, i]
		x_max = x + d[3, i]
		rotate_mat = get_rotate_mat(-angle[i])
		
		temp_x = np.array([[x_min, x_max, x_max, x_min]]) - x
		temp_y = np.array([[y_min, y_min, y_max, y_max]]) - y
		coordidates = np.concatenate((temp_x, temp_y), axis=0)
		res = np.dot(rotate_mat, coordidates)
		res[0,:] += x
		res[1,:] += y
		
		if is_valid_poly(res, score_shape, scale):
			index.append(i)
			polys.append([res[0,0], res[1,0], res[0,1], res[1,1], res[0,2], res[1,2],res[0,3], res[1,3]])
	return np.array(polys), index


def get_boxes(score, geo, score_thresh=0.9, nms_thresh=0.2):
	'''get boxes from feature map
	Input:
		score       : score map from model <numpy.ndarray, (1,row,col)>
		geo         : geo map from model <numpy.ndarray, (5,row,col)>
		score_thresh: threshold to segment score map
		nms_thresh  : threshold in nms
	Output:
		boxes       : final polys <numpy.ndarray, (n,9)>
	'''
	time_6 = time.time()
	score = score[0,:,:]
	xy_text = np.argwhere(score > score_thresh) # n x 2, format is [r, c]
	if xy_text.size == 0:
		return None, 0
	xy_text = xy_text[np.argsort(xy_text[:, 0])]
	valid_pos = xy_text[:, ::-1].copy() # n x 2, [x, y]
	valid_geo = geo[:, xy_text[:, 0], xy_text[:, 1]] # 5 x n
	polys_restored, index = restore_polys(valid_pos, valid_geo, score.shape) 
	if polys_restored.size == 0:
		return None, 0
	boxes = np.zeros((polys_restored.shape[0], 9), dtype=np.float32)
	boxes[:, :8] = polys_restored
	boxes[:, 8] = score[xy_text[index, 0], xy_text[index, 1]]

	boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thresh)

	for i, box in enumerate(boxes):
		mask = np.zeros_like(score, dtype=np.uint8)
		cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
		# print('111111111111-------',np.sum(mask))
		boxes[i, 8] = cv2.mean(score, mask)[0]
	boxes = boxes[boxes[:, 8] > 0.15]

	time_4 = time.time()
	time_nms = time_4 - time_6
	# print('nms---------',time_nms)
	return boxes, time_nms


def adjust_ratio(boxes, ratio_w, ratio_h):
	'''refine boxes
	Input:
		boxes  : detected polys <numpy.ndarray, (n,9)>
		ratio_w: ratio of width
		ratio_h: ratio of height
	Output:
		refined boxes
	'''
	if boxes is None or boxes.size == 0:
		return None
	boxes[:,[0,2,4,6]] /= ratio_w
	boxes[:,[1,3,5,7]] /= ratio_h
	return np.around(boxes)
	
	
def detect(img, model, device):
	'''detect text regions of img using model
	Input:
		img   : PIL Image
		model : detection model
		device: gpu if gpu is available
	Output:
		detected polys
	'''

	img, ratio_h, ratio_w = resize_img(img)

	time_1 = time.time()
	with torch.no_grad():
		score, geo = model(load_pil(img).to(device))
	time_2 = time.time()
	boxes, time_nms = get_boxes(score.squeeze(0).cpu().numpy(), geo.squeeze(0).cpu().numpy())
	time_modle = time_2 - time_1
	return adjust_ratio(boxes, ratio_w, ratio_h), time_modle + time_nms

def detect_dataset(model, device, test_img_path, submit_path):
	'''detection on whole dataset, save .txt results in submit_path
	Input:
		model        : detection model
		device       : gpu if gpu is available
		test_img_path: dataset path
		submit_path  : submit result for evaluation
	'''
	img_files = os.listdir(test_img_path)
	img_files = sorted([os.path.join(test_img_path, img_file) for img_file in img_files])
	time_total = 0
	start = time.time()
	for i, img_file in enumerate(img_files):
		print('evaluating {} image'.format(i), end='\r')
		img = Image.open(img_file)
		im_write = cv2.imread(img_file)[:, :, ::-1]
		boxes,time_temp = detect(img, model, device)
		time_total += time_temp

		if boxes is not None:
			boxes = boxes[:, :8].reshape((-1, 4, 2))
			res_file = os.path.join(
				'./output/2/',
				'res_{}.txt'.format(
					os.path.basename(img_file).split('.')[0]))

			with open(res_file,'w') as f:
				for box in boxes:
					box = np.around(box).astype(np.int32)
					# if np.linalg.norm(box[0] - box[1]) < 2 or np.linalg.norm(box[3] - box[0]) < 2:
					# 	continue
					f.write('{},{},{},{},{},{},{},{}\r\n'.format(
						box[0, 0], box[0, 1], box[1, 0], box[1, 1], box[2, 0], box[2, 1], box[3, 0], box[3, 1],
					))
			cv2.polylines(im_write[:, :, ::-1], [box.astype(np.int32).reshape((-1, 1, 2))], True, color=(255, 255, 0),thickness=3)

		img_path = os.path.join('./output/img/', os.path.basename(img_file))
		cv2.imwrite(img_path, im_write[:, :, ::-1])

	time_total_new = time.time() - start
	print('total_time ----------',time_total)
	print('fps----------',500 / time_total)
	print('fps----------',500 / time_total_new)



if __name__ == '__main__':
	test_path = '/home/yxwang/tensorflow/east/test/testimages'
	sub_path= './output/'
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model = R_Net().to(device)
	# img_path    = '../ICDAR_2015/test_img/img_2.jpg'
	model_path  = '/home/yxwang/pytorch/EAST/pths/pre_synth/fine_tune/7/model_epoch_930.pth'
	model.load_state_dict(torch.load(model_path))
	model.eval()
	# img = Image.open(img_path)

	boxes = detect_dataset(model, device,test_path, sub_path)



