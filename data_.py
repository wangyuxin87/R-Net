from PIL import Image
import cv2
import os
import torch
import torchvision.transforms as transforms
from torch.utils import data
from data.rotation import img_rotation, get_rotate, vertices_rotation, pixels_rotation
from data.crop import height_adjusted, crop_img, is_cross_text
from data.points import *


def find_min_rect_angle(vertices):
    '''find the best angle to rotate poly and obtain min rectangle
    '''
    angle_interval = 1
    angle_list = list(range(-90, 90, angle_interval))
    area_list = []
    for theta in angle_list:
        rotated = vertices_rotation(vertices, theta / 180 * math.pi)
        x1, y1, x2, y2, x3, y3, x4, y4 = rotated
        temp_area = (max(x1, x2, x3, x4) - min(x1, x2, x3, x4)) * \
                    (max(y1, y2, y3, y4) - min(y1, y2, y3, y4))
        area_list.append(temp_area)

    sorted_area_index = sorted(list(range(len(area_list))), key=lambda k: area_list[k])
    min_error = float('inf')
    best_index = -1
    rank_num = 10
    # find the best angle with correct orientation
    for index in sorted_area_index[:rank_num]:
        rotated = vertices_rotation(vertices, angle_list[index] / 180 * math.pi)
        temp_error = cal_error(rotated)
        if temp_error < min_error:
            min_error = temp_error
            best_index = index
    return angle_list[best_index] / 180 * math.pi

def gt(img, vertices, labels, scale, length):
    '''generate score gt and geometry gt
    '''
    geo = np.zeros((int(img.height * scale), int(img.width * scale), 5), np.float32)
    score_map = np.zeros((int(img.height * scale), int(img.width * scale), 1), np.float32)
    ignored_map = np.zeros((int(img.height * scale), int(img.width * scale), 1), np.float32)
    temp_mask = np.zeros(score_map.shape[:-1], np.float32)

    index = np.arange(0, length, int(1 / scale))
    index_x, index_y = np.meshgrid(index, index)
    ignored_polys = []
    polys = []

    for i, vertice in enumerate(vertices):
        if labels[i] == 0:
            ignored_polys.append(np.around(scale * vertice.reshape((4, 2))).astype(np.int32))
            continue

        poly = np.around(scale * shrink_poly(vertice).reshape((4, 2))).astype(np.int32)  # scaled & shrinked
        polys.append(poly)

        cv2.fillPoly(temp_mask, [poly], 1)

        theta = find_min_rect_angle(vertice)
        rotate_mat = get_rotate(theta)

        rotated_vertices = vertices_rotation(vertice, theta)
        x_min, x_max, y_min, y_max = get_boundary(rotated_vertices)
        rotated_x, rotated_y = pixels_rotation(rotate_mat, vertice[0], vertice[1], length)

        d1 = rotated_y - y_min
        d1[d1 < 0] = 0
        d2 = y_max - rotated_y
        d2[d2 < 0] = 0
        d3 = rotated_x - x_min
        d3[d3 < 0] = 0
        d4 = x_max - rotated_x
        d4[d4 < 0] = 0

        geo[:, :, 0] += d1[index_y, index_x] * temp_mask
        geo[:, :, 1] += d2[index_y, index_x] * temp_mask
        geo[:, :, 2] += d3[index_y, index_x] * temp_mask
        geo[:, :, 3] += d4[index_y, index_x] * temp_mask
        geo[:, :, 4] += theta * temp_mask

    cv2.fillPoly(ignored_map, ignored_polys, 1)
    cv2.fillPoly(score_map, polys, 1)
    return torch.Tensor(score_map).permute(2, 0, 1), torch.Tensor(geo).permute(2, 0, 1), torch.Tensor(ignored_map).permute(2, 0, 1)


class dataset_load(data.Dataset):
    def __init__(self, img_path, gt_path, scale=0.25, length=512):
        super(dataset_load, self).__init__()
        self.files = [os.path.join(img_path, img_file) for img_file in sorted(os.listdir(img_path))]
        self.gt_files = [os.path.join(gt_path, gt_file) for gt_file in sorted(os.listdir(gt_path))]
        self.scale = scale
        self.length = length

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        with open(self.gt_files[index], 'r',encoding= 'utf-8') as f:
            lines = f.readlines()
        transform = transforms.Compose([transforms.ColorJitter(0.5, 0.5, 0.5, 0.25), transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
        plogs, labels = extract_vertices(lines)

        img = Image.open(self.files[index])
        img, plogs = height_adjusted(img, plogs)
        img, plogs = img_rotation(img, plogs)
        img, plogs = crop_img(img, plogs, labels, self.length)

        score_map, geo_map, ignored_map = gt(img, plogs, labels, self.scale, self.length)
        return transform(img), score_map, geo_map, ignored_map

