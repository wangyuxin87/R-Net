import numpy as np
from PIL import Image
import math

def img_rotation(img, vertices, angle_range=10):
    '''rotate image [-10, 10] degree to aug data
    '''
    angle = angle_range * (np.random.rand() * 2 - 1)
    center_x = (img.width - 1) / 2
    center_y = (img.height - 1) / 2

    img = img.rotate(angle, Image.BILINEAR)
    new_vertices = np.zeros(vertices.shape)

    for i, vertice in enumerate(vertices):
        new_vertices[i, :] = vertices_rotation(vertice, -angle / 180 * math.pi, np.array([[center_x], [center_y]]))
    return img, new_vertices

def get_rotate(theta):
    '''positive theta value means rotate clockwise'''
    return np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])

def vertices_rotation(vertices, theta, anchor=None):
    '''rotate vertices around anchor
    '''
    ver = vertices.reshape((4, 2)).T
    if anchor is None:
        anchor = ver[:, :1]
    rotate = get_rotate(theta)
    res = np.dot(rotate, ver - anchor)
    return (res + anchor).T.reshape(-1)

def pixels_rotation(rotate, anchor_x, anchor_y, length):
    '''get rotated locations of all pixels for next stages
    '''
    x = np.arange(length)
    y = np.arange(length)
    x, y = np.meshgrid(x, y)
    x_l = x.reshape((1, x.size))
    y_l = y.reshape((1, x.size))

    coord = np.concatenate((x_l, y_l), 0)
    rotated_coord = np.dot(rotate, coord - np.array([[anchor_x], [anchor_y]])) + np.array([[anchor_x], [anchor_y]])

    r_x = rotated_coord[0, :].reshape(x.shape)
    r_y = rotated_coord[1, :].reshape(y.shape)
    return r_x, r_y
