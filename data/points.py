import numpy as np
import math

def move_points(vertices, index1, index2, r, coef):
    '''move the two points to shrink edge
    '''

    index1 = index1 % 4
    index2 = index2 % 4

    r1 = r[index1]
    r2 = r[index2]

    x1 = index1 * 2 + 0
    y1 = index1 * 2 + 1
    x2 = index2 * 2 + 0
    y2 = index2 * 2 + 1

    w = vertices[x1] - vertices[x2]
    h = vertices[y1] - vertices[y2]
    length = eu_distance(vertices[x1], vertices[y1], vertices[x2], vertices[y2])

    if length > 1:
        ratio = (r1 * coef) / length
        vertices[x1] += ratio * (w)
        vertices[y1] += ratio * (h)
        ratio = (r2 * coef) / length
        vertices[x2] += ratio * w
        vertices[y2] += ratio * h
    return vertices


def eu_distance(x1, y1, x2, y2):
    '''Euclidean distance'''
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def shrink_poly(vertices, coef=0.3):
    '''shrink the text region
    '''

    x1, y1, x2, y2, x3, y3, x4, y4 = vertices
    r1 = min(eu_distance(x1, y1, x2, y2), eu_distance(x1, y1, x4, y4))
    r2 = min(eu_distance(x2, y2, x1, y1), eu_distance(x2, y2, x3, y3))
    r3 = min(eu_distance(x3, y3, x2, y2), eu_distance(x3, y3, x4, y4))
    r4 = min(eu_distance(x4, y4, x1, y1), eu_distance(x4, y4, x3, y3))
    r = [r1, r2, r3, r4]

    if eu_distance(x1, y1, x2, y2) + eu_distance(x3, y3, x4, y4) > eu_distance(x2, y2, x3, y3) + eu_distance(x1, y1, x4, y4):
        offset = 0
    else:
        offset = 1

    v = vertices.copy()
    v = move_points(v, 0 + offset, 1 + offset, r, coef)
    v = move_points(v, 2 + offset, 3 + offset, r, coef)
    v = move_points(v, 1 + offset, 2 + offset, r, coef)
    v = move_points(v, 3 + offset, 4 + offset, r, coef)
    return v


def get_boundary(vertices):
    '''get the tight boundary around given vertices
    '''
    x1, y1, x2, y2, x3, y3, x4, y4 = vertices
    x_min = min(x1, x2, x3, x4)
    x_max = max(x1, x2, x3, x4)
    y_min = min(y1, y2, y3, y4)
    y_max = max(y1, y2, y3, y4)
    return x_min, x_max, y_min, y_max


def cal_error(vertices):
    '''default orientation is x1y1 : left-top, x2y2 : right-top, x3y3 : right-bot, x4y4 : left-bot
    '''
    x_min, x_max, y_min, y_max = get_boundary(vertices)
    x1, y1, x2, y2, x3, y3, x4, y4 = vertices
    err = eu_distance(x1, y1, x_min, y_min) + eu_distance(x2, y2, x_max, y_min) + \
          eu_distance(x3, y3, x_max, y_max) + eu_distance(x4, y4, x_min, y_max)
    return err


def extract_vertices(lines):
    labels = []
    vertices = []
    for line in lines:
        vertices.append(list(map(int,line.rstrip('\n').lstrip('\ufeff').split(',')[:8])))
        label = 0 if '###' in line else 1
        labels.append(label)
    return np.array(vertices), np.array(labels)





