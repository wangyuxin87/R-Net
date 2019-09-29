import numpy as np
from PIL import Image
from shapely.geometry import Polygon

def height_adjusted(img, vertices, ratio=0.2):
    '''adjust height of image to augmented data
    '''


    h_ratio = 1 + ratio * (np.random.rand() * 2 - 1)
    old_h = img.height
    new_h = int(np.around(old_h * h_ratio))
    img = img.resize((img.width, new_h), Image.BILINEAR)

    new_vertices = vertices.copy()
    if vertices.size > 0:
        new_vertices[:, [1, 3, 5, 7]] = vertices[:, [1, 3, 5, 7]] * (new_h / old_h)
    return img, new_vertices

def crop_img(img, vertices, labels, length):
    '''crop img patches to obtain batch and augment
    '''
    w, h = img.width, img.height

    if h >= w and w < length:
        img = img.resize((length, int(h * length / w)), Image.BILINEAR)
    elif h < w and h < length:
        img = img.resize((int(w * length / h), length), Image.BILINEAR)

    ratio_w = img.width / w
    ratio_h = img.height / h
    r_h = img.height - length
    r_w = img.width - length
    flag = True
    cnt = 0

    assert (ratio_w >= 1 and ratio_h >= 1)

    new_vertices = np.zeros(vertices.shape)
    if vertices.size > 0:
        new_vertices[:, [0, 2, 4, 6]] = vertices[:, [0, 2, 4, 6]] * ratio_w
        new_vertices[:, [1, 3, 5, 7]] = vertices[:, [1, 3, 5, 7]] * ratio_h

    while flag and cnt < 1000:
        cnt += 1
        w = int(np.random.rand() * r_w)
        h = int(np.random.rand() * r_h)
        flag = is_cross_text([w, h], length, new_vertices[labels == 1, :])

    box = (w, h, w + length, h + length)
    region = img.crop(box)
    if new_vertices.size == 0:
        return region, new_vertices

    new_vertices[:, [0, 2, 4, 6]] -= w
    new_vertices[:, [1, 3, 5, 7]] -= h
    return region, new_vertices


def is_cross_text(lt, length, vertices):
    '''check if the crop image crosses text regions
    '''

    if vertices.size == 0:
        return False
    w, h = lt
    a = np.array([w, h, w + length, h, w + length, h + length, w, h + length]).reshape((4, 2))

    p1 = Polygon(a).convex_hull
    for vertice in vertices:
        p2 = Polygon(vertice.reshape((4, 2))).convex_hull
        inter = p1.intersection(p2).area
        if 0.01 <= inter / p2.area <= 0.99:
            return True
    return False
