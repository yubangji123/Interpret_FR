import os
import scipy.io as sio
import scipy
import random
import numpy as np
from PIL import Image, ImageDraw

def save_images(images, size, image_path, inverse = True):
    if (inverse):
        images = inverse_transform(images)

    return imsave(images, size, image_path)

def inverse_transform(images):
    return (images+1.)/2.


def imsave(images, size, path):
    img = merge(images, size)

    return scipy.misc.imsave(path, img)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img

lmks = sio.loadmat('landmarks68_IJBA.mat')
lmks = lmks.get('landmarks68')
tVM = sio.loadmat('tVM.mat')
tVM = tVM.get('tVM')
lm_frontal = sio.loadmat('landmark68_7.mat')
lm_frontal = lm_frontal.get('landmark68_7')

fd = open('../../DATA/IJBA_recrop_images_96_96_test.dat')
images = np.fromfile(file=fd,dtype=np.uint8)
fd.close()
images = images.reshape((-1,96,96,3)).astype(np.float32)
images = images/127.5 - 1.

for count, face in enumerate(images):
    face = face.reshape((-1, 96, 96, 3)).astype(np.float32)

    w = 32
    h = 12
    batch_shape = face.shape
    im_w = 110
    im_h = 110
    im_c = 3
    batch_images_occl = []

    left_corner_x = random.sample([ii for ii in range(10, im_w - w - 9)], 1)
    left_corner_y = random.sample([ii for ii in range(10, im_h - h - 9)], 1)
    left_corner_x = left_corner_x[0]
    left_corner_y = left_corner_y[0]
    if left_corner_x == 10:
        left_corner_x += 1
    if left_corner_x == im_w - w - 10:
        left_corner_x -= 1

    if left_corner_y == 10:
        left_corner_y += 1
    if left_corner_y == im_h - h - 10:
        left_corner_y -= 1

    ps = [[left_corner_x, left_corner_y], [left_corner_x, left_corner_y + h], [left_corner_x + w, left_corner_y + h],
          [left_corner_x + w, left_corner_y]]
    ps_1 = []
    for j in range(0, len(ps)):
        for i in range(0, len(tVM)):
            t_v = np.where(tVM[i, :])
            xl = lm_frontal[0, t_v] + 7
            yl = lm_frontal[1, t_v] + 7
            xl = xl[0]
            yl = yl[0]

            x = ps[j][0]
            y = ps[j][1]

            alpha = ((x - xl[2]) * (yl[1] - yl[2]) - (y - yl[2]) * (xl[1] - xl[2])) / (
                    (xl[0] - xl[2]) * (yl[1] - yl[2]) - (yl[0] - yl[2]) * (xl[1] - xl[2]))

            beta = ((x - xl[2]) * (yl[0] - yl[2]) - (y - yl[2]) * (xl[0] - xl[2])) / (
                    (xl[1] - xl[2]) * (yl[0] - yl[2]) - (yl[1] - yl[2]) * (xl[0] - xl[2]))

            gamma = 1 - alpha - beta

            if (alpha >= 0) and (alpha <= 1) and (beta >= 0) and (beta <= 1) and ((alpha + beta) <= 1):
                x_new = np.sum(lmks[count][0][0, t_v[0]] * [alpha, beta, gamma])
                y_new = np.sum(lmks[count][0][1, t_v[0]] * [alpha, beta, gamma])
                x_new = x_new - 7
                y_new = y_new - 7
                ps_1.append([x_new, y_new])

    polygon = [(ps_1[0][0], ps_1[0][1]), (ps_1[1][0], ps_1[1][1]), (ps_1[2][0], ps_1[2][1]),
               (ps_1[3][0], ps_1[3][1])]
    img = Image.new('L', (96, 96), 0)
    ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
    mask = np.array(img)
    masks = np.zeros([96, 96, 3], dtype=np.bool)
    masks[:, :, 0] = mask
    masks[:, :, 1] = mask
    masks[:, :, 2] = mask
    batch_image_occl = np.copy(face[0, :, :, :])
    batch_image_occl[masks] = -1

    if not os.path.exists('IJB-A_occl'):
        os.mkdir('IJB-A_occl')
    save_images(np.reshape(batch_image_occl, newshape=[1, 96, 96, 3]), size=[1, 1], image_path="IJB-A_occl/%d.png" % (count), inverse=True)