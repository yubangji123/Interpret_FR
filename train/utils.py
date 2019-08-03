"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import csv
import json
import random
import pprint
import scipy.misc
import numpy as np
import gc
from glob import glob
import os
#import matplotlib.pyplot as plt
from time import gmtime, strftime



pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

def get_image(image_path, image_size, is_crop=True, is_random_crop = False, resize_w=64, is_grayscale = False):
    return transform(imread(image_path, is_grayscale), image_size, is_crop, is_random_crop, resize_w)

def save_images(images, size, image_path, inverse = True):
    if (inverse):
        images = inverse_transform(images)

    return imsave(images, size, image_path)

def imread(path, is_grayscale = False):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten = True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)

def merge_images(images, size):
    return inverse_transform(images)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img

def imsave(images, size, path):
    img = merge(images, size)

    #plt.imshow(img)
    #plt.show()
    
    return scipy.misc.imsave(path, img)

def center_crop(x, crop_h, crop_w=None, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w],
                               [resize_w, resize_w])

def random_crop(x, crop_h, crop_w=None, with_crop_size=None ):
    if crop_w is None:
        crop_w = crop_h
    if with_crop_size is None:
        with_crop_size = False
    h, w = x.shape[:2]

    j = random.randint(0, h - crop_h)
    i = random.randint(0, w - crop_w)

    if with_crop_size:
        return x[j:j+crop_h, i:i+crop_w,:], j, i
    else:
        return x[j:j+crop_h, i:i+crop_w,:]

def crop(x, crop_h, crop_w, j, i):
    if crop_w is None:
        crop_w = crop_h
    
    return x[j:j+crop_h, i:i+crop_w]


    #return scipy.misc.imresize(x, [96, 96] )

def transform(image, npx=64, is_crop=True, is_random_crop=True, resize_w=64):
    # npx : # of pixels width/height of image
    if is_crop:
        if is_random_crop:
            cropped_image = random_crop(image, npx)
        else:  
            cropped_image = center_crop(image, npx, resize_w=resize_w)
    else:
        cropped_image = image
    return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
    return (images+1.)/2.


def to_json(output_path, *layers):
    with open(output_path, "w") as layer_f:
        lines = ""
        for w, b, bn in layers:
            layer_idx = w.name.split('/')[0].split('h')[1]

            B = b.eval()

            if "lin/" in w.name:
                W = w.eval()
                depth = W.shape[1]
            else:
                W = np.rollaxis(w.eval(), 2, 0)
                depth = W.shape[0]

            biases = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(B)]}
            if bn != None:
                gamma = bn.gamma.eval()
                beta = bn.beta.eval()

                gamma = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(gamma)]}
                beta = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(beta)]}
            else:
                gamma = {"sy": 1, "sx": 1, "depth": 0, "w": []}
                beta = {"sy": 1, "sx": 1, "depth": 0, "w": []}

            if "lin/" in w.name:
                fs = []
                for w in W.T:
                    fs.append({"sy": 1, "sx": 1, "depth": W.shape[0], "w": ['%.2f' % elem for elem in list(w)]})

                lines += """
                    var layer_%s = {
                        "layer_type": "fc", 
                        "sy": 1, "sx": 1, 
                        "out_sx": 1, "out_sy": 1,
                        "stride": 1, "pad": 0,
                        "out_depth": %s, "in_depth": %s,
                        "biases": %s,
                        "gamma": %s,
                        "beta": %s,
                        "filters": %s
                    };""" % (layer_idx.split('_')[0], W.shape[1], W.shape[0], biases, gamma, beta, fs)
            else:
                fs = []
                for w_ in W:
                    fs.append({"sy": 5, "sx": 5, "depth": W.shape[3], "w": ['%.2f' % elem for elem in list(w_.flatten())]})

                lines += """
                    var layer_%s = {
                        "layer_type": "deconv", 
                        "sy": 5, "sx": 5,
                        "out_sx": %s, "out_sy": %s,
                        "stride": 2, "pad": 1,
                        "out_depth": %s, "in_depth": %s,
                        "biases": %s,
                        "gamma": %s,
                        "beta": %s,
                        "filters": %s
                    };""" % (layer_idx, 2**(int(layer_idx)+2), 2**(int(layer_idx)+2),
                             W.shape[0], W.shape[3], biases, gamma, beta, fs)
        layer_f.write(" ".join(lines.replace("'","").split()))

def load_CASIA(isDev = False):
    print('Loading CASIA ...')
    fileList = open('../DATA/CASIA_recrop_fileList.dat', 'r')
    reader = csv.reader(fileList)
    filename = []
    pid = []
    yaw = []
    for row in reader:
        name = row[0]
        filename.append(name)
        pid.append(int(row[2]))
        yaw.append(float(row[3]))
    fileList.close()
    print(len(yaw))
    if isDev:
        all_images = np.zeros((len(pid), 110, 110, 3), np.uint8)
    else:
        fd = open('../DATA/CASIA_all_images_110_110.dat')
        size_img = 110 * 110 * 3
        trlen = int(len(yaw) / 1)
        filename = filename[0:trlen]
        pid = pid[0:trlen]
        yaw = yaw[0:trlen]
        all_images_sp = np.fromfile(file=fd, dtype=np.uint8, count=size_img * trlen)
        all_images_sp_2 = all_images_sp.reshape((-1, 110, 110, 3)).astype(np.uint8)
        del all_images_sp
        gc.collect()
        all_images = all_images_sp_2
        del all_images_sp_2
        gc.collect()
        fd.close()

    assert (all_images.shape[0] == len(pid)), "Number of samples must be the same"
    print('Finish loading CASIA.')
    return all_images, filename, pid, yaw

def load_IJBA_recrop_test(isFlip = False):
    print('Loading IJBA recrop...')

    if (isFlip):
        fd = open('../DATA/IJBA_recrop_images_96_96_test.dat')
    else:
        fd = open('../DATA/IJBA_recrop_images_96_96_test.dat')
    images = np.fromfile(file=fd,dtype=np.uint8)
    fd.close()

    images = images.reshape((-1,96,96,3)).astype(np.float32)
    images = images/127.5 - 1.
    print('    DONE. Finish loading IJBA recrop with ' + str(images.shape[0]) + ' images')

    return images