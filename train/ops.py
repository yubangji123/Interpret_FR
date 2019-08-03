import math
import numpy as np
import scipy.io as sio
import tensorflow as tf
from PIL import Image, ImageDraw

from tensorflow.python.framework import ops

from utils import *

class batch_norm(object):
    """Code modification of http://stackoverflow.com/a/33950177"""
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
          self.epsilon  = epsilon
          self.momentum = momentum
          self.name = name

    def __call__(self, x, train=True, reuse=False ):
        return tf.contrib.layers.batch_norm(x,
                      decay=self.momentum, 
                      updates_collections=None,
                      epsilon=self.epsilon,
                      scale=True,
                      reuse = reuse,
                      is_training=train,
                      scope=self.name)

def binary_cross_entropy(preds, targets, name=None):
    """Computes binary cross entropy given `preds`.

    For brevity, let `x = `, `z = targets`.  The logistic loss is

        loss(x, z) = - sum_i (x[i] * log(z[i]) + (1 - x[i]) * log(1 - z[i]))

    Args:
        preds: A `Tensor` of type `float32` or `float64`.
        targets: A `Tensor` of the same type and shape as `preds`.
    """
    eps = 1e-12
    with ops.op_scope([preds, targets], name, "bce_loss") as name:
        preds = ops.convert_to_tensor(preds, name="preds")
        targets = ops.convert_to_tensor(targets, name="targets")
        return tf.reduce_mean(-(targets * tf.log(preds + eps) +
                              (1. - targets) * tf.log(1. - preds + eps)))

def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat(axis=3, values=[x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])])

def conv2d(input_, output_dim, 
           k_h=3, k_w=3, d_h=2, d_w=2, stddev=0.02,
           name="conv2d", reuse = False):
    with tf.variable_scope(name, reuse = reuse):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv

def deconv2d(input_, output_shape,
             k_h=3, k_w=3, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d", with_w=False, reuse = False):
    with tf.variable_scope(name, reuse = reuse):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_h, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        
        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                strides=[1, d_h, d_w, 1])

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                                strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv

def maxpool2d(x, k=2, padding='VALID'):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding=padding)

       
def prelu(x, name, reuse = False):
    shape = x.get_shape().as_list()[-1:]

    with tf.variable_scope(name, reuse = reuse):
        alphas = tf.get_variable('alpha', shape, tf.float32,
                            initializer=tf.constant_initializer(value=0.2))

        return tf.nn.relu(x) + tf.multiply(alphas, (x - tf.abs(x))) * 0.5

def relu(x, name='relu'):
    return tf.nn.relu(x, name)  

def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak*x)

def elu(x, name='elu'):
  return tf.nn.elu(x, name)

def linear(input_, output_size, scope="Linear", reuse = False, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear", reuse = reuse):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias

def triplet_loss(anchor_output, positive_output, negative_output, margin = 0.2 ):
    d_pos = tf.reduce_sum(tf.square(anchor_output - positive_output), 1)
    d_neg = tf.reduce_sum(tf.square(anchor_output - negative_output), 1)

    loss = tf.maximum(0., margin + d_pos - d_neg)
    
    return loss

def cosine_loss(anchor_output, positive_output):
    anchor_output_norm = tf.nn.l2_normalize(anchor_output, 1)
    positive_output_norm = tf.nn.l2_normalize(positive_output, 1)
    loss = 1 - tf.reduce_sum(tf.multiply(anchor_output_norm, positive_output_norm), 1)

    return loss

def cosine_triplet_loss(anchor_output, positive_output, negative_output, margin = 0.2 ):
    anchor_output_norm = tf.nn.l2_normalize(anchor_output, 1)
    positive_output_norm = tf.nn.l2_normalize(positive_output, 1)
    negative_output_norm = tf.nn.l2_normalize(negative_output, 1)

    sim_pos = tf.reduce_sum(tf.multiply(anchor_output_norm, positive_output_norm), 1)
    sim_neg = tf.reduce_sum(tf.multiply(anchor_output_norm, negative_output_norm), 1)

    loss = tf.maximum(0., margin - sim_pos + sim_neg)

    return loss

def splineInterpolation(x, x1, matrix):

    N  = int(x.get_shape()[0])
    N1 = int(x1.get_shape()[0])

    distance = tf.square(tf.tile(tf.reshape(x, shape = [N,1, 2]), [1, N1, 1]) - tf.tile(tf.reshape(x1, shape = [1,N1, 2]), [N, 1, 1]))
    distance = tf.sqrt(tf.reduce_sum(distance, axis = 2))

    A = distance
    B = tf.concat(axis=1, values=[tf.ones([x.get_shape()[0], 1], tf.float32), x ])

    return tf.matmul(tf.concat(axis=1, values=[A, B]), matrix)

def bilinear2D(Q, x, y):

    x = tf.clip_by_value(x, clip_value_min = 1, clip_value_max = 126) 
    y = tf.clip_by_value(y, clip_value_min = 1, clip_value_max = 126) 

    x1 = tf.floor(x) 
    x2 = x1 + 1

    y1 = tf.floor(y)
    y2 = y1 + 1

    #i = tf.reshape( tf.concat(1, [x2-x, x-x1]), [-1,1,1,2] )

    k = int(Q.get_shape()[2]) # Number of channels
    q11 = tf.reshape( tf.gather_nd( Q, tf.to_int32(tf.concat(axis=1, values=[x1, y1 ])) ), shape = [-1,k,1,1] ) 
    q12 = tf.reshape( tf.gather_nd( Q, tf.to_int32(tf.concat(axis=1, values=[x1, y2 ])) ), shape = [-1,k,1,1] )
    q21 = tf.reshape( tf.gather_nd( Q, tf.to_int32(tf.concat(axis=1, values=[x2, y1 ])) ), shape = [-1,k,1,1] )
    q22 = tf.reshape( tf.gather_nd( Q, tf.to_int32(tf.concat(axis=1, values=[x2, y2 ])) ), shape = [-1,k,1,1] )

    q = tf.concat( axis=2, values=[  tf.concat(axis=3, values=[q11, q12]) ,   tf.concat(axis=3, values=[q21, q22])   ] )



    #print("q11")
    #print(q.get_shape())


    xx =  tf.tile(   tf.reshape( tf.concat(axis=1, values=[x2-x, x-x1]), shape = [-1,1,1,2] ), multiples = [1,k,1,1])
    yy =  tf.tile(   tf.reshape( tf.concat(axis=1, values=[y2-y, y-y1]), shape = [-1,1,2,1] ), multiples = [1,k,1,1])

    Q_new = tf.matmul(tf.matmul(xx, q), yy)

    #print("Q_new")
    #print(Q_new.get_shape())

    return Q_new
def diversityLossB(input_, k_h=3, k_w=3, d_h=2, d_w=2):

    print(input_.get_shape().as_list())
    i_s = input_.get_shape().as_list()
    g_k_array = np.zeros([3, 3, i_s[3], i_s[3]])

    for i in range(0,i_s[3]):
        g_k_array[:,:,i,i] = [[0.0947416, 0.118318, 0.0947416], [0.118318, 0.147761, 0.118318], [0.0947416, 0.118318, 0.0947416]]

    print(g_k_array.shape)
    guasian_k = tf.constant(g_k_array, dtype=tf.float32, shape=g_k_array.shape)

    conv = tf.nn.conv2d(input_, guasian_k, strides=[1, d_h, d_w, 1], padding='SAME')

    c_s = input_.get_shape().as_list()
    diver_loss = 0
    batch_array = tf.split(conv, num_or_size_splits=c_s[0], axis=0)
    for i in range(0,c_s[0]):
        batch_tmp = tf.reshape(batch_array[i], [c_s[1]*c_s[2], c_s[3]])
        batch_tmp_tr = tf.transpose(batch_tmp, [1,0])
        m_1 = tf.constant(1, dtype=tf.float32, shape=[c_s[1]*c_s[2], 1])
        m_1_tr = tf.transpose(m_1, [1,0])
        m_mul = tf.matmul(batch_tmp_tr, batch_tmp)
        m_mul = m_mul-tf.diag(tf.diag_part(m_mul))
        #mini_value = tf.constant(0.000001, dtype=tf.float32, shape=[1])
        #m_loss = tf.pow(tf.div(m_mul, tf.add(tf.matmul(tf.sqrt(tf.matmul(tf.pow(batch_tmp_tr,2), m_1)), tf.sqrt(tf.matmul(m_1_tr, tf.pow(batch_tmp,2)))),mini_value)),2)
        m_loss = tf.pow(tf.div(m_mul, tf.matmul(tf.sqrt(tf.matmul(tf.pow(batch_tmp_tr, 2), m_1)),
                                                       tf.sqrt(tf.matmul(m_1_tr, tf.pow(batch_tmp, 2))))),2)
        m_1 = tf.constant(1, dtype=tf.float32, shape=[c_s[3], 1])
        m_1_tr = tf.transpose(m_1, [1,0])
        div_l_real = (tf.matmul(tf.matmul(m_1_tr, m_loss),m_1))/2
        diver_loss = diver_loss + div_l_real/((c_s[3]*c_s[3]-c_s[3])/2)
    diver_loss = diver_loss/c_s[0]

    #conv_transp = tf.transpose(conv)
    #diver_loss = tf.tensordot(conv,conv_transp, axes=4)/(tf.sqrt(tf.tensordot(conv, conv, axes=4))*tf.sqrt(tf.tensordot(conv_transp, conv_transp, axes=4)))

    return diver_loss


def occludedFaces(batch_images, tVM, lm_frontal, lmks_batch, crop_h, crop_w, w=32, h=12):

    batch_shape = batch_images.shape
    im_w = 110
    im_h = 110
    im_c = 3
    batch_images_occl = []

    '''w = random.sample([ii for ii in range(16, 32)], 1)
    h = random.sample([ii for ii in range(16, 32)], 1)
    w = w[0]
    h = h[0]'''

    left_corner_x = random.sample([ii for ii in range(10, im_w-w-9)], 1)
    left_corner_y = random.sample([ii for ii in range(10, im_h-h-9)], 1)
    left_corner_x = left_corner_x[0]
    left_corner_y = left_corner_y[0]
    if left_corner_x == 10:
        left_corner_x += 1
    if left_corner_x == im_w-w-10:
        left_corner_x -= 1

    if left_corner_y == 10:
        left_corner_y += 1
    if left_corner_y == im_h-h-10:
        left_corner_y -= 1

    '''left_corner_x = 37c
    left_corner_y = 62'''
    ps = [[left_corner_x, left_corner_y], [left_corner_x, left_corner_y + h], [left_corner_x + w, left_corner_y + h], [left_corner_x + w, left_corner_y]]
    for i_b in range(0, len(batch_images)):
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

                if (alpha>=0) and (alpha<=1) and (beta>=0) and (beta<=1) and ((alpha+beta)<=1):
                    x_new = np.sum(lmks_batch[i_b][0][0, t_v[0]] * [alpha, beta, gamma])
                    y_new = np.sum(lmks_batch[i_b][0][1, t_v[0]] * [alpha, beta, gamma])
                    x_new = x_new - crop_w[i_b]
                    y_new = y_new - crop_h[i_b]
                    ps_1.append([x_new, y_new])

        polygon = [(ps_1[0][0], ps_1[0][1]), (ps_1[1][0], ps_1[1][1]), (ps_1[2][0], ps_1[2][1]), (ps_1[3][0], ps_1[3][1])]
        img = Image.new('L', (96, 96), 0)
        ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
        mask = np.array(img)
        masks = np.zeros([96, 96, 3], dtype=np.bool)
        masks[:, :, 0] = mask
        masks[:, :, 1] = mask
        masks[:, :, 2] = mask
        batch_image_occl = np.copy(batch_images[i_b, :, :, :])
        batch_image_occl[masks] = -1

        batch_images_occl.append(batch_image_occl)


        #save_images(np.reshape(batch_images[i_b, :, :, :], newshape=[1, 96, 96, 3]), size=[1, 1], image_path="img_test.png", inverse=True)
        #save_images(np.reshape(batch_image_occl, newshape=[1, 96, 96, 3]), size=[1, 1], image_path="img_o_test.png", inverse=True)

    return batch_images_occl