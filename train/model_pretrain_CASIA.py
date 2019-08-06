from __future__ import division
import os
import time
import csv
import random
from random import randint
from math import floor
from glob import glob
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import scipy.io as sio
from six.moves import xrange
from ops import *
from utils import *


SUBJECT_NUM_MTPIE = 200
SUBJECT_NUM_MTPIE_FULL = 347
SUBJECT_NUM_CASIA = 10575


class InterpFR(object):
    def __init__(self, sess, image_size=108, batch_size=64, sample_size = 64, output_size=64,
                 y_dim=None, z_dim=100, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default',
                 checkpoint_dir='checkpoint', samples_dir='samples', devices=None):
        """
        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            output_size: (optional) The resolution in pixels of the images. [64]
            y_dim: (optional) Dimension of dim for y. [None]
            z_dim: (optional) Dimension of dim for Z. [100]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
            dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
            c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
        """
        self.sess = sess
        self.is_grayscale = (c_dim == 1)
        self.batch_size = batch_size
        self.image_size = image_size
        self.sample_size = sample_size
        self.output_size = output_size
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim
        self.c_dim = c_dim
        
        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn0_0 = batch_norm(name='d_k_bn0_0')
        self.d_bn0_1 = batch_norm(name='d_k_bn0_1')
        self.d_bn0_2 = batch_norm(name='d_k_bn0_2')
        self.d_bn1_0 = batch_norm(name='d_k_bn1_0')
        self.d_bn1_1 = batch_norm(name='d_k_bn1_1')
        self.d_bn1_2 = batch_norm(name='d_k_bn1_2')
        self.d_bn1_3 = batch_norm(name='d_k_bn1_3')
        self.d_bn2_0 = batch_norm(name='d_k_bn2_0')
        self.d_bn2_1 = batch_norm(name='d_k_bn2_1')
        self.d_bn2_2 = batch_norm(name='d_k_bn2_2')
        self.d_bn3_0 = batch_norm(name='d_k_bn3_0')
        self.d_bn3_1 = batch_norm(name='d_k_bn3_1')
        self.d_bn3_2 = batch_norm(name='d_k_bn3_2')
        self.d_bn3_3 = batch_norm(name='d_k_bn3_3')
        self.d_bn4_0 = batch_norm(name='d_k_bn4_0')
        self.d_bn4_1 = batch_norm(name='d_k_bn4_1')
        self.d_bn4_2 = batch_norm(name='d_k_bn4_2')
        self.d_bn5   = batch_norm(name='d_k_bn5')
        self.d_PCA3_2 = batch_norm(name='d_PCA3_2_new')
        self.d_PCA4_2 = batch_norm(name='d_PCA4_2_new')
        self.d_k5 = batch_norm(name='d_k5_new')

        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.samples_dir = samples_dir
        model_dir = "%s_%s_%s_%s_%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size, self.gf_dim, self.gfc_dim, self.df_dim, self.dfc_dim)

        if not os.path.exists(self.samples_dir+"/"+model_dir):
            os.makedirs(self.samples_dir+"/"+model_dir)
        if not os.path.exists(self.checkpoint_dir+"/"+model_dir):
            os.makedirs(self.checkpoint_dir+"/"+model_dir)
        self.devices = devices
        self.build_model()
    def build_model(self):
        if self.y_dim:
            if self.dataset_name == 'MTPIE' or self.dataset_name == 'CASIA' or self.dataset_name == 'CASIA_MTPIE':
                if self.dataset_name == 'MTPIE':
                    self.subject_num = SUBJECT_NUM_MTPIE
                elif self.dataset_name == 'CASIA':
                    self.subject_num = SUBJECT_NUM_CASIA
                else:
                    self.subject_num = SUBJECT_NUM_CASIA #+ SUBJECT_NUM_MTPIE_FULL
                    
                self.onehot_labels= tf.placeholder(tf.float32, [self.batch_size, self.subject_num], name='positive_hot_code_labels')     
                self.input_images = tf.placeholder(tf.float32, [self.batch_size] + [self.output_size, self.output_size, self.c_dim], name='input_images')
                self.input_images_occl = tf.placeholder(tf.float32, [self.batch_size] + [self.output_size, self.output_size, self.c_dim], name='input_images_occl')
                
            else:
                self.y= tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')
        

        self.sample_images= tf.placeholder(tf.float32, [self.sample_size] + [self.output_size, self.output_size, self.c_dim], name='sample_images')
        self.sample_input_images = tf.placeholder(tf.float32, [1, self.output_size, self.output_size, self.c_dim], name='sample_input_images')

        # Networks
        self.D_R_id_logits, self.k6, self.k5, self.k6_org =  self.discriminator(self.input_images, is_reuse=False) #_, self.D_R_log
        _, _, self.k5_o, self.k6_org_o = self.discriminator(self.input_images_occl, is_reuse=True)
        self.sampler_feature, self.sampler_k5 = self.sampler(self.sample_input_images)

        # Diversity Loss
        self.diver_loss_A = 100 * self.diversityLossA()
        self.diver_loss = diversityLossB(self.k5, d_w=1, d_h=1)
        self.diver_loss_o = diversityLossB(self.k5_o, d_w=1, d_h=1)

        # Occlusion Loss
        self.k6_changed = tf.abs(tf.subtract(self.k6_org, self.k6_org_o))
        self.k6_changed = tf.reduce_mean(self.k6_changed, axis=0)
        self.k6_changed_tpk, _ = tf.nn.top_k(tf.negative(self.k6_changed), 260)
        self.occl_loss = tf.reduce_sum(tf.negative(self.k6_changed_tpk)) * 0.1

        k6_changed_tpk_splits = tf.split(self.k6_changed_tpk, 260)
        self.k6_changed_bin_occl = tf.subtract(1.0, tf.cast(
            tf.greater(self.k6_changed, tf.negative(k6_changed_tpk_splits[259])), dtype=tf.float32))
        self.k6_org_o = tf.multiply(self.k6_org_o, self.k6_changed_bin_occl)
        self.k6_occl = tf.nn.dropout(self.k6_org_o, keep_prob=0.6)
        self.D_R_id_logits_occl = linear(self.k6_occl, self.subject_num, 'd_k7_id_lin_new', reuse=True)

        # l2 Regularization
        self.l2_regu_id_logits = tf.nn.l2_loss(self.D_R_id_logits) * 0.1
        self.l2_regu_id_logits_occl = tf.nn.l2_loss(self.D_R_id_logits_occl) * 0.1

        # D Loss
        self.d_loss_real_id = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.D_R_id_logits, labels=self.onehot_labels)) * 2
        self.d_loss_real_id_occl = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.D_R_id_logits_occl, labels=self.onehot_labels)) * 2

        self.d_loss =  self.d_loss_real_id + self.d_loss_real_id_occl + self.diver_loss_A + self.diver_loss + self.diver_loss_o + self.occl_loss # Total loss (can add more term here)

        self.d_acc = slim.metrics.accuracy(tf.argmax(self.D_R_id_logits,1), tf.argmax(self.onehot_labels,1), weights=100.0)
        self.d_acc_occl = slim.metrics.accuracy(tf.argmax(self.D_R_id_logits_occl, 1), tf.argmax(self.onehot_labels, 1), weights=100.0)
  
        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'd_' in var.name]

        self.d_vars_new = [var for var in t_vars if 'new' in var.name]
        self.d_vars_old = [var for var in t_vars if 'new' not in var.name]

        for var in self.d_vars:
            print(var.op.name)

        self.saver_old = tf.train.Saver(self.d_vars_old, keep_checkpoint_every_n_hours=.5, max_to_keep = 20)
        self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=.5, max_to_keep = 20)

        
    def cos_loop(self,matrix, vector):
        """
        Calculating pairwise cosine distance using a common for loop with manually calculated cosine value.
        """
        neighbors = []
        for row in range(matrix.shape[0]):
            vector_norm = np.linalg.norm(vector)
            row_norm = np.linalg.norm(matrix[row,:])
            cos_val = vector.dot(matrix[row,:]) / (vector_norm * row_norm)
            neighbors.append(cos_val)
        return neighbors
             
                
    def train(self, config):

        # Read data
        images, files, pid_CASIA, yaw_CASIA = load_CASIA()

        lmks = sio.loadmat('landmarks68.mat')
        lmks = lmks.get('landmarks68')
        tVM = sio.loadmat('tVM.mat')
        tVM = tVM.get('tVM')
        lm_frontal = sio.loadmat('landmark68_7.mat')
        lm_frontal = lm_frontal.get('landmark68_7')

        num_CASIA = len(pid_CASIA)
        valid_idx = np.random.permutation(num_CASIA)

        hot_code_id = np.zeros((num_CASIA, self.subject_num), dtype=np.float16)
        for i, label in enumerate(pid_CASIA):
            hot_code_id[i, int(pid_CASIA[i])] = 1.0

        # np.random.shuffle(data)
        d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.d_loss,
                                                                                            var_list=self.d_vars)
        tf.global_variables_initializer().run()
        #self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=.5, max_to_keep=20)

        """Training """
        # Load model if available
        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=.5, max_to_keep=20)
        
    
        counter = 0
        start_time = time.time()
        
        for epoch in xrange(config.epoch):
            batch_idxs = min(len(valid_idx), config.train_size) // config.batch_size
            
            for idx in xrange(0, batch_idxs):


                # Get training batch
                batch_files = valid_idx[idx*config.batch_size:(idx+1)*config.batch_size]
                batch = []
                row_j = []
                col_i = []
                for batch_file in batch_files:
                    b, j, i = random_crop(images[batch_file,:,:], self.output_size, self.output_size, with_crop_size=True)
                    batch.append(b)
                    row_j.append(j)
                    col_i.append(i)

                if (self.is_grayscale):
                    batch_images = np.array(batch).astype(np.float32)[:, :, :, None]/127.5 - 1.
                else:
                    batch_images = np.array(batch).astype(np.float32)/127.5 - 1.

                batch_images_occl = occludedFaces(batch_images, tVM=tVM, lm_frontal=lm_frontal, lmks_batch=lmks[batch_files], crop_h=row_j, crop_w=col_i)
        
                batch_id_labels = [hot_code_id[batch_file,:] for batch_file in batch_files]

                # Update D network
                self.sess.run(d_optim, feed_dict={ self.input_images: batch_images,
                                                   self.input_images_occl: batch_images_occl,
                                                   self.onehot_labels:batch_id_labels})

                model_dir = "%s_%s_%s_%s_%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size, self.gf_dim, self.gfc_dim, self.df_dim, self.dfc_dim)
                counter += 1

                if np.mod(counter, 25) == 1:
                    errD_real_id, errD_real_id_o, d_acc, d_acc_o, div_loss, div_loss_a, k5_np, o_loss = self.sess.run(
                        [self.d_loss_real_id,
                         self.d_loss_real_id_occl,
                         self.d_acc,
                         self.d_acc_occl,
                         self.diver_loss,
                         self.diver_loss_A,
                         self.k5,
                         self.occl_loss],
                        feed_dict={self.input_images: batch_images,
                                   self.input_images_occl: batch_images_occl,
                                   self.onehot_labels: batch_id_labels})

                    print(
                    "Epoch: [%2d] [%4d/%4d] time: %4.1f, d_loss: %.4f d_loss_o: %.4f div_loss: %.8f div_loss_a:%.8f o_loss:%.8f (id: %.3f (R: %.2f, R_o: %.2f))" \
                    % (epoch, idx, batch_idxs, time.time() - start_time, errD_real_id, errD_real_id_o, div_loss,
                       div_loss_a, o_loss, errD_real_id, d_acc, d_acc_o))

                    k6, k6_o, k6_c, o_loss, l2_r_id_log, l2_r_id_log_o, k6_c_bin_oc, k5_o = self.sess.run(
                        [self.k6_org,
                         self.k6_org_o,
                         self.k6_changed,
                         self.occl_loss,
                         self.l2_regu_id_logits,
                         self.l2_regu_id_logits_occl,
                         self.k6_changed_bin_occl,
                         self.k5_o],
                        feed_dict={self.input_images: batch_images,
                                   self.input_images_occl: batch_images_occl,
                                   self.onehot_labels: batch_id_labels})
                    print(" o_loss:%.8f norm_k6:%.8f norm_k6_o:%.8f l2_r_id_log:%.8f l2_r_id_log_o:%.8f" % \
                          (o_loss, np.linalg.norm(k6), np.linalg.norm(k6_o), l2_r_id_log, l2_r_id_log_o))
                
            # Save model after every epocj
            self.save(config.checkpoint_dir, epoch)
                



    def discriminator(self, image,  is_reuse=False, is_training = True):

        s16 = int(self.output_size/16)
        k0_0 = image
        k0_1 = elu(self.d_bn0_1(conv2d(k0_0, self.df_dim*1, d_h=1, d_w =1, name='d_k01_conv', reuse = is_reuse), train=is_training, reuse = is_reuse), name='d_k01_prelu')
        k0_2 = elu(self.d_bn0_2(conv2d(k0_1, self.df_dim*2, d_h=1, d_w =1, name='d_k02_conv', reuse = is_reuse), train=is_training, reuse = is_reuse), name='d_k02_prelu')
        #k0_3 =               maxpool2d(k0_2, k=2, padding='VALID')
        k1_0 = elu(self.d_bn1_0(conv2d(k0_2, self.df_dim*2, d_h=2, d_w =2, name='d_k10_conv', reuse = is_reuse), train=is_training, reuse = is_reuse), name='d_k10_prelu')
        k1_1 = elu(self.d_bn1_1(conv2d(k1_0, self.df_dim*2, d_h=1, d_w =1, name='d_k11_conv', reuse = is_reuse), train=is_training, reuse = is_reuse), name='d_k11_prelu')
        k1_2 = elu(self.d_bn1_2(conv2d(k1_1, self.df_dim*4, d_h=1, d_w =1, name='d_k12_conv', reuse = is_reuse), train=is_training, reuse = is_reuse), name='d_k12_prelu')
        #k1_3 =               maxpool2d(k1_2, k=2, padding='VALID')
        k2_0 = elu(self.d_bn2_0(conv2d(k1_2, self.df_dim*4, d_h=2, d_w =2, name='d_k20_conv', reuse = is_reuse), train=is_training, reuse = is_reuse), name='d_k20_prelu')
        k2_1 = elu(self.d_bn2_1(conv2d(k2_0, self.df_dim*3, d_h=1, d_w =1, name='d_k21_conv', reuse = is_reuse), train=is_training, reuse = is_reuse), name='d_k21_prelu')
        k2_2 = elu(self.d_bn2_2(conv2d(k2_1, self.df_dim*6, d_h=1, d_w =1, name='d_k22_conv', reuse = is_reuse), train=is_training, reuse = is_reuse), name='d_k22_prelu')
        #k2_3 =               maxpool2d(k2_2, k=2, padding='VALID')
        k3_0 = elu(self.d_bn3_0(conv2d(k2_2, self.df_dim*6, d_h=2, d_w =2, name='d_k30_conv', reuse = is_reuse), train=is_training, reuse = is_reuse), name='d_k30_prelu')
        k3_1 = elu(self.d_bn3_1(conv2d(k3_0, self.df_dim*4, d_h=1, d_w =1, name='d_k31_conv', reuse = is_reuse), train=is_training, reuse = is_reuse), name='d_k31_prelu')
        k3_2 = elu(self.d_bn3_2(conv2d(k3_1, self.df_dim*8, d_h=1, d_w =1, name='d_k32_conv', reuse = is_reuse), train=is_training, reuse = is_reuse), name='d_k32_prelu')
        #k3_3 =               maxpool2d(k3_2, k=2, padding='VALID')
        k4_0 = elu(self.d_bn4_0(conv2d(k3_2, self.df_dim*8, d_h=2, d_w =2, name='d_k40_conv', reuse = is_reuse), train=is_training, reuse = is_reuse), name='d_k40_prelu')
        k4_1 = elu(self.d_bn4_1(conv2d(k4_0, self.df_dim*5, d_h=1, d_w =1, name='d_k41_conv', reuse = is_reuse), train=is_training, reuse = is_reuse), name='d_k41_prelu')
        k4_2 =     self.d_bn4_2(conv2d(k4_1, self.gfc_dim,  d_h=1, d_w =1, name='d_k42_conv', reuse = is_reuse), train=is_training, reuse = is_reuse)

        shape = k2_2.get_shape().as_list()
        up3_2 = tf.image.resize_images(k3_2, [shape[1], shape[2]])
        up4_2 = tf.image.resize_images(k4_2, [shape[1], shape[2]])

        PCA3_2 = tf.nn.l2_normalize(elu(self.d_PCA3_2(conv2d(up3_2, self.df_dim * 6, k_h=1, k_w=1, d_h=1, d_w=1, name='d_pca32_conv_new', reuse = is_reuse), train=is_training, reuse = is_reuse)), dim=3)
        PCA4_2 = tf.nn.l2_normalize(elu(self.d_PCA4_2(conv2d(up4_2, self.df_dim * 6, k_h=1, k_w=1, d_h=1, d_w=1, name='d_pca42_conv_new', reuse = is_reuse), train=is_training, reuse = is_reuse)), dim=3)

        HC = tf.concat([tf.nn.l2_normalize(k2_2,dim=3), PCA3_2, PCA4_2], axis=3)
        print('HC is')
        print(HC.get_shape().as_list())

        k5 = self.d_k5(conv2d(HC, self.dfc_dim,  d_h=1, d_w =1, name='d_k5_conv_new', reuse = is_reuse), train=is_training, reuse = is_reuse)

        print('k5 is')
        print(k5.get_shape().as_list())

        k5_abs = tf.abs(k5)
        k5_sh = k5.get_shape().as_list()
        k5_rs = tf.reshape(k5_abs, [k5_sh[0], k5_sh[1] * k5_sh[2], k5_sh[3]])
        k5_sl = tf.split(k5_rs, num_or_size_splits=k5_sh[0], axis=0)
        k5_k = []
        for i in range(0, k5_sh[0]):
            k5_tpk = tf.transpose(k5_sl[i], perm=[0, 2, 1])
            k5_tpk, _ = tf.nn.top_k(k5_tpk, k5_sh[1] * 1) # change k here, k = 1
            k5_tpk = tf.transpose(k5_tpk, perm=[0, 2, 1])
            k5_tpk = tf.reshape(tf.reduce_min(k5_tpk, axis=1), [k5_sh[3]])
            k5_k.append(k5_tpk)
        k5_k = tf.stack(k5_k)
        k5_tru = tf.greater(k5_abs, tf.expand_dims(tf.expand_dims(k5_k, 1), 1))
        k5_tru = tf.cast(k5_tru, tf.float32)
        k5 = tf.multiply(k5, k5_tru)

        k6 = tf.nn.avg_pool(k5, ksize=[1, k5.shape[1], k5.shape[2], 1], strides = [1,1,1,1],padding = 'VALID')
        print('k6 is')
        print(k6.get_shape().as_list())


        #k5 = tf.nn.avg_pool(k4_2, ksize = [1, s16, s16, 1], strides = [1,1,1,1],padding = 'VALID')
        k6 = tf.reshape(k6, [-1, self.dfc_dim])
        k6_org = k6
        if (is_training):
            k6 = tf.nn.dropout(k6, keep_prob = 0.6)

        k7_id = linear(k6, self.subject_num, 'd_k7_id_lin_new', reuse = is_reuse)


        return k7_id, k6, k5, k6_org

     
            
    def save(self, checkpoint_dir, step):
        model_name = "DCGAN.model"
        model_dir = "%s_%s_%s_%s_%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size, self.gf_dim, self.gfc_dim, self.df_dim, self.dfc_dim)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        model_dir = "%s_%s_%s_%s_%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size, self.gf_dim, self.gfc_dim, self.df_dim, self.dfc_dim)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver_old.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))

            return True
        else:
            return False


    def test_IJBA(self):
        # Single image IJBA
        images = load_IJBA_recrop_test()

        N = images.shape[0]
        gallery_features = np.zeros((N, self.dfc_dim), dtype=np.float32)

        for idx in range(0, N):
            print(idx)
            batch_img = np.array(images[idx, :, :, :]).astype(np.float32).reshape((1, self.output_size, self.output_size, self.c_dim))

            gallery_feature = self.sess.run(self.sampler_feature, feed_dict={self.sample_input_images: batch_img})
            gallery_features[idx, :] = gallery_feature

        np.savetxt("IJBA_features.txt", gallery_features)

        print("Writed to IJBA_features.txt")

    def test_feat_diff_natural_occl(self):
        # Do the feat difference for AR faces
        filename_list = glob('../natural_occl_ar/org/*.bmp')
        filename_list.sort(key=lambda s:int(s[22:25]))
        fm_dim = 320
        ar_k5 = np.zeros((len(filename_list), 24 * 24 * fm_dim), dtype=np.float32)
        ar_feature = np.zeros((len(filename_list), fm_dim), dtype=np.float32)
        for i, filename in enumerate(filename_list):  # assuming gif
            face = scipy.misc.imread(filename)
            face = face.reshape((-1, 96, 96, 3)).astype(np.float32)
            face = face / 127.5 - 1.
            face_feature, face_k5 = self.sess.run([self.sampler_feature, self.sampler_k5], feed_dict={self.sample_input_images: face})
            face_k5 = np.reshape(face_k5, [24 * 24, fm_dim])
            face_k5 = np.reshape(face_k5, [24 * 24 * fm_dim])
            ar_k5[i, :] = face_k5
            face_feature /= np.linalg.norm(face_feature)
            ar_feature[i, :] = face_feature
            print(filename)
        np.savetxt("AR_non_occl_k5.txt", ar_k5)

        for name in ['eyeglasses', 'scarf']:

            ar_feature_o = np.zeros((len(filename_list), fm_dim), dtype=np.float32)
            filename_list = glob('natural_occl_ar/'+name+'/*.bmp')
            filename_list.sort(key=lambda s: int(s[15+len(name)+4:15+len(name)+7]))
            for i, filename in enumerate(filename_list):  # assuming gif
                face_o = scipy.misc.imread(filename)
                face_o = face_o.reshape((-1, 96, 96, 3)).astype(np.float32)
                face_o = face_o / 127.5 - 1.
                face_feature_o = self.sess.run(self.sampler_feature, feed_dict={self.sample_input_images: face_o})
                face_feature_o /= np.linalg.norm(face_feature_o)
                ar_feature_o[i, :] = face_feature_o
                print(filename)

            ind_changed_means = np.mean(np.abs(ar_feature_o - ar_feature), 0)

            np.savetxt("ind_changed_new_div_mean_" + name + ".txt", ind_changed_means)

    def test_feat_diff_synthetic_occl(self):
        face_feat_list = []
        face_list = []
        filename_lists = []
        for filename in glob('../filtered_images/*.png'):  # assuming gif
            face = scipy.misc.imread(filename)
            face = face.reshape((-1, 96, 96, 3)).astype(np.float32)
            face = face / 127.5 - 1.
            face_feature = self.sess.run(self.sampler_feature, feed_dict={self.sample_input_images: face})
            face_feature /= np.linalg.norm(face_feature)
            face_feat_list.append(face_feature)
            face_list.append(face)
            filename_lists.append(filename)
            print(filename)

        for i, j, name in zip([10, 15, 30, 30, 55, 55], [10, 35, 68, 55, 55, 35], ['forehead', 'l_eye', 'mouth', 'nose', 'r_cheek', 'r_eye']):
            w = 32
            h = 12
            feat_dim = 320
            ind_changed_var = []
            ind_changed_means = []
            o_idx = 0
            f_idx = 0
            ind_changed_list = []
            face_list_cp = np.copy(face_list)
            for face_occlude in face_list_cp:
                if o_idx == 0:
                    save_images(face_occlude, size=[1, 1], image_path="./sample_images/img_%d.png" % (f_idx), inverse=True)
                    face_occlude[0, j:(j + h), i:(i + w), :] = -1.
                    face_occlude_feature = self.sess.run(self.sampler_feature,
                                                         feed_dict={self.sample_input_images: face_occlude})
                    face_occlude_feature /= np.linalg.norm(face_occlude_feature)
                if o_idx == 0:
                    save_images(face_occlude, size=[1, 1], image_path="./sample_images/img_o_%d.png" % (f_idx), inverse=True)

                face_feature = face_feat_list[f_idx]
                feature_diff = np.abs(face_feature - face_occlude_feature)
                ind_changed_list.append(feature_diff)
                f_idx += 1

            ind_changed_list = np.reshape(ind_changed_list, newshape=[-1, feat_dim])
            ind_changed_means.append(np.mean(ind_changed_list, axis=0))
            ind_changed_var.append(np.var(ind_changed_list, axis=0))
            print(o_idx)
            o_idx += 1
            ind_changed_means = np.reshape(ind_changed_means, newshape=[-1, feat_dim])
            np.savetxt("ind_changed_new_div_mean_" + name + ".txt", ind_changed_means)

    def test_average_location(self):
        images = load_IJBA_recrop_test()

        N = 1000
        fm_dim = 320
        gallery_features = np.zeros((N, 24 * 24 * fm_dim), dtype=np.float32)

        idxes = range(0, N)

        ii = 0
        for co in range(0, N):
            idx = idxes[co]

            print(idx)

            if (co > len(idxes)):
                idx = input("Input")

            batch_img = np.array(images[idx, :, :, :]).astype(np.float32).reshape(
                (1, self.output_size, self.output_size, self.c_dim))
            batch_img = np.array(batch_img).reshape(1, 96, 96, 3)
            for i in range(0, 1):
                gallery_feature = self.sess.run(self.sampler_k5, feed_dict={self.sample_input_images: batch_img})

                gallery_feature = np.reshape(gallery_feature, [24 * 24, fm_dim])
                gallery_feature = np.reshape(gallery_feature, [24 * 24 * fm_dim])

                gallery_features[ii, :] = gallery_feature

                ii = ii + 1

        # checkpoint_dir = os.path.join(self.checkpoint_dir, self.model_dir)
        np.savetxt("IJBA_feature_maps.txt", gallery_features)

        print("Writed to IJBA_feature_maps.txt")

    def sampler(self, img):

        k7_id, k6, k5,_  = self.discriminator(img, is_reuse=True, is_training=False)

        return k6, k5

    def diversityLossA(self):
        diver_loss = 0
        with tf.variable_scope('d_k5_conv_new', reuse=True):
            w = tf.get_variable('w')
            w_s = w.get_shape().as_list()
            w = tf.reshape(w, [w_s[0]*w_s[1], w_s[2], w_s[3]])
            filter_array = tf.split(w, num_or_size_splits=w_s[3], axis=2)
            for i in range(0, w_s[3]):
                filter_array[i] = tf.reshape(filter_array[i], [w_s[0]*w_s[1], w_s[2]])
                #print filter_array[i].get_shape().as_list()
                filter_array_tr = tf.transpose(filter_array[i], [1, 0])
                #print filter_array_tr.get_shape().as_list()
                m_mul = tf.matmul(filter_array[i], filter_array_tr)
                m_mul = m_mul - tf.diag(tf.diag_part(m_mul))
                m_1 = tf.constant(1, dtype=tf.float32, shape=[w_s[2], 1])
                #print m_1.get_shape().as_list()
                m_1_tr = tf.transpose(m_1, [1, 0])
                #print m_1_tr.get_shape().as_list()
                #mini_value = tf.constant(0.000001, dtype=tf.float32, shape=[1])
                m_loss = tf.pow(tf.div(m_mul, tf.matmul(tf.sqrt(tf.matmul(tf.pow(filter_array[i], 2), m_1)),tf.sqrt(tf.matmul(m_1_tr, tf.pow(filter_array_tr, 2))))), 2)
                #m_loss = tf.pow(tf.div(m_mul, tf.matmul(tf.sqrt(tf.matmul(tf.pow(filter_array[i], 2), m_1)),
                                            #tf.sqrt(tf.matmul(m_1_tr, tf.pow(filter_array_tr, 2))))), 2)
                m_1 = tf.constant(1, dtype=tf.float32, shape=[w_s[0]*w_s[1], 1])
                m_1_tr = tf.transpose(m_1, [1, 0])
                div_l_real = (tf.matmul(tf.matmul(m_1_tr, m_loss), m_1)) / 2
                diver_loss = diver_loss + div_l_real / ((w_s[0]*w_s[1]*(w_s[0]*w_s[1]-1)) / 2)
            diver_loss = diver_loss / w_s[3]


        return diver_loss

