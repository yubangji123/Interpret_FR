import os
import scipy.misc
import numpy as np

from model_pretrain_CASIA import InterpFR

import tensorflow as tf

flags = tf.flags
flags.DEFINE_integer("epoch", 1000, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", 1000000, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("sample_size", 169, "The size of batch images [64]")
flags.DEFINE_integer("image_size", 110, "The size of image to use (will be center cropped) [108]")
flags.DEFINE_integer("output_size", 64, "The size of the output images to produce [64]")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
flags.DEFINE_boolean("is_with_y", True, "True for with lable")
flags.DEFINE_string("dataset", "celebA", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("samples_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("is_crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
flags.DEFINE_integer("gf_dim", 32, "")
flags.DEFINE_integer("gfc_dim", 512, "")
flags.DEFINE_integer("df_dim", 32, "")
flags.DEFINE_integer("dfc_dim", 512, "")
flags.DEFINE_integer("z_dim", 50, "")
flags.DEFINE_string("gpu", "1,2", "GPU to use [0]")
FLAGS = flags.FLAGS


def main(_):

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.samples_dir):
        os.makedirs(FLAGS.samples_dir)

    gpu_options = tf.GPUOptions(visible_device_list=FLAGS.gpu, per_process_gpu_memory_fraction=0.95,
                                    allow_growth=True)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, gpu_options=gpu_options)) as sess:
        interp_fr = InterpFR(sess, image_size=FLAGS.image_size, batch_size=FLAGS.batch_size, sample_size=FLAGS.sample_size, output_size=96, y_dim = 320, z_dim=100, gf_dim=FLAGS.gf_dim, gfc_dim=FLAGS.gfc_dim, df_dim=FLAGS.df_dim, dfc_dim=FLAGS.dfc_dim, c_dim=FLAGS.c_dim, dataset_name=FLAGS.dataset,checkpoint_dir=FLAGS.checkpoint_dir, samples_dir=FLAGS.samples_dir)
        if FLAGS.is_train:
            interp_fr.train(FLAGS)
        else:
            interp_fr.load(FLAGS.checkpoint_dir)
            interp_fr.test_IJBA()

if __name__ == '__main__':
    tf.app.run()
