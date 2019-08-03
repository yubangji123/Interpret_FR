"""Routine for decoding the CIFAR-10 binary file format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

# Process images of this size. Note that this differs from the original CIFAR
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
# IMAGE_SIZE = 24
IMAGE_SIZE = 96
# Global constants describing the CIFAR-10 data set.

NUM_CLASSES = 10

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000

NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000



def read_CASIA(filename_queue, filelist_queue):
    class CIFAR10Record(object):
        pass

    result = CIFAR10Record()

    # Dimensions of the images in the CIFAR-10 dataset.
    # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
    # input format.

    label_bytes = 8+8+8+8  # 2 for CIFAR-100
    result.height = 110
    result.width = 110

    result.depth = 3
    image_bytes = result.height * result.width * result.depth
    # Every record consists of a label followed by the image, with a
    # fixed number of bytes for each.

    record_bytes = image_bytes

    # Read a record, getting filenames from the filename_queue.  No
    # header or footer in the CIFAR-10 format, so we leave header_bytes
    # and footer_bytes at their default of 0.

    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, img_value = reader.read(filename_queue)

    # Convert from a string to a vector of uint8 that is record_bytes long.
    record_bytes = tf.decode_raw(img_value, tf.uint8)

    # The remaining bytes after the label represent the image, which we reshape
    # from [depth * height * width] to [depth, height, width].
    depth_major = tf.reshape(tf.strided_slice(record_bytes, [0], [image_bytes]), [result.depth, result.height, result.width])

    # Convert from [depth, height, width] to [height, width, depth].
    result.uint8image = tf.transpose(depth_major, [1, 2, 0])

    record_bytes = label_bytes
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, list_value = reader.read(filename_queue)
    record_bytes = tf.decode_raw(list_value, tf.uint8)

    # The first bytes represent the label, which we convert from uint8->int32.
#    result.filename = tf.cast(tf.strided_slice(list_value, [0], [30]), tf.string)
    result.pid = tf.cast(tf.strided_slice(record_bytes, [2], [3]), tf.uint8)
    result.yaw = tf.cast(tf.strided_slice(record_bytes, [3], [4]), tf.float32)


    return result


def _generate_image_and_label_batch(read_input, min_queue_examples,
                                    batch_size, shuffle):
    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.

    num_preprocess_threads = 8
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
            [read_input.uint8image, read_input.pid],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)

    # Display the training images in the visualizer.
    tf.summary.image('images', images)

    return images, tf.reshape(label_batch, [batch_size])


def distorted_inputs(batch_size):

    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer(['DATA/CASIA_all_images_110_110.dat'])
    filelist_queue = tf.train.string_input_producer(['DATA/CASIA_recrop_fileList.dat'])

    # Read examples from files in the filename queue.
    read_input = read_CASIA(filename_queue, filelist_queue)

    read_input.pid.set_shape([1])

    # Ensure that the random shuffling has good mixing properties.

    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             min_fraction_of_examples_in_queue)
    print('Filling queue with %d CIFAR images before starting to train. '
          'This will take a few minutes.' % min_queue_examples)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(read_input, min_queue_examples, batch_size, shuffle=True)
#if __name__ == '__main__':
    #distorted_inputs(64)
