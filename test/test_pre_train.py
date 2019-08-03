import argparse
import tensorflow as tf
import numpy as np

def load_IJBA_recrop_test(isFlip = False):
    print('Loading IJBA recrop...')

    fd = open('../DATA/IJBA_recrop_images_96_96_test.dat')
    images = np.fromfile(file=fd,dtype=np.uint8)
    fd.close()

    images = images.reshape((-1,96,96,3)).astype(np.float32)
    images = images/127.5 - 1.
    print('    DONE. Finish loading IJBA recrop with ' + str(images.shape[0]) + ' images')

    return images

def load_graph(frozen_graph_filename):
    with open(frozen_graph_filename, 'rb') as f:
        fileContent = f.read()

    graph_def = tf.GraphDef()
    graph_def.ParseFromString(fileContent)
    tf.import_graph_def(graph_def)
    graph = tf.get_default_graph()

    return graph

if __name__ == '__main__':

    # We use our "load_graph" function
    graph = load_graph('models/pretrained_model.pb')

    # We can verify that we can access the list of operations in the graph
    for op in graph.get_operations():
        print(op.name)

    # We access the input and output nodes
    sample_input_images = graph.get_tensor_by_name('import/sample_input_images:0')
    sample_feature = graph.get_tensor_by_name('import/Reshape_134:0')

    # Get test input images
    images = load_IJBA_recrop_test()
    N = images.shape[0]
    gallery_features = np.zeros((N, 320), dtype=np.float32)

    # We launch a Session
    with tf.Session(graph=graph) as sess:
        for idx in range(N):
            print(idx)
            batch_img = np.array(images[idx, :, :, :]).astype(np.float32).reshape( (1, 96, 96, 3))
            gallery_feature = sess.run(sample_feature, feed_dict={sample_input_images:  batch_img})
            gallery_features[idx, :] = gallery_feature

    np.savetxt("IJBA_features_iter_0.txt", gallery_features)
    print("Writed to IJBA_features_iter_0.txt")