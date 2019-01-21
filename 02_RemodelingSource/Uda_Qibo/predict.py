#!/usr/bin/env python3
import re
import os.path
import tensorflow as tf
import warnings
import math
import time
import scipy.misc
import numpy as np
from glob import glob
from tqdm import tqdm
from collections import namedtuple
from distutils.version import LooseVersion
#import project_tests as tests

from timeit import default_timer as timer


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

# parameters used in training
STDDEV = 0.01   # std for kernel initializer
L2_REG = 1e-3   # regularization parameter for kernal
KEEP_PROB = 0.5
LEARNING_RATE = 5e-4

DATA_DIR = './data'
RUNS_DIR = './runs'
MODELS = './models'
TRAIN_SUBDIR = 'seg_train_images'
TRAIN_GT_SUBDIR = 'seg_train_annotations'
TEST_SUBDIR = 'seg_test_images'
FL_UDA_SEV = 1
VGG_DIR = './data'      # this setting is for udacity workspace

EPOCHS = 30
BATCH_SIZE = 16
IMAGE_SHAPE = (320, 480)    # AI contest dataset uses 1216x1936 images
IMAGE_SHAPE_ORIGIN = (1216, 1936)
FL_resume = True
num_classes = 20

# num_classes 20
# background (unlabeled) + 4 classes as per official benchmark
# cf "The Cityscapes Dataset for Semantic Urban Scene Understanding"
Label = namedtuple('Label', ['name', 'color'])
label_defs = [
    Label('undefined',     (  0,   0,   0)),
    Label('car',           (  0,   0, 255)),
    Label('pedestrian',    (255,   0,   0)),
    Label('signal',        (255, 255,   0)),
    Label('lane',          ( 69,  47, 142)),
    #Add
    Label('sidewalk',      (  0, 255, 255)),
    Label('building',      (  0, 203, 151)),
    Label('wall',          ( 92, 136, 125)),
    Label('fence',         (215,   0, 255)),
    Label('pole',          (180, 131, 135)),
    Label('trafficsign',   (255, 134,   0)),
    Label('vegetation',    ( 85, 255,  50)),
    Label('terrain',       (136,  45,  66)),
    Label('sky',           (  0, 152, 225)),
    Label('rider',         ( 86,  62,  67)),
    Label('truck',         (180,   0, 129)),
    Label('bus',           (193, 214,   0)),
    Label('train',         (255, 121, 166)),
    Label('motorcycle',    ( 65, 166,   1)),
    Label('bicycle',       (208, 149,   1))]

label_colors = {i: np.array(l.color) for i, l in enumerate(label_defs)}


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    
    input_image = graph.get_tensor_by_name(vgg_input_tensor_name)
    
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)

    layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    return input_image, keep_prob, layer3_out, layer4_out, layer7_out
#tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """

    # 1x1 convolution for vgg layer 3
    layer3_conv_1x1 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, strides=(1,1), padding='same',
                                       kernel_initializer= tf.random_normal_initializer(stddev=STDDEV),
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_REG))

    # 1x1 convolution for vgg layer 4
    layer4_conv_1x1 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, strides=(1,1), padding='same',
                                       kernel_initializer= tf.random_normal_initializer(stddev=STDDEV),
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_REG))

    # 1x1 convolution for vgg layer 7
    layer7_conv_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, strides=(1,1), padding='same',
                                       kernel_initializer= tf.random_normal_initializer(stddev=STDDEV),
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_REG))

    # 2x upsample for layer 7
    layer7_2x = tf.layers.conv2d_transpose(layer7_conv_1x1, num_classes, 4, strides=(2,2), padding='same',
                                           kernel_initializer= tf.random_normal_initializer(stddev=STDDEV), 
                                           kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_REG))

    # skip connection
    layer4_skip = tf.add(layer7_2x, layer4_conv_1x1)

    # 2x upsample for layer 4 & layer 7_2x
    layer47_2x = tf.layers.conv2d_transpose(layer4_skip, num_classes, 4, strides=(2,2), padding='same',
                                            kernel_initializer= tf.random_normal_initializer(stddev=STDDEV),
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_REG))

    # skip connection
    layer3_skip = tf.add(layer47_2x, layer3_conv_1x1)
    
    # 8x upsample for layer 3 & layer 4_2x & layer 7_4x
    nn_last_layer = tf.layers.conv2d_transpose(layer3_skip, num_classes, 16, strides=(8,8), padding='same',
                                               kernel_initializer= tf.random_normal_initializer(stddev=STDDEV),
                                               kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_REG))
    
    return nn_last_layer
#tests.test_layers(layers)


def build_predictor(nn_last_layer):
    softmax_output = tf.nn.softmax(nn_last_layer)
    predictions_argmax = tf.argmax(softmax_output, axis=-1)
    return softmax_output, predictions_argmax


def build_metrics(correct_label, predictions_argmax, num_classes):
    labels_argmax = tf.argmax(correct_label, axis=-1)
    iou, iou_op = tf.metrics.mean_iou(labels_argmax, predictions_argmax, num_classes)
    return iou, iou_op


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels))

    # loss function of weight
    regularization_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)) # Scalar
    cross_entropy_loss = cross_entropy_loss + regularization_loss

    # define optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)

    # define training operation
    train_op = optimizer.minimize(cross_entropy_loss)
    
    return logits, train_op, cross_entropy_loss
#tests.test_optimize(optimize)


def gen_test_output(sess, logits, keep_prob, image_pl, image_files, image_shape):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_pl: TF Placeholder for the image placeholder
    :param data_folder: Path to the folder that contains the datasets
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """
    for image_file in image_files:
        image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)

        im_softmax = sess.run(
            [tf.argmax(tf.nn.softmax(logits), axis=-1)],
            {keep_prob: 1.0, image_pl: [image]})

        im_softmax = im_softmax[0].reshape(image_shape[0], image_shape[1])
        labels_colored = np.zeros([image_shape[0], image_shape[1], 3])

        for label in label_colors:
            label_mask = (im_softmax == label)
            labels_colored[label_mask] = np.array(label_colors[label])

        img = scipy.misc.imresize(labels_colored, IMAGE_SHAPE_ORIGIN, interp='nearest')
        street_im = scipy.misc.toimage(img, mode="RGB", cmin=0, cmax=255)
        #street_im = scipy.misc.toimage(image)
        #street_im.paste(mask, box=None, mask=mask)

        yield re.sub(r'jpg', 'png', os.path.basename(image_file)), street_im


def save_inference_samples(runs_dir, image_files, sess, image_shape, logits, keep_prob, input_image):
    # Make folder for current run
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    print('Training Finished. Saving test images to: {}'.format(output_dir))
    image_outputs = gen_test_output(sess, logits, keep_prob, input_image, image_files, image_shape)
    for name, image in image_outputs:
        #scipy.misc.imsave(os.path.join(output_dir, name), image)
        image.save(os.path.join(output_dir, name))


def run():
    # Path to vgg model
    vgg_path = os.path.join(VGG_DIR, 'vgg')

    correct_label = tf.placeholder(tf.int32, [None, IMAGE_SHAPE[0], IMAGE_SHAPE[1], num_classes])
    learning_rate = tf.placeholder(tf.float32)
            
    with tf.Session() as sess:
        # Build NN using load_vgg, layers, and optimize function
        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        nn_last_layer = layers(layer3_out, layer4_out, layer7_out, num_classes)
        logits, train_op, cross_entropy_loss = optimize(nn_last_layer, correct_label, learning_rate, num_classes)

        # WARNING run those initializer _BEFORE_ restore
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())  # important for IoU calculation
        saver = tf.train.Saver()

        # resume training from saved params
        if FL_resume:
            saver.restore(sess, tf.train.latest_checkpoint(MODELS))
            print("resume")

        # Save inference data using helper.save_inference_samples
        inference_images = glob(os.path.join(DATA_DIR, TEST_SUBDIR, '*.jpg'))
        #inference_images = ['./data/seg_test_images/test_009.jpg', './data/seg_test_images/test_010.jpg']
        save_inference_samples(RUNS_DIR, inference_images, sess, IMAGE_SHAPE, logits, keep_prob, input_image)


if __name__ == '__main__':
    run()
