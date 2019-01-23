#!/usr/bin/env python3
import os.path
import tensorflow as tf
import helper
import warnings
import math
from glob import glob

from timeit import default_timer as timer

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
MODELS = './FCN8_Model'
TRAIN_SUBDIR = 'seg_images_train'
TRAIN_GT_SUBDIR = 'seg_annotations_train'
TEST_SUBDIR = 'seg_images_test'
VGG_DIR = './data'      # this setting is for udacity workspace

EPOCHS = 1
BATCH_SIZE = 4
IMAGE_SHAPE = (320, 480)    # AI contest dataset uses 1216x1936 images
FL_resume = False


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



def train_nn(sess, epochs, batch_size, get_train_batches_fn, get_valid_batches_fn, train_op, cross_entropy_loss,
             input_image, correct_label, keep_prob, learning_rate, iou, iou_op, saver, n_train, n_valid, lr):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    print("Start training with lr {} ...".format(lr))
    best_iou = 0
    for epoch in range(epochs):
        # train process
        start = timer()
        losses = []
        ious = []

        description = "Train Epoch {:>2}/{}".format(epoch+1,epochs)
        print(description)

        for image, label in get_train_batches_fn(batch_size):
            _, loss, _ = sess.run([train_op, cross_entropy_loss, iou_op],
                                feed_dict={input_image: image, correct_label: label,
                                            keep_prob: KEEP_PROB, learning_rate: lr})
            print("Loss = {:.3f}".format(loss))
            losses.append(loss)
            ious.append(sess.run(iou))

        end = timer()
        # save figure of loss
        #helper.plot_loss(RUNS_DIR, losses, "loss_graph_training")
        print("EPOCH {} with lr {} ...".format(epoch + 1, lr))
        print("  time {} ...".format(end - start))
        print("  Train Xentloss = {:.4f}".format(sum(losses) / len(losses)))
        print("  Train IOU = {:.4f}".format(sum(ious) / len(ious)))


        # validation process
        start = timer()
        losses = []
        ious = []

        for image, label in get_valid_batches_fn(batch_size):
            loss, _ = sess.run([cross_entropy_loss, iou_op],
                                feed_dict={input_image: image, correct_label: label, keep_prob: 1})
            print("Loss = {:.3f}".format(loss))
            losses.append(loss)
            ious.append(sess.run(iou))
        end = timer()
        # save figure of loss
        #helper.plot_loss(RUNS_DIR, losses, "loss_graph_validating")
        print("  time {} ...".format(end - start))
        print("  Valid Xentloss = {:.4f}".format(sum(losses) / len(losses)))
        valid_iou = sum(ious) / len(ious)
        print("  Valid IOU = {:.4f}".format(valid_iou))


        # check the result
        if (valid_iou > best_iou):
            saver.save(sess, os.path.join(MODELS, 'fcn8s'))
            #saver.save(sess, os.path.join(MODELS, 'fcn8s.ckpt'))
            with open(os.path.join(MODELS, 'training.txt'), "w") as text_file:
                text_file.write("models/fcn8s: epoch {}, lr {}, valid_iou {}".format(epoch + 1, lr, valid_iou))
            print("  model saved")
            best_iou = valid_iou
        else:
            lr *= 0.5  # lr scheduling: halving on failure
            print("  no improvement => lr downscaled to {} ...".format(lr))
    pass



def test_nn(sess, batch_size, get_test_batches_fn, predictions_argmax, input_image, correct_label, keep_prob, iou,
            iou_op, n_batches):
    ious = []
    for image, label in get_test_batches_fn(batch_size):
        labels, _ = sess.run([predictions_argmax, iou_op],
                            feed_dict={input_image: image, correct_label: label, keep_prob: 1})
        ious.append(sess.run(iou))
    print("  Test IOU = {:.4f}".format(sum(ious) / len(ious)))


def run():
    # Path to vgg model
    vgg_path = os.path.join(VGG_DIR, 'vgg')

    # prepare images (train, valid, test)
    train_images, valid_images, test_images, num_classes = helper.load_data(DATA_DIR)
    print("len: train_images {}, valid_images {}, test_images {}".format(len(train_images), len(valid_images), len(test_images)))

    # Create function to get batches
    get_train_batches_fn = helper.gen_batch_function(train_images, IMAGE_SHAPE)
    get_valid_batches_fn = helper.gen_batch_function(valid_images, IMAGE_SHAPE)
    get_test_batches_fn = helper.gen_batch_function(test_images, IMAGE_SHAPE)

    correct_label = tf.placeholder(tf.int32, [None, IMAGE_SHAPE[0], IMAGE_SHAPE[1], num_classes])
    learning_rate = tf.placeholder(tf.float32)
    # learning rate
    lr = LEARNING_RATE

    #Solve ran out of gpu memory
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
            
    with tf.Session(config=config) as sess:
        # Build NN using load_vgg, layers, and optimize function
        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        nn_last_layer = layers(layer3_out, layer4_out, layer7_out, num_classes)
        logits, train_op, cross_entropy_loss = optimize(nn_last_layer, correct_label, learning_rate, num_classes)

        softmax_output, predictions_argmax = build_predictor(nn_last_layer)
        iou, iou_op = build_metrics(correct_label, predictions_argmax, num_classes)

        # WARNING run those initializer _BEFORE_ restore
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())  # important for IoU calculation
        saver = tf.train.Saver()

        # resume training from saved params
        if FL_resume:
            saver.restore(sess, tf.train.latest_checkpoint(MODELS))
            print("resume")
            
        # Train NN using the train_nn function
        n_train = int(math.ceil(len(train_images) / BATCH_SIZE))
        n_valid = int(math.ceil(len(valid_images) / BATCH_SIZE))
        train_nn(sess, EPOCHS, BATCH_SIZE, get_train_batches_fn, get_valid_batches_fn, train_op, cross_entropy_loss,
                 input_image, correct_label, keep_prob, learning_rate, iou, iou_op, saver, n_train, n_valid, lr)

        # Test process
        #n_batches = int(math.ceil(len(test_images) / BATCH_SIZE))
        # batch_size 32 is ok (and faster) with GTX 1080 TI and 11 GB memory
        #test_nn(sess, 32, get_test_batches_fn, predictions_argmax, input_image,
        #       correct_label, keep_prob, iou, iou_op, n_batches)


        # Save inference data using helper.save_inference_samples
        inference_images = glob(os.path.join(DATA_DIR, TEST_SUBDIR, '*.jpg'))
        helper.save_inference_samples(RUNS_DIR, inference_images, sess, IMAGE_SHAPE, logits, keep_prob, input_image)

        # save model
        output_node_names = 'Softmax'
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,  # The session is used to retrieve the weights
            tf.get_default_graph().as_graph_def(),  # The graph_def is used to retrieve the nodes
            output_node_names.split(",")  # The output node names are used to select the usefull nodes
        )

        saver.save(sess, os.path.join(MODELS, 'optimized', 'fcn8s.ckpt'))
        tf.train.write_graph(tf.get_default_graph().as_graph_def(), '',  os.path.join(MODELS, 'optimized', 'base_graph.pb'), False)
        tf.train.write_graph(output_graph_def, '', os.path.join(MODELS, 'optimized' 'frozen_graph.pb'), False)
        print("{} ops in the final graph.".format(len(output_graph_def.node)))

if __name__ == '__main__':
    run()
