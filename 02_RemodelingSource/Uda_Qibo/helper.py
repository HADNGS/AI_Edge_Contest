import re
import random
import numpy as np
import os.path
import scipy.misc
import shutil
import zipfile
import time
import tensorflow as tf
from glob import glob
from urllib.request import urlretrieve
#from tqdm import tqdm
from collections import namedtuple
import matplotlib.pyplot as plt


#class DLProgress(tqdm):
#    last_block = 0

#    def hook(self, block_num=1, block_size=1, total_size=None):
#        self.total = total_size
#        self.update((block_num - self.last_block) * block_size)
#        self.last_block = block_num


def maybe_download_pretrained_vgg(data_dir):
    """
    Download and extract pretrained vgg model if it doesn't exist
    :param data_dir: Directory to download the model to
    """
    vgg_filename = 'vgg.zip'
    vgg_path = os.path.join(data_dir, 'vgg')
    vgg_files = [
        os.path.join(vgg_path, 'variables/variables.data-00000-of-00001'),
        os.path.join(vgg_path, 'variables/variables.index'),
        os.path.join(vgg_path, 'saved_model.pb')]

    missing_vgg_files = [vgg_file for vgg_file in vgg_files if not os.path.exists(vgg_file)]
    if missing_vgg_files:
        # Clean vgg dir
        if os.path.exists(vgg_path):
            shutil.rmtree(vgg_path)
        os.makedirs(vgg_path)

        # Download vgg
        print('Downloading pre-trained vgg model...')
        with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
            urlretrieve(
                'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip',
                os.path.join(vgg_path, vgg_filename),
                pbar.hook)

        # Extract vgg
        print('Extracting model...')
        zip_ref = zipfile.ZipFile(os.path.join(vgg_path, vgg_filename), 'r')
        zip_ref.extractall(data_dir)
        zip_ref.close()

        # Remove zip file to save space
        os.remove(os.path.join(vgg_path, vgg_filename))

# num_classes 5
# background (unlabeled) + 4 classes as per official benchmark
# cf "The Cityscapes Dataset for Semantic Urban Scene Understanding"
Label = namedtuple('Label', ['name', 'color'])
label_defs = [
    Label('undefined',     (  0,   0,   0)),
    Label('car',           (  0,   0, 255)),
    Label('pedestrian',    (255,   0,   0)),
    Label('signal',        (255, 255,   0)),
    Label('lane',          ( 69,  47, 142))]

label_colors = {i: np.array(l.color) for i, l in enumerate(label_defs)}


def load_data(data_folder, train_image_folder, train_gt_folder):
    # make list of all files
    image_files = glob(os.path.join(data_folder, train_image_folder, '*.jpg'))
    file_list = []
    for f in image_files:
        image_file_base = os.path.basename(f)
        gt_file_base = re.sub(r'jpg', 'png', image_file_base)
        f_gt = os.path.join(data_folder, train_gt_folder, gt_file_base)
        # check existance of annotation file
        if os.path.exists(f_gt):
            file_list.append((f, f_gt))
                                    
    # random.shuffle(file_list)
    # split list into train, validation, test lists
    train_images = file_list[0:1999]
    valid_images = file_list[2000:]
    test_images = []#file_list[2200:]
    #train_images = file_list[0:70]
    #valid_images = file_list[71:90]
    #test_images = file_list[91:]
    # label & classes
    num_classes = len(label_defs)
    #label_colors = {i: np.array(l.color) for i, l in enumerate(label_defs)}

    return train_images, valid_images, test_images, num_classes


def bc_img(img, s=1.0, m=0.0):
    img = img.astype(np.int)
    img = img * s + m
    img[img > 255] = 255
    img[img < 0] = 0
    img = img.astype(np.uint8)
    return img


def gen_batch_function(image_list, image_shape):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """
    def get_batches_fn(batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        random.shuffle(image_list)
        background_color = np.array([255, 0, 0])

        for batch_i in range(0, len(image_list), batch_size):
            images = []
            gt_images = []
            for f in image_list[batch_i:batch_i+batch_size]:
                # read image and groundtruth files
                image_file = f[0]
                gt_file = f[1]
                image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
                gt_image = scipy.misc.imresize(scipy.misc.imread(gt_file), image_shape)

                # add random noises
                contrast = random.uniform(0.85, 1.15)  # Contrast augmentation
                bright = random.randint(-45, 30)  # Brightness augmentation
                image = bc_img(image, contrast, bright)
                
                gt_bg = np.zeros([image_shape[0], image_shape[1]], dtype=bool)
                gt_list = []
                for ldef in label_defs[1:]:
                    gt_current = np.all(gt_image == np.array(ldef.color), axis=2)
                    gt_bg |= gt_current
                    gt_list.append(gt_current)
                gt_bg = ~gt_bg
                gt_all = np.dstack([gt_bg, *gt_list])
                gt_all = gt_all.astype(np.float32)

                images.append(image)
                gt_images.append(gt_all)

            yield np.array(images), np.array(gt_images)
    return get_batches_fn


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
        labels_colored = np.zeros([image_shape[0], image_shape[1], 4])

        for label in label_colors:
            label_mask = (im_softmax == label)
            labels_colored[label_mask] = np.array((*label_colors[label], 255))

        mask = scipy.misc.toimage(labels_colored, mode="RGBA")
        street_im = scipy.misc.toimage(image)
        street_im.paste(mask, box=None, mask=mask)

        yield os.path.basename(image_file), np.array(street_im)


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
        scipy.misc.imsave(os.path.join(output_dir, name), image)


def plot_loss(runs_dir, loss, folder_name):
    _, axes = plt.subplots()
    plt.plot(range(0, len(loss)), loss)
    plt.title('Cross-entropy loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    if not os.path.exists(runs_dir):
        #shutil.rmtree(runs_dir)
        os.makedirs(runs_dir)

    output_file = os.path.join(runs_dir, folder_name + ".png")
    plt.savefig(output_file)
