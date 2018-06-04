from __future__ import absolute_import, division, print_function
import tensorflow as tf
import glob
import os.path
import sys
import PIL.Image as img
import numpy as np
from matplotlib import pyplot as plt

from bilinear_sampler2 import *

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('cityscapes_root', '/home/jiarui/Results', 'Cityscapes dataset root folder.')

_FOLDERS_MAP = {
    'right': 'cityscapes_vis/rightvis/segmentation_results',
    'left': 'cityscapes_vis/vis/segmentation_results',
    'disparity': 'dis/',
}

_PATTERN_MAP = {
    'right': '_image.png',
    'left': '_prediction.png',
    'disparity': '.png',
}

def _get_files(data):
    pattern = '*%s' % (_PATTERN_MAP[data])
    search_files = os.path.join(FLAGS.cityscapes_root, _FOLDERS_MAP[data], pattern)
    filenames = glob.glob(search_files)
    return sorted(filenames)

def read_image(image_path):
    image  = tf.image.decode_png(tf.read_file(image_path))
    #image  = tf.image.convert_image_dtype(image, tf.float32)
    image  = tf.image.convert_image_dtype(image, tf.uint8)
    image = tf.cast(image, tf.float32)
    return image

def _stack_disparity(disparity_map_files, count_disparity):
    disparity_map = read_image(disparity_map_files[count_disparity])
    count_disparity = count_disparity + 1
    for j in range(7):
        disparity_map = tf.concat([disparity_map, read_image(disparity_map_files[count_disparity])], 0)
        count_disparity = count_disparity + 1

    return disparity_map, count_disparity

def main(_):
    right_image_files = _get_files('right')
    disparity_map_files = _get_files('disparity')
    num_images = len(right_image_files)
    #num_disparities = len(disparity_map_files)
    count_disparity = 0
    num_images = 1

    for i in range(num_images):
        right_image  = read_image(right_image_files[i])
        right_image = tf.expand_dims(right_image, 0)
        #right_image  = tf.image.decode_png(tf.read_file(right_image_files[i]))
        #right_image  = tf.image.convert_image_dtype(right_image,  tf.uint8)
        #right_image = tf.cast(right_image, tf.float32)

        disparity_map = read_image(disparity_map_files[i])
        disparity_map = tf.slice(disparity_map, [0,0,1], [1024, 2048, 1])
        #disparity_map, count_disparity = _stack_disparity(disparity_map_files, count_disparity)
        #disparity_map = disparity_map * 2048
        #disparity_map  = tf.image.decode_png(tf.read_file(disparity_map_files[i]))
        #disparity_map  = tf.image.convert_image_dtype(disparity_map,  tf.uint16)
        #disparity_map = tf.cast(disparity_map, tf.float32)
        #disparity_map = tf.divide(disparity_map-1, 255)

        left_est = bilinear_sampler_1d_h(right_image, disparity_map)
        left_est_summary = tf.expand_dims(left_est, 0)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        init = tf.global_variables_initializer()
        sess.run(init)
        writer = tf.summary.FileWriter('logs')
        summary_op = tf.summary.image('left_est_'+str(i), left_est_summary)

        summary = sess.run(summary_op)
        writer.add_summary(summary)

        image_data = tf.image.convert_image_dtype(left_est, dtype = tf.uint8)
        image_data = tf.image.encode_png(image_data)

        #image_data = sess.run(left_est)
        #image_array = np.array(left_est)
        #image_data = tf.cast(left_est, tf.uint8)
        #image_data = img.fromarray(image_array.astype(dtype=np.uint8))
        image_name = os.path.basename(right_image_files[i])
        save_dir = '/home/jiarui/Results/left_prediction';
        filename = '%s/%s' % (save_dir, image_name)
        image_data_hd = sess.run(image_data)
        hd = tf.gfile.FastGFile(filename, 'w')
        hd.write(image_data_hd)
        hd.close()
        #with tf.gfile.Open('%s/%s' % (save_dir, image_name), mode='wb') as f:
        #    f.write(image_data)
        #    image_data.save(f, 'PNG')

    writer.close()
    sess.close()

if __name__ == '__main__':
    tf.app.run()
