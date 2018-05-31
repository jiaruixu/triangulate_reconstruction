from __future__ import absolute_import, division, print_function
import tensorflow as tf
import glob
import os.path
import sys
import PIL.Image as img
import numpy as np
from matplotlib import pyplot as plt

# from bilinear_sampler_left_recons import *
# from bilinear_sampler import *

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('cityscapes_root', '/home/jiarui/Results', 'Cityscapes dataset root folder.')

_FOLDERS_MAP = {
    'right': 'cityscapes_vis/rightvis/segmentation_results',
    'left': 'cityscapes_vis/vis/segmentation_results',
    'disparity': 'cityscapes_disparity/',
}

_PATTERN_MAP = {
    'right': '_prediction.png',
    'left': '_prediction.png',
    'disparity': '.png',
}

def bilinear_sampler_1d_h(input_images, x_offset, depth_map, wrap_mode='edge', name='bilinear_sampler', **kwargs):
    def _repeat(x, n_repeats):
        with tf.variable_scope('_repeat'):
            rep = tf.tile(tf.expand_dims(x, 1), [1, n_repeats])
            return tf.reshape(rep, [-1])

    def _interpolate(im, x, y):
        with tf.variable_scope('_interpolate'):

            # handle both texture border types
            _edge_size = 0
            if _wrap_mode == 'border':
                _edge_size = 1
                im = tf.pad(im, [[1, 1], [1, 1], [0, 0]], mode='CONSTANT')
                x = x + _edge_size
                y = y + _edge_size
            elif _wrap_mode == 'edge':
                _edge_size = 0
            else:
                return None

            x = tf.clip_by_value(x, 0.0,  _width_f - 1 + 2 * _edge_size)

            x0_f = tf.floor(x)
            y0_f = tf.floor(y)
            x1_f = x0_f + 1

            x0 = tf.cast(x0_f, tf.int32)
            y0 = tf.cast(y0_f, tf.int32)
            x1 = tf.cast(tf.minimum(x1_f,  _width_f - 1 + 2 * _edge_size), tf.int32)

            dim2 = (_width + 2 * _edge_size)
            dim1 = (_width + 2 * _edge_size) * (_height + 2 * _edge_size)
            # base = _repeat(tf.range(1) * dim1, _height * _width)
            # base = tf.reshape(tf.tile(tf.expand_dims(tf.range(1), 1), [1, _height * _width]), [-1])
            base = tf.zeros([1, _height * _width], tf.int32)
            base = tf.reshape(base, [-1])
            base_y0 = base + y0 * dim2
            idx_l = base_y0 + x0
            idx_r = base_y0 + x1

            im_flat = tf.reshape(im, tf.stack([-1, _num_channels]))

            pix_l = tf.gather(im_flat, idx_l)
            pix_r = tf.gather(im_flat, idx_r)

            weight_l = tf.expand_dims(x1_f - x, 1)
            weight_r = tf.expand_dims(x - x0_f, 1)

            return weight_l * pix_l + weight_r * pix_r

    def _transform(input_images, x_offset, depth_map):
        with tf.variable_scope('transform'):
            # grid of (x_t, y_t, 1), eq (1) in ref [1]
            x_t, y_t = tf.meshgrid(tf.linspace(0.0,   _width_f - 1.0,  _width),
                                   tf.linspace(0.0 , _height_f - 1.0 , _height))

            x_t_flat = tf.reshape(x_t, [-1])
            y_t_flat = tf.reshape(y_t, [-1])

            #sess1 = tf.Session()
            #print(sess1.run(tf.reduce_min(x_offset)))

            #x_offset_unlabel_idx = tf.reshape(tf.where(tf.less_equal(tf.reshape(x_offset, [-1]), 0)), [-1])

            #print(sess1.run(x_offset_unlabel_idx))

            x_t_flat = x_t_flat - tf.reshape(x_offset, [-1])
            #x_t_flat = x_t_flat - tf.reshape(x_offset, [-1]) * _width_f

            depth_flat = tf.reshape(depth_map, [-1])



            input_transformed = _interpolate(input_images, x_t_flat, y_t_flat)

            #print(sess1.run(tf.gather(input_transformed, x_offset_unlabel_idx)))

            # Mark pixel with disparity<0 as unlabeled
            #input_transformed_handle = tf.Variable(input_transformed, validate_shape = False)
            #vals = tf.tile([[0, 0, 0]], multiples = [tf.shape(x_offset_unlabel_idx)[0],1])
            #vals = tf.cast(vals, tf.float32)
            #input_transformed_new = tf.scatter_update(input_transformed_handle, x_offset_unlabel_idx, vals)
            #output = tf.reshape(input_transformed_new, tf.stack([_height, _width, _num_channels]))

            #sess1.run(tf.global_variables_initializer())
            #print(sess1.run(tf.gather(input_transformed_new, x_offset_unlabel_idx)))

            output = tf.reshape(input_transformed, tf.stack([_height, _width, _num_channels]))
            return output

    with tf.variable_scope(name):
        # _num_batch    = tf.shape(input_images)[0]
        _height       = tf.shape(input_images)[0]
        _width        = tf.shape(input_images)[1]
        _num_channels = tf.shape(input_images)[2]

        _height_f = tf.cast(_height, tf.float32)
        _width_f  = tf.cast(_width,  tf.float32)

        _wrap_mode = wrap_mode

        output = _transform(input_images, x_offset, depth_map)
        return output

def _get_files(data):
    pattern = '*%s' % (_PATTERN_MAP[data])
    search_files = os.path.join(FLAGS.cityscapes_root, _FOLDERS_MAP[data], pattern)
    filenames = glob.glob(search_files)
    return sorted(filenames)

def read_image(image_path):
    image  = tf.image.decode_png(tf.read_file(image_path))
    image  = tf.image.convert_image_dtype(image, tf.float32)
    #image  = tf.image.convert_image_dtype(image, tf.uint8)
    #image = tf.cast(image, tf.float32)
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
    num_images = 3

    for i in range(num_images):
        right_image  = read_image(right_image_files[i])
        #right_image  = tf.image.decode_png(tf.read_file(right_image_files[i]))
        #right_image  = tf.image.convert_image_dtype(right_image,  tf.uint8)
        #right_image = tf.cast(right_image, tf.float32)

        #disparity_map = read_image(disparity_map_files[i])
        disparity_map, count_disparity = _stack_disparity(disparity_map_files, count_disparity)
        disparity_map = disparity_map * 228
        #disparity_map  = tf.image.decode_png(tf.read_file(disparity_map_files[i]))
        #disparity_map  = tf.image.convert_image_dtype(disparity_map,  tf.uint16)
        #disparity_map = tf.cast(disparity_map, tf.float32)
        #disparity_map = tf.divide(disparity_map-1, 255)

        # Compute depth. focal length = 2268.36 (in pixels), baseline = 0.222126 meters, disparities are in pixels
        depth_map = tf.divide(2268.26 * 0.222126, disparity_map) # in meters

        left_est = bilinear_sampler_1d_h(right_image, disparity_map, depth_map)
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
