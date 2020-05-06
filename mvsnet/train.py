#!/usr/bin/env python
"""
Copyright 2019, Yao Yao, HKUST.
Training script.
"""

from __future__ import print_function

import os
import os.path
import time
import sys
import math
import argparse
from random import randint
import sacred
from sacred import Experiment
from sacred.observers import MongoObserver

from getpass import getpass


import cv2
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

sys.path.append("../")
from tools.common import Notify

from preprocess import *
from model import *
from loss import * 
from homography_warping import get_homographies, homography_warping



# username = os.environ['LOGNAME'] # If you are using run-docker
# serverip = os.environ['MONGOIP'] # If you are using run-docker
# password = getpass()
# uri = "mongodb://{}:{}@{}/?authSource=admin".format(username, password, serverip)

# obs = MongoObserver.create(uri, 'romanoni_db')

# ex = Experiment()


# paths
tf.app.flags.DEFINE_string('dtu_data_root', '/data/dtu/', 
                           """Path to dtu dataset.""")
tf.app.flags.DEFINE_string('log_dir2', '/data/tf_log',
                           """Path to store the log.""")
tf.app.flags.DEFINE_string('model_dir', '/data/tf_model',
                           """Path to save the model.""")
tf.app.flags.DEFINE_boolean('train_dtu', True, 
                            """Whether to train.""")
tf.app.flags.DEFINE_boolean('use_pretrain', False, 
                            """Whether to train.""")
tf.app.flags.DEFINE_boolean('use_colmap', True, 
                            """Whether to train.""")
tf.app.flags.DEFINE_integer('ckpt_step', 5,
                            """ckpt step.""")

# input parameters
tf.app.flags.DEFINE_integer('view_num', 3, 
                            """Number of images (1 ref image and view_num - 1 view images).""")
tf.app.flags.DEFINE_integer('max_d', 128, #192, 
                            """Maximum depth step when training.""")
tf.app.flags.DEFINE_integer('max_w', 500,#640, 
                            """Maximum image width when training.""")
tf.app.flags.DEFINE_integer('max_h', 375,#512, 
                            """Maximum image height when training.""")
tf.app.flags.DEFINE_float('sample_scale', 0.25, 
                            """Downsample scale for building cost volume.""")
tf.app.flags.DEFINE_float('interval_scale', 1.06, 
                            """Downsample scale for building cost volume.""")

# network architectures
tf.app.flags.DEFINE_string('regularization', '3DCNNs',
                           """Regularization method.""")
tf.app.flags.DEFINE_boolean('refinement', False,
                           """Whether to apply depth map refinement for 3DCNNs""")

# training parameters
tf.app.flags.DEFINE_integer('num_gpus', 1, 
                            """Number of GPUs.""")
tf.app.flags.DEFINE_integer('batch_size', 1, 
                            """Training batch size.""")
tf.app.flags.DEFINE_integer('epoch', 6, 
                            """Training epoch number.""")
tf.app.flags.DEFINE_float('val_ratio', 0, 
                          """Ratio of validation set when splitting dataset.""")
tf.app.flags.DEFINE_float('base_lr', 0.001,
                          """Base learning rate.""")
tf.app.flags.DEFINE_integer('display', 1,
                            """Interval of loginfo display.""")
tf.app.flags.DEFINE_integer('stepvalue', 10000,
                            """Step interval to decay learning rate.""")
tf.app.flags.DEFINE_integer('snapshot', 5000,
                            """Step interval to save the model.""")
tf.app.flags.DEFINE_float('gamma', 0.85,
                          """Learning rate decay rate.""")


FLAGS = tf.app.flags.FLAGS

FLAGS.log_dir2 = FLAGS.log_dir2 + '/col_prob_colmap_filt_zeros_prob-1.5_' + str(FLAGS.use_colmap) +  \
    '_views_' + str(FLAGS.view_num) + '_' + str(FLAGS.max_d) + \
    '_' + str(FLAGS.max_w) + '_' + str(FLAGS.max_h) + \
    '_sc_' + str(FLAGS.sample_scale) + '_' + str(FLAGS.interval_scale) + \
    '_' + str(FLAGS.base_lr) + '_' + str(FLAGS.gamma) + '_' + str(FLAGS.regularization) + \
    '_ref_' + str(FLAGS.refinement) + '/'



class MVSGenerator: 
    """ data generator class, tf only accept generator without param """
    def __init__(self, sample_list, view_num):
        self.sample_list = sample_list
        self.view_num = view_num
        self.sample_num = len(sample_list)
        self.counter = 0
    
    def __iter__(self):
        while True:
            for data in self.sample_list: 
                start_time = time.time()

                ###### read input data ######
                images = []
                cams = []
                for view in range(self.view_num):
                    image = center_image(cv2.imread(data[4 * view]))#2 to 4 after colmap added
                    cam = load_cam(open(data[4 * view + 1]))
                    cam[1][3][1] = cam[1][3][1] * FLAGS.interval_scale * 192/96
                    images.append(image)
                    cams.append(cam)
                with open(data[4 * self.view_num], 'rb') as f:
                    depth_image = load_pfm(f)
                
                
                colmap_image = load_bin(data[4 * view + 2],1200,1600)

                if os.path.isfile(data[4 * view + 3]):
                    prob_image = load_bin(data[4 * view + 3],1200,1600)
                else:
                    prob_image = np.zeros([1200,1600,1], np.float32) 

                if (
                    (np.mean(colmap_image)>-0.00000001 and np.mean(colmap_image)<0.00000001) or 
                    (np.mean(prob_image)>-0.00000001  and np.mean(prob_image)<0.00000001)
                    ):
                    continue

                # import ipdb; ipdb.set_trace()
                # mask out-of-range depth pixels (in a relaxed range)
                depth_start = cams[0][1, 3, 0] + cams[0][1, 3, 1]
                depth_end = cams[0][1, 3, 0] + (FLAGS.max_d - 2) * cams[0][1, 3, 1]

                maskMin = colmap_image<depth_start
                maskMax = colmap_image>depth_end
                prob_image[maskMin]=0
                prob_image[maskMax]=0

                colmap_image = np.minimum(colmap_image, depth_end)
                colmap_image = np.maximum(colmap_image, depth_start)
                # colmap_image = mask_depth_image(colmap_image, depth_start, depth_end)

                cropx = 1280
                cropy = 1024
                y= colmap_image.shape[0]
                x= colmap_image.shape[1]
                startx = x//2-(cropx//2)
                starty = y//2-(cropy//2)    
                
                colmap_image = colmap_image[starty:starty+cropy,startx:startx+cropx,:]
                prob_image = prob_image[starty:starty+cropy,startx:startx+cropx,:]
                depth_image = mask_depth_image(depth_image, depth_start, depth_end)

                # return mvs input
                self.counter += 1
                duration = time.time() - start_time
                images = np.stack(images, axis=0)
                cams = np.stack(cams, axis=0)
                yield (images, cams, depth_image, colmap_image, prob_image)

                # return backward mvs input for GRU
                if FLAGS.regularization == 'GRU':
                    self.counter += 1
                    start_time = time.time()
                    cams[0][1, 3, 0] = cams[0][1, 3, 0] + (FLAGS.max_d - 1) * cams[0][1, 3, 1]
                    cams[0][1, 3, 1] = -cams[0][1, 3, 1]
                    duration = time.time() - start_time
                    print('Back pass: d_min = %f, d_max = %f.' % \
                        (cams[0][1, 3, 0], cams[0][1, 3, 0] + (FLAGS.max_d - 1) * cams[0][1, 3, 1]))
                    yield (images, cams, depth_image, colmap_image, prob_image)

def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
        List of pairs of (gradient, variable) where the gradient has been averaged
        across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def train(traning_list):
    """ training mvsnet """
    training_sample_size = len(traning_list)
    if FLAGS.regularization == 'GRU':
        training_sample_size = training_sample_size * 2
    print ('sample number: ', training_sample_size)
    # from pudb import set_trace; set_trace()
    with tf.Graph().as_default(), tf.device('/gpu:0'): 

        ########## data iterator #########
        # training generators
        training_generator = iter(MVSGenerator(traning_list, FLAGS.view_num))
        generator_data_type = (tf.float32, tf.float32, tf.float32, tf.float32, tf.float32)
        # dataset from generator
        training_set = tf.data.Dataset.from_generator(lambda: training_generator, generator_data_type)
        training_set = training_set.batch(FLAGS.batch_size)
        training_set = training_set.prefetch(buffer_size=1)
        # iterators
        training_iterator = training_set.make_initializable_iterator()

        ########## optimization options ##########
        global_step = tf.Variable(0, trainable=False, name='global_step')
        lr_op = tf.train.exponential_decay(FLAGS.base_lr, global_step=global_step, 
                                           decay_steps=FLAGS.stepvalue, decay_rate=FLAGS.gamma, name='lr')
        opt = tf.train.RMSPropOptimizer(learning_rate=lr_op)

        tower_grads = []
        for i in range(FLAGS.num_gpus):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('Model_tower%d' % i) as scope:
                    # generate data

                    images, cams, depth_image, colmap_img, prob_im = training_iterator.get_next() 

                    images.set_shape(tf.TensorShape([None, FLAGS.view_num, None, None, 3])) 
                    cams.set_shape(tf.TensorShape([None, FLAGS.view_num, 2, 4, 4]))
                    depth_image.set_shape(tf.TensorShape([None, None, None, 1]))
                    colmap_img.set_shape(tf.TensorShape([None, None, None, 1]))
                    prob_im.set_shape(tf.TensorShape([None, None, None, 1]))
                    depth_start = tf.reshape(
                        tf.slice(cams, [0, 0, 1, 3, 0], [FLAGS.batch_size, 1, 1, 1, 1]), [FLAGS.batch_size])
                    depth_interval = tf.reshape(
                        tf.slice(cams, [0, 0, 1, 3, 1], [FLAGS.batch_size, 1, 1, 1, 1]), [FLAGS.batch_size])


                    is_master_gpu = False
                    if i == 0:
                        is_master_gpu = True

                    # inference
                    if FLAGS.regularization == '3DCNNs':

                        # initial depth map
                        if FLAGS.use_colmap:
                            ref_image = tf.squeeze(tf.slice(images, [0, 0, 0, 0, 0], [-1, 1, -1, -1, 3]), axis=1)
                            depth_shape = tf.shape(ref_image)
                            depth_shape = depth_shape /4
                            depth_shape = tf.cast(depth_shape,tf.int32)
                            # resize normalized image to the same size of depth image
                            resized_colmap_image = tf.image.resize_nearest_neighbor(colmap_img, [depth_shape[1], depth_shape[2]])
                            resized_prob_image = tf.image.resize_nearest_neighbor(prob_im, [depth_shape[1], depth_shape[2]])

                            # tmp = depth_image

                            # depth_end = depth_start + (tf.cast(FLAGS.max_d, tf.float32) - 1) * depth_interval
                            # tmp = tf.minimum(depth_image, tf.cast(depth_end, tf.float32))
                            # tmp = tf.maximum(tmp, depth_start)
                            # tmp_prob = 0.75 * tf.ones(tf.shape(depth_image))

                            # depth_map, prob_map = inference_with_init(
                            #     images, cams, FLAGS.max_d, depth_start, depth_interval,tmp,tmp_prob, is_master_gpu)
                            depth_map, prob_map = inference_with_init(
                                images, cams, FLAGS.max_d, depth_start, depth_interval,resized_colmap_image,resized_prob_image, is_master_gpu)
                        else:
                            depth_map, prob_map = inference(
                                images, cams, FLAGS.max_d, depth_start, depth_interval, is_master_gpu)

                        # refinement
                        if FLAGS.refinement:
                            ref_image = tf.squeeze(
                                tf.slice(images, [0, 0, 0, 0, 0], [-1, 1, -1, -1, 3]), axis=1)
                            if FLAGS.use_colmap:

                                depth_shape = tf.shape(depth_map)
                                # resize normalized image to the same size of depth image
                                resized_colmap_image = tf.image.resize_nearest_neighbor(colmap_img, [depth_shape[1], depth_shape[2]])
                                resized_prob_image = tf.image.resize_nearest_neighbor(prob_im, [depth_shape[1], depth_shape[2]])
                           	    

                                refined_depth_map = depth_refine(depth_map, ref_image, 
                                        FLAGS.max_d, depth_start, depth_interval, 
                                        resized_colmap_image, depth_start, resized_prob_image,
                                        is_master_gpu)
                            else:
                                refined_depth_map = depth_refine(depth_map, ref_image, 
                                        FLAGS.max_d, depth_start, depth_interval, 
                                        is_master_gpu=is_master_gpu)
                        else:
                            refined_depth_map = depth_map

                        # regression loss
                        loss0, less_one_temp, less_three_temp = mvsnet_regression_loss(
                            depth_map, depth_image, depth_interval)
                        loss1, less_one_accuracy, less_three_accuracy = mvsnet_regression_loss(
                            refined_depth_map, depth_image, depth_interval)
                        loss = (loss0 + loss1) / 2

                    elif FLAGS.regularization == 'GRU':

                        # probability volume
                        prob_volume = inference_prob_recurrent(
                            images, cams, FLAGS.max_d, depth_start, depth_interval, is_master_gpu)

                        # classification loss
                        loss, mae, less_one_accuracy, less_three_accuracy, depth_map = \
                            mvsnet_classification_loss(
                                prob_volume, depth_image, FLAGS.max_d, depth_start, depth_interval)
                    
                    # retain the summaries from the final tower.
                    summaries =  tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
                    # depth_map = tf.Print(depth_map, [tf.reduce_mean(resized_colmap_image), tf.reduce_min(resized_colmap_image), tf.reduce_max(resized_colmap_image)], "diffGT_COLMAP ",summarize=5)
                    # depth_image = tf.Print(depth_map, [tf.shape(depth_image), tf.reduce_min(depth_image), tf.reduce_max(depth_image)], "depth_image ",summarize=5)
        
                    # summaries.append(tf.summary.image("depth_map", depth_map))
                    # summaries.append(tf.summary.image("GT", depth_image))

                    # if FLAGS.use_colmap:
                        # summaries.append(tf.summary.image("COLMAP", colmap_img))

                    # calculate the gradients for the batch of data on this CIFAR tower.
                    grads = opt.compute_gradients(loss)

                    # keep track of the gradients across all towers.
                    tower_grads.append(grads)
        
        # average gradient
        grads = average_gradients(tower_grads)
        
        # training opt
        train_opt = opt.apply_gradients(grads, global_step=global_step)

        # # summary 
        summaries.append(tf.summary.scalar('loss', loss))
        summaries.append(tf.summary.scalar('less_one_accuracy', less_one_accuracy))
        summaries.append(tf.summary.scalar('less_three_accuracy', less_three_accuracy))
        summaries.append(tf.summary.scalar('lr', lr_op)) 

        # weights_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        # for var in weights_list:
        #     summaries.append(tf.summary.histogram(var.op.name, var))
        # for grad, var in grads:
        #     if grad is not None:
        #         summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))
        
        # # saver
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)        
        summary_op = tf.summary.merge(summaries)

        # initialization option
        init_op = tf.global_variables_initializer()
        config = tf.ConfigProto(allow_soft_placement = True)#, log_device_placement=True)
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:     
            
            # initialization
            total_step = 0
            sess.run(init_op)
            summary_writer = tf.summary.FileWriter(FLAGS.log_dir2, sess.graph)

            # load pre-trained model
            if FLAGS.use_pretrain:
                pretrained_model_path = os.path.join(FLAGS.model_dir, FLAGS.regularization, 'model.ckpt')
                restorer = tf.train.Saver(tf.global_variables())
                restorer.restore(sess, '-'.join([pretrained_model_path, str(FLAGS.ckpt_step)]))
                print(Notify.INFO, 'Pre-trained model restored from %s' %
                    ('-'.join([pretrained_model_path, str(FLAGS.ckpt_step)])), Notify.ENDC)
                total_step = FLAGS.ckpt_step

            # training several epochs
            for epoch in range(FLAGS.epoch):

                # training of one epoch
                step = 0
                sess.run(training_iterator.initializer)
                for _ in range(int(training_sample_size / FLAGS.num_gpus)):

                    # run one batch
                    start_time = time.time()
                    try:
                        out_summary_op, out_opt, out_loss, out_less_one, out_less_three = sess.run(
                        [summary_op, train_opt, loss, less_one_accuracy, less_three_accuracy])
                    except tf.errors.OutOfRangeError:
                        print("End of dataset")  # ==> "End of dataset"
                        break
                    duration = time.time() - start_time


                    # print info
                    if step % FLAGS.display == 0:
                        print(Notify.INFO,
                            'epoch, %d, step %d, total_step %d, loss = %.4f, (< 1px) = %.4f, (< 3px) = %.4f (%.3f sec/step)' %
                            (epoch, step, total_step, out_loss, out_less_one, out_less_three, duration), Notify.ENDC)
                    
                    # write summary
                    if step % (FLAGS.display * 10) == 0:
                        summary_writer.add_summary(out_summary_op, total_step)
                   
                    # save the model checkpoint periodically
                    if (total_step % FLAGS.snapshot == 0 or step == (training_sample_size - 1)):
                        model_folder = os.path.join(FLAGS.model_dir, FLAGS.regularization)
                        if not os.path.exists(model_folder):
                            os.mkdir(model_folder)
                        ckpt_path = os.path.join(model_folder, 'model.ckpt')
                        print(Notify.INFO, 'Saving model to %s' % ckpt_path, Notify.ENDC)
                        saver.save(sess, ckpt_path, global_step=total_step)
                    step += FLAGS.batch_size * FLAGS.num_gpus
                    total_step += FLAGS.batch_size * FLAGS.num_gpus

# @ex.automain
def main(argv=None):  # pylint: disable=unused-argument
    """ program entrance """
    # Prepare all training samples
    sample_list = gen_dtu_resized_path(FLAGS.dtu_data_root)
    # Shuffle
    random.shuffle(sample_list)
    # Training entrance.
    train(sample_list)


if __name__ == '__main__':
    print ('Training MVSNet with %d views' % FLAGS.view_num)
    tf.app.run()
