#!/usr/bin/env python
"""
Copyright 2019, Yao Yao, HKUST.
Loss formulations.
"""

import sys
import math
import tensorflow as tf
import numpy as np

sys.path.append("../")
from cnn_wrapper.mvsnet import *
from convgru import ConvGRUCell
from homography_warping import *

FLAGS = tf.app.flags.FLAGS


def get_propability_map(cv, depth_map, depth_start, depth_interval):
    """ get probability map from cost volume """

    def _repeat_(x, num_repeats):
        """ repeat each element num_repeats times """
        x = tf.reshape(x, [-1])
        ones = tf.ones((1, num_repeats), dtype='int32')
        x = tf.reshape(x, shape=(-1, 1))
        x = tf.matmul(x, ones)
        return tf.reshape(x, [-1])

    shape = tf.shape(depth_map)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]
    depth = tf.shape(cv)[1]

    # byx coordinate, batched & flattened
    b_coordinates = tf.range(batch_size)
    y_coordinates = tf.range(height)
    x_coordinates = tf.range(width)
    b_coordinates, y_coordinates, x_coordinates = tf.meshgrid(b_coordinates, y_coordinates, x_coordinates)
    b_coordinates = _repeat_(b_coordinates, batch_size)
    y_coordinates = _repeat_(y_coordinates, batch_size)
    x_coordinates = _repeat_(x_coordinates, batch_size)

    # d coordinate (floored and ceiled), batched & flattened
    d_coordinates = tf.reshape((depth_map - depth_start) / depth_interval, [-1])
    d_coordinates_left0 = tf.clip_by_value(tf.cast(tf.floor(d_coordinates), 'int32'), 0, depth - 1)
    d_coordinates_left1 = tf.clip_by_value(d_coordinates_left0 - 1, 0, depth - 1)

#    import ipdb; ipdb.set_trace()
    d_coordinates1_right0 = tf.clip_by_value(tf.cast(tf.ceil(d_coordinates), 'int32'), 0, depth - 1)    
    d_coordinates1_right1 = tf.clip_by_value(d_coordinates1_right0 + 1, 0, depth - 1)

    # voxel coordinates
    voxel_coordinates_left0 = tf.stack(
        [b_coordinates, d_coordinates_left0, y_coordinates, x_coordinates], axis=1)
    voxel_coordinates_left1 = tf.stack(
        [b_coordinates, d_coordinates_left1, y_coordinates, x_coordinates], axis=1)
    voxel_coordinates_right0 = tf.stack(
        [b_coordinates, d_coordinates1_right0, y_coordinates, x_coordinates], axis=1)
    voxel_coordinates_right1 = tf.stack(
        [b_coordinates, d_coordinates1_right1, y_coordinates, x_coordinates], axis=1)

    # get probability image by gathering and interpolation
    prob_map_left0 = tf.gather_nd(cv, voxel_coordinates_left0)
    prob_map_left1 = tf.gather_nd(cv, voxel_coordinates_left1)
    prob_map_right0 = tf.gather_nd(cv, voxel_coordinates_right0)
    prob_map_right1 = tf.gather_nd(cv, voxel_coordinates_right1)
    prob_map = prob_map_left0 + prob_map_left1 + prob_map_right0 + prob_map_right1
    prob_map = tf.reshape(prob_map, [batch_size, height, width, 1])

    return prob_map


def inference(images, cams, depth_num, depth_start, depth_interval, is_master_gpu=True):
    """ infer depth image from multi-view images and cameras """

    # dynamic gpu params
    depth_end = depth_start + (tf.cast(depth_num, tf.float32) - 1) * depth_interval

    # reference image
    ref_image = tf.squeeze(tf.slice(images, [0, 0, 0, 0, 0], [-1, 1, -1, -1, 3]), axis=1)
    ref_cam = tf.squeeze(tf.slice(cams, [0, 0, 0, 0, 0], [-1, 1, 2, 4, 4]), axis=1)

    # image feature extraction    
    if is_master_gpu:
        ref_tower = UNetDS2GN({'data': ref_image}, is_training=True, reuse=False)
    else:
        ref_tower = UNetDS2GN({'data': ref_image}, is_training=True, reuse=True)
    view_towers = []



    for view in range(1, FLAGS.view_num):
        view_image = tf.squeeze(tf.slice(images, [0, view, 0, 0, 0], [-1, 1, -1, -1, -1]), axis=1)
        view_tower = UNetDS2GN({'data': view_image}, is_training=True, reuse=True)
        view_towers.append(view_tower)

    # get all homographies
    view_homographies = []
    for view in range(1, FLAGS.view_num):
        view_cam = tf.squeeze(tf.slice(cams, [0, view, 0, 0, 0], [-1, 1, 2, 4, 4]), axis=1)
        homographies = get_homographies(ref_cam, view_cam, depth_num=depth_num,
                                        depth_start=depth_start, depth_interval=depth_interval)
        # homographies = tf.Print(homographies, [tf.shape(homographies)], "+++++view_homographies() ",summarize=5)
        view_homographies.append(homographies)

    # build cost volume by differentialble homography
    with tf.name_scope('cost_volume_homography'):
        depth_costs = []
        for d in range(depth_num):
            # compute cost (variation metric)
            ave_feature = ref_tower.get_output()
            ave_feature2 = tf.square(ref_tower.get_output())
            for view in range(0, FLAGS.view_num - 1):
                homography = tf.slice(view_homographies[view], begin=[0, d, 0, 0], size=[-1, 1, 3, 3])
                homography = tf.squeeze(homography, axis=1)
                warped_view_feature = homography_warping(view_towers[view].get_output(), homography)
                #warped_view_feature = tf_transform_homography(view_towers[view].get_output(), homography)
                ave_feature = ave_feature + warped_view_feature
                ave_feature2 = ave_feature2 + tf.square(warped_view_feature)
            ave_feature = ave_feature / FLAGS.view_num
            ave_feature2 = ave_feature2 / FLAGS.view_num
            cost = ave_feature2 - tf.square(ave_feature)
            depth_costs.append(cost)


        cost_volume = tf.stack(depth_costs, axis=1)

    # import ipdb; ipdb.set_trace()
    # filtered cost volume, size of (B, D, H, W, 1)
    if is_master_gpu:
        filtered_cost_volume_tower = RegNetUS0({'data': cost_volume}, is_training=True, reuse=False)
    else:
        filtered_cost_volume_tower = RegNetUS0({'data': cost_volume}, is_training=True, reuse=True)
    filtered_cost_volume = tf.squeeze(filtered_cost_volume_tower.get_output(), axis=-1)

    # depth map by softArgmin
    with tf.name_scope('soft_arg_min'):
        # probability volume by soft max
        probability_volume = tf.nn.softmax(
            tf.scalar_mul(-1, filtered_cost_volume), axis=1, name='prob_volume')
        # depth image by soft argmin
        volume_shape = tf.shape(probability_volume)


        soft_2d = []
        for i in range(FLAGS.batch_size):
            soft_1d = tf.linspace(depth_start[i], depth_end[i], tf.cast(depth_num, tf.int32))
            # soft_1d = tf.Print(soft_1d, [tf.shape(soft_1d)], "soft_1d ",summarize=6)
            soft_2d.append(soft_1d)
        # soft_2d = tf.Print(soft_2d, [tf.shape(soft_2d)], "soft_2d ",summarize=6)
        soft_2d = tf.reshape(tf.stack(soft_2d, axis=0), [volume_shape[0], volume_shape[1], 1, 1])
        # soft_2d = tf.Print(soft_2d, [tf.shape(soft_2d)], "soft_2d ",summarize=6)
        soft_4d = tf.tile(soft_2d, [1, 1, volume_shape[2], volume_shape[3]])
        # soft_4d = tf.Print(soft_4d, [tf.shape(soft_4d)], "soft_4d ",summarize=6)
        
        # probability_volume = tf.Print(probability_volume, [depth_end,tf.reduce_min(soft_4d),tf.reduce_max(probability_volume),tf.reduce_min(soft_4d),tf.reduce_max(soft_4d)], "soft_4d ",summarize=5)
        
       
        estimated_depth_map = tf.reduce_sum(soft_4d * probability_volume, axis=1)
        estimated_depth_map = tf.expand_dims(estimated_depth_map, axis=3)

    # probability map
    prob_map = get_propability_map(probability_volume, estimated_depth_map, depth_start, depth_interval)

    return estimated_depth_map, prob_map  # , filtered_depth_map, probability_volume




def inference_with_init(images, cams, depth_num, depth_start, depth_interval, colmap_img,prob_im, is_master_gpu=True):
    """ infer depth image from multi-view images and cameras """

    # dynamic gpu params
    depth_end = depth_start + (tf.cast(depth_num, tf.float32) - 1) * depth_interval

    # reference image
    ref_image = tf.squeeze(tf.slice(images, [0, 0, 0, 0, 0], [-1, 1, -1, -1, 3]), axis=1)
    ref_cam = tf.squeeze(tf.slice(cams, [0, 0, 0, 0, 0], [-1, 1, 2, 4, 4]), axis=1)

    # image feature extraction
    if is_master_gpu:
        ref_tower = UNetDS2GN({'data': ref_image}, is_training=True, reuse=False)
    else:
        ref_tower = UNetDS2GN({'data': ref_image}, is_training=True, reuse=True)
    view_towers = []
    for view in range(1, FLAGS.view_num):
        view_image = tf.squeeze(tf.slice(images, [0, view, 0, 0, 0], [-1, 1, -1, -1, -1]), axis=1)
        view_tower = UNetDS2GN({'data': view_image}, is_training=True, reuse=True)
        view_towers.append(view_tower)


    # get all homographies
    view_homographies = []
    tmp_homographies = []
    for view in range(1, FLAGS.view_num):
        view_cam = tf.squeeze(tf.slice(cams, [0, view, 0, 0, 0], [-1, 1, 2, 4, 4]), axis=1)

        h = tf.shape(colmap_img)[0]
        w = tf.shape(colmap_img)[1]
        # homographiesOrig = get_homographies(ref_cam, view_cam, depth_num=depth_num,
        #                                 depth_start=depth_start, depth_interval=depth_interval)

        homographies = get_homographies_initialized(ref_cam, view_cam, depth_num=depth_num,
                                        depth_start=depth_start, depth_interval=depth_interval,
                                        init_depth=colmap_img, prob_depth=prob_im)

        view_homographies.append(homographies)
        # tmp_homographies.append(homographiesOrig)
    
    # warped0= []
    # warped1= []
    # warped2= []
    # build cost volume by differentialble homography
    with tf.name_scope('cost_volume_homography'):
        depth_costs = []
        for d in range(depth_num):
            # compute cost (variation metric)
            ave_feature = ref_tower.get_output()
            ave_feature2 = tf.square(ref_tower.get_output())
            for view in range(0, FLAGS.view_num - 1):
                homography = tf.slice(view_homographies[view], begin=[0, 0,0,d, 0, 0], size=[-1,-1,-1,1, 3, 3])
                # homography = tf.Print(homography, [tf.shape(view_homographies[view]),tf.shape(homography)]," view_homographies[view] ",summarize=6)
                homography = tf.squeeze(homography, axis=3)
                warped_view_feature = homography_warping_multi(view_towers[view].get_output(), homography)              

                
                ave_feature = ave_feature + warped_view_feature
                ave_feature2 = ave_feature2 + tf.square(warped_view_feature)
            ave_feature = ave_feature / FLAGS.view_num
            ave_feature2 = ave_feature2 / FLAGS.view_num
            cost = ave_feature2 - tf.square(ave_feature)
            depth_costs.append(cost)
        cost_volume = tf.stack(depth_costs, axis=1)

    # filtered cost volume, size of (B, D, H, W, 1)
    if is_master_gpu:
        filtered_cost_volume_tower = RegNetUS0({'data': cost_volume}, is_training=True, reuse=False)
    else:
        filtered_cost_volume_tower = RegNetUS0({'data': cost_volume}, is_training=True, reuse=True)
    filtered_cost_volume = tf.squeeze(filtered_cost_volume_tower.get_output(), axis=-1)

    # depth map by softArgmin
    with tf.name_scope('soft_arg_min'):
        # probability volume by soft max
        probability_volume = tf.nn.softmax(
            tf.scalar_mul(-1, filtered_cost_volume), axis=1, name='prob_volume')
        volume_shape = tf.shape(probability_volume)

        h = tf.shape(colmap_img)[1]
        w = tf.shape(colmap_img)[2]

        min_depth = depth_start
        max_depth = depth_start + tf.cast(depth_num, tf.float32)  * depth_interval

        L = (max_depth - min_depth) * (1.5-prob_im)
        min_depth_corrected = tf.maximum (min_depth, colmap_img - L) # h x w
        max_depth_corrected = tf.minimum (max_depth, colmap_img + L) # h x w
        min_depth_corrected = tf.transpose(min_depth_corrected, perm=[ 0, 3, 1, 2])
        max_depth_corrected = tf.transpose(max_depth_corrected, perm=[ 0, 3, 1, 2])
        # max_depth_corrected = tf.Print(max_depth_corrected, [tf.shape(max_depth_corrected)], "max_depth_corrected ",summarize=6)
        depths_start = tf.tile(min_depth_corrected,[1,depth_num,1,1])


        depthInt = tf.tile(tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.cast(tf.range(depth_num), tf.float32), axis=0), axis=0), axis=0),[1,h,w,1])
        depthInt = tf.transpose(depthInt, perm=[ 0, 3, 1, 2])
        # depths_start = tf.Print(depths_start,[tf.shape(depths_start),tf.shape(min_depth_corrected),tf.shape(depthInt)], "depths_start ",summarize=259)

        # min_depth_corrected =   init_depth - 100.0  # h x w
        # max_depth_corrected =  init_depth + 100.0 # h x w
        # depths_start = tf.tile(init_depth-100.0,[1,1,1,depth_num])
        
        depth_intervals = (max_depth_corrected - min_depth_corrected) / tf.cast(depth_num,tf.float32)
        # depth_intervals = tf.Print(depth_intervals,[tf.shape(depth_intervals)], "depth_intervals ",summarize=259)

        # depth_intervals = tf.cast(depth_interval, tf.float32) * tf.ones_like(min_depth_corrected)

        # depth_intervals = tf.squeeze(depth_intervals,axis=3)
        # depth_intervals = tf.expand_dims(depth_intervals,axis=0)
        soft_4d = depths_start + depthInt * depth_intervals
        # soft_4d = tf.tile(depth, [FLAGS.batch_size,1,1,1])
        
        # soft_4d = tf.Print(soft_4d, [
        #     tf.reduce_sum(probability_volume, axis=1),
        #     tf.reduce_sum(soft_4d * probability_volume, axis=1)], "soft_4d ",summarize=100)
        # soft_4d = tf.Print(soft_4d, [tf.shape(depth_intervals),
        #     depth_intervals[:,:,tf.cast(h/3,tf.int32), tf.cast(w/2,tf.int32)]], "depth_intervals ",summarize=100)

        soft_2d = []
        for i in range(FLAGS.batch_size):
            soft_1d = tf.linspace(depth_start[i], depth_end[i], tf.cast(depth_num, tf.int32))
            soft_2d.append(soft_1d)
        soft_2d = tf.reshape(tf.stack(soft_2d, axis=0), [volume_shape[0], volume_shape[1], 1, 1])
        soft_4dold = tf.tile(soft_2d, [1, 1, volume_shape[2], volume_shape[3]])
        estimated_depth_map = tf.reduce_sum(soft_4d * probability_volume, axis=1)
        estimated_depth_map = tf.expand_dims(estimated_depth_map, axis=3)

    # probability map
    prob_map = get_propability_map(probability_volume, estimated_depth_map, depth_start, depth_interval)

    return estimated_depth_map, prob_map#, warped0[10], warped1[10], warped2[10]  # , filtered_depth_map, probability_volume


def inference_mem_with_init(images, cams, depth_num, depth_start, depth_interval, colmap_img,prob_im, is_master_gpu=True):
    """ infer depth image from multi-view images and cameras """

    # dynamic gpu params
    depth_end = depth_start + (tf.cast(depth_num, tf.float32) - 1) * depth_interval
    feature_c = 32
    feature_h = FLAGS.max_h / 4
    feature_w = FLAGS.max_w / 4

    # reference image
    ref_image = tf.squeeze(tf.slice(images, [0, 0, 0, 0, 0], [-1, 1, -1, -1, 3]), axis=1)
    ref_cam = tf.squeeze(tf.slice(cams, [0, 0, 0, 0, 0], [-1, 1, 2, 4, 4]), axis=1)

    # image feature extraction    
    if is_master_gpu:
        ref_tower = UNetDS2GN({'data': ref_image}, is_training=True, reuse=False)
    else:
        ref_tower = UNetDS2GN({'data': ref_image}, is_training=True, reuse=True)
    ref_feature = ref_tower.get_output()
    ref_feature2 = tf.square(ref_feature)

    view_features = []
    for view in range(1, FLAGS.view_num):
        view_image = tf.squeeze(tf.slice(images, [0, view, 0, 0, 0], [-1, 1, -1, -1, -1]), axis=1)
        view_tower = UNetDS2GN({'data': view_image}, is_training=True, reuse=True)
        view_features.append(view_tower.get_output())
    view_features = tf.stack(view_features, axis=0)

    # get all homographies
    view_homographies = []
    for view in range(1, FLAGS.view_num):
        view_cam = tf.squeeze(tf.slice(cams, [0, view, 0, 0, 0], [-1, 1, 2, 4, 4]), axis=1)
        # homographies = get_homographies(ref_cam, view_cam, depth_num=depth_num,
        #                                 depth_start=depth_start, depth_interval=depth_interval)
        
        homographies = get_homographies_initialized(ref_cam, view_cam, depth_num=depth_num,
                                depth_start=depth_start, depth_interval=depth_interval,
                                init_depth=colmap_img, prob_depth=prob_im)


        view_homographies.append(homographies)
    view_homographies = tf.stack(view_homographies, axis=0)

    # build cost volume by differentialble homography
    with tf.name_scope('cost_volume_homography'):
        depth_costs = []

        for d in range(depth_num):
            # compute cost (standard deviation feature)
            ave_feature = tf.Variable(tf.zeros(
                [FLAGS.batch_size, feature_h, feature_w, feature_c]),
                name='ave', trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
            ave_feature2 = tf.Variable(tf.zeros(
                [FLAGS.batch_size, feature_h, feature_w, feature_c]),
                name='ave2', trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
            ave_feature = tf.assign(ave_feature, ref_feature)
            ave_feature2 = tf.assign(ave_feature2, ref_feature2)

            def body(view, ave_feature, ave_feature2):
                """Loop body."""
                # homography = tf.slice(view_homographies[view], begin=[0, d, 0, 0], size=[-1, 1, 3, 3])
                homography = tf.slice(view_homographies[view], begin=[0, 0,0,d, 0, 0], size=[-1,-1,-1,1, 3, 3])
                # homography = tf.squeeze(homography, axis=1)
                homography = tf.squeeze(homography, axis=3)
                # warped_view_feature = homography_warping(view_features[view], homography)
                # warped_view_feature = tf_transform_homography(view_features[view], homography)
                warped_view_feature = homography_warping_multi(view_towers[view].get_output(), homography)              
                ave_feature = tf.assign_add(ave_feature, warped_view_feature)
                ave_feature2 = tf.assign_add(ave_feature2, tf.square(warped_view_feature))
                view = tf.add(view, 1)
                return view, ave_feature, ave_feature2

            view = tf.constant(0)
            cond = lambda view, *_: tf.less(view, FLAGS.view_num - 1)
            _, ave_feature, ave_feature2 = tf.while_loop(
                cond, body, [view, ave_feature, ave_feature2], back_prop=False, parallel_iterations=1)

            ave_feature = tf.assign(ave_feature, tf.square(ave_feature) / (FLAGS.view_num * FLAGS.view_num))
            ave_feature2 = tf.assign(ave_feature2, ave_feature2 / FLAGS.view_num - ave_feature)
            depth_costs.append(ave_feature2)
        cost_volume = tf.stack(depth_costs, axis=1)

    # filtered cost volume, size of (B, D, H, W, 1)
    if is_master_gpu:
        filtered_cost_volume_tower = RegNetUS0({'data': cost_volume}, is_training=True, reuse=False)
    else:
        filtered_cost_volume_tower = RegNetUS0({'data': cost_volume}, is_training=True, reuse=True)
    filtered_cost_volume = tf.squeeze(filtered_cost_volume_tower.get_output(), axis=-1)

    # depth map by softArgmin
    with tf.name_scope('soft_arg_min'):
        # probability volume by soft max
        probability_volume = tf.nn.softmax(tf.scalar_mul(-1, filtered_cost_volume),
                                           axis=1, name='prob_volume')

        # depth image by soft argmin
        volume_shape = tf.shape(probability_volume)


        h = tf.shape(colmap_img)[1]
        w = tf.shape(colmap_img)[2]

        min_depth = depth_start
        max_depth = depth_start + tf.cast(depth_num, tf.float32)  * depth_interval

        L = (max_depth - min_depth) * (1.5-prob_im)
        min_depth_corrected = tf.maximum (min_depth, colmap_img - L) # h x w
        max_depth_corrected = tf.minimum (max_depth, colmap_img + L) # h x w
        min_depth_corrected = tf.transpose(min_depth_corrected, perm=[ 0, 3, 1, 2])
        max_depth_corrected = tf.transpose(max_depth_corrected, perm=[ 0, 3, 1, 2])
        # max_depth_corrected = tf.Print(max_depth_corrected, [tf.shape(max_depth_corrected)], "max_depth_corrected ",summarize=6)
        depths_start = tf.tile(min_depth_corrected,[1,depth_num,1,1])


        depthInt = tf.tile(tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.cast(tf.range(depth_num), tf.float32), axis=0), axis=0), axis=0),[1,h,w,1])
        depthInt = tf.transpose(depthInt, perm=[ 0, 3, 1, 2])
        # depths_start = tf.Print(depths_start,[tf.shape(depths_start),tf.shape(min_depth_corrected),tf.shape(depthInt)], "depths_start ",summarize=259)

        # min_depth_corrected =   init_depth - 100.0  # h x w
        # max_depth_corrected =  init_depth + 100.0 # h x w
        # depths_start = tf.tile(init_depth-100.0,[1,1,1,depth_num])
        
        depth_intervals = (max_depth_corrected - min_depth_corrected) / tf.cast(depth_num,tf.float32)
        # depth_intervals = tf.Print(depth_intervals,[tf.shape(depth_intervals)], "depth_intervals ",summarize=259)

        # depth_intervals = tf.cast(depth_interval, tf.float32) * tf.ones_like(min_depth_corrected)

        # depth_intervals = tf.squeeze(depth_intervals,axis=3)
        # depth_intervals = tf.expand_dims(depth_intervals,axis=0)
        soft_4d = depths_start + depthInt * depth_intervals
        # soft_4d = tf.tile(depth, [FLAGS.batch_size,1,1,1])
        
        # soft_4d = tf.Print(soft_4d, [
        #     tf.reduce_sum(probability_volume, axis=1),
        #     tf.reduce_sum(soft_4d * probability_volume, axis=1)], "soft_4d ",summarize=100)
        # soft_4d = tf.Print(soft_4d, [tf.shape(depth_intervals),
        #     depth_intervals[:,:,tf.cast(h/3,tf.int32), tf.cast(w/2,tf.int32)]], "depth_intervals ",summarize=100)

        soft_2d = []
        for i in range(FLAGS.batch_size):
            soft_1d = tf.linspace(depth_start[i], depth_end[i], tf.cast(depth_num, tf.int32))
            soft_2d.append(soft_1d)
        soft_2d = tf.reshape(tf.stack(soft_2d, axis=0), [volume_shape[0], volume_shape[1], 1, 1])
        soft_4dold = tf.tile(soft_2d, [1, 1, volume_shape[2], volume_shape[3]])
        estimated_depth_map = tf.reduce_sum(soft_4d * probability_volume, axis=1)
        estimated_depth_map = tf.expand_dims(estimated_depth_map, axis=3)

    # probability map
    prob_map = get_propability_map(probability_volume, estimated_depth_map, depth_start, depth_interval)

    # return filtered_depth_map, 
    return estimated_depth_map, prob_map


def inference_mem(images, cams, depth_num, depth_start, depth_interval, is_master_gpu=True):
    """ infer depth image from multi-view images and cameras """

    # dynamic gpu params
    depth_end = depth_start + (tf.cast(depth_num, tf.float32) - 1) * depth_interval
    feature_c = 32
    feature_h = FLAGS.max_h / 4
    feature_w = FLAGS.max_w / 4

    # reference image
    ref_image = tf.squeeze(tf.slice(images, [0, 0, 0, 0, 0], [-1, 1, -1, -1, 3]), axis=1)
    ref_cam = tf.squeeze(tf.slice(cams, [0, 0, 0, 0, 0], [-1, 1, 2, 4, 4]), axis=1)

    # image feature extraction    
    if is_master_gpu:
        ref_tower = UNetDS2GN({'data': ref_image}, is_training=True, reuse=False)
    else:
        ref_tower = UNetDS2GN({'data': ref_image}, is_training=True, reuse=True)
    ref_feature = ref_tower.get_output()
    ref_feature2 = tf.square(ref_feature)

    view_features = []
    for view in range(1, FLAGS.view_num):
        view_image = tf.squeeze(tf.slice(images, [0, view, 0, 0, 0], [-1, 1, -1, -1, -1]), axis=1)
        view_tower = UNetDS2GN({'data': view_image}, is_training=True, reuse=True)
        view_features.append(view_tower.get_output())
    view_features = tf.stack(view_features, axis=0)

    # get all homographies
    view_homographies = []
    for view in range(1, FLAGS.view_num):
        view_cam = tf.squeeze(tf.slice(cams, [0, view, 0, 0, 0], [-1, 1, 2, 4, 4]), axis=1)
        homographies = get_homographies(ref_cam, view_cam, depth_num=depth_num,
                                        depth_start=depth_start, depth_interval=depth_interval)
        view_homographies.append(homographies)
    view_homographies = tf.stack(view_homographies, axis=0)

    # build cost volume by differentialble homography
    with tf.name_scope('cost_volume_homography'):
        depth_costs = []

        for d in range(depth_num):
            # compute cost (standard deviation feature)
            ave_feature = tf.Variable(tf.zeros(
                [FLAGS.batch_size, feature_h, feature_w, feature_c]),
                name='ave', trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
            ave_feature2 = tf.Variable(tf.zeros(
                [FLAGS.batch_size, feature_h, feature_w, feature_c]),
                name='ave2', trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
            ave_feature = tf.assign(ave_feature, ref_feature)
            ave_feature2 = tf.assign(ave_feature2, ref_feature2)

            def body(view, ave_feature, ave_feature2):
                """Loop body."""
                homography = tf.slice(view_homographies[view], begin=[0, d, 0, 0], size=[-1, 1, 3, 3])
                homography = tf.squeeze(homography, axis=1)
                # warped_view_feature = homography_warping(view_features[view], homography)
                warped_view_feature = tf_transform_homography(view_features[view], homography)
                ave_feature = tf.assign_add(ave_feature, warped_view_feature)
                ave_feature2 = tf.assign_add(ave_feature2, tf.square(warped_view_feature))
                view = tf.add(view, 1)
                return view, ave_feature, ave_feature2

            view = tf.constant(0)
            cond = lambda view, *_: tf.less(view, FLAGS.view_num - 1)
            _, ave_feature, ave_feature2 = tf.while_loop(
                cond, body, [view, ave_feature, ave_feature2], back_prop=False, parallel_iterations=1)

            ave_feature = tf.assign(ave_feature, tf.square(ave_feature) / (FLAGS.view_num * FLAGS.view_num))
            ave_feature2 = tf.assign(ave_feature2, ave_feature2 / FLAGS.view_num - ave_feature)
            depth_costs.append(ave_feature2)
        cost_volume = tf.stack(depth_costs, axis=1)

    # filtered cost volume, size of (B, D, H, W, 1)
    if is_master_gpu:
        filtered_cost_volume_tower = RegNetUS0({'data': cost_volume}, is_training=True, reuse=False)
    else:
        filtered_cost_volume_tower = RegNetUS0({'data': cost_volume}, is_training=True, reuse=True)
    filtered_cost_volume = tf.squeeze(filtered_cost_volume_tower.get_output(), axis=-1)

    # depth map by softArgmin
    with tf.name_scope('soft_arg_min'):
        # probability volume by soft max
        probability_volume = tf.nn.softmax(tf.scalar_mul(-1, filtered_cost_volume),
                                           axis=1, name='prob_volume')

        # depth image by soft argmin
        volume_shape = tf.shape(probability_volume)
        soft_2d = []
        for i in range(FLAGS.batch_size):
            soft_1d = tf.linspace(depth_start[i], depth_end[i], tf.cast(depth_num, tf.int32))
            soft_2d.append(soft_1d)
        soft_2d = tf.reshape(tf.stack(soft_2d, axis=0), [volume_shape[0], volume_shape[1], 1, 1])
        soft_4d = tf.tile(soft_2d, [1, 1, volume_shape[2], volume_shape[3]])
        estimated_depth_map = tf.reduce_sum(soft_4d * probability_volume, axis=1)
        estimated_depth_map = tf.expand_dims(estimated_depth_map, axis=3)

    # probability map
    prob_map = get_propability_map(probability_volume, estimated_depth_map, depth_start, depth_interval)

    # return filtered_depth_map, 
    return estimated_depth_map, prob_map


def inference_prob_recurrent(images, cams, depth_num, depth_start, depth_interval, is_master_gpu=True):
    """ infer disparity image from stereo images and cameras """

    # dynamic gpu params
    depth_end = depth_start + (tf.cast(depth_num, tf.float32) - 1) * depth_interval

    # reference image
    ref_image = tf.squeeze(tf.slice(images, [0, 0, 0, 0, 0], [-1, 1, -1, -1, 3]), axis=1)
    ref_cam = tf.squeeze(tf.slice(cams, [0, 0, 0, 0, 0], [-1, 1, 2, 4, 4]), axis=1)

    # image feature extraction    
    if is_master_gpu:
        ref_tower = UNetDS2GN({'data': ref_image}, is_training=True, reuse=False)
    else:
        ref_tower = UNetDS2GN({'data': ref_image}, is_training=True, reuse=True)
    view_towers = []
    for view in range(1, FLAGS.view_num):
        view_image = tf.squeeze(tf.slice(images, [0, view, 0, 0, 0], [-1, 1, -1, -1, -1]), axis=1)
        view_tower = UNetDS2GN({'data': view_image}, is_training=True, reuse=True)
        view_towers.append(view_tower)

    # get all homographies
    view_homographies = []
    for view in range(1, FLAGS.view_num):
        view_cam = tf.squeeze(tf.slice(cams, [0, view, 0, 0, 0], [-1, 1, 2, 4, 4]), axis=1)
        homographies = get_homographies(ref_cam, view_cam, depth_num=depth_num,
                                        depth_start=depth_start, depth_interval=depth_interval)
        view_homographies.append(homographies)

    gru1_filters = 16
    gru2_filters = 4
    gru3_filters = 2
    feature_shape = [FLAGS.batch_size, FLAGS.max_h / 4, FLAGS.max_w / 4, 32]
    gru_input_shape = [feature_shape[1], feature_shape[2]]
    state1 = tf.zeros([FLAGS.batch_size, feature_shape[1], feature_shape[2], gru1_filters])
    state2 = tf.zeros([FLAGS.batch_size, feature_shape[1], feature_shape[2], gru2_filters])
    state3 = tf.zeros([FLAGS.batch_size, feature_shape[1], feature_shape[2], gru3_filters])
    conv_gru1 = ConvGRUCell(shape=gru_input_shape, kernel=[3, 3], filters=gru1_filters)
    conv_gru2 = ConvGRUCell(shape=gru_input_shape, kernel=[3, 3], filters=gru2_filters)
    conv_gru3 = ConvGRUCell(shape=gru_input_shape, kernel=[3, 3], filters=gru3_filters)

    exp_div = tf.zeros([FLAGS.batch_size, feature_shape[1], feature_shape[2], 1])
    soft_depth_map = tf.zeros([FLAGS.batch_size, feature_shape[1], feature_shape[2], 1])

    with tf.name_scope('cost_volume_homography'):

        # forward cost volume
        depth_costs = []
        for d in range(depth_num):

            # compute cost (variation metric)
            ave_feature = ref_tower.get_output()
            ave_feature2 = tf.square(ref_tower.get_output())

            for view in range(0, FLAGS.view_num - 1):
                homography = tf.slice(
                    view_homographies[view], begin=[0, d, 0, 0], size=[-1, 1, 3, 3])
                homography = tf.squeeze(homography, axis=1)
                # warped_view_feature = homography_warping(view_towers[view].get_output(), homography)
                warped_view_feature = tf_transform_homography(view_towers[view].get_output(), homography)
                ave_feature = ave_feature + warped_view_feature
                ave_feature2 = ave_feature2 + tf.square(warped_view_feature)
            ave_feature = ave_feature / FLAGS.view_num
            ave_feature2 = ave_feature2 / FLAGS.view_num
            cost = ave_feature2 - tf.square(ave_feature)

            # gru
            reg_cost1, state1 = conv_gru1(-cost, state1, scope='conv_gru1')
            reg_cost2, state2 = conv_gru2(reg_cost1, state2, scope='conv_gru2')
            reg_cost3, state3 = conv_gru3(reg_cost2, state3, scope='conv_gru3')
            reg_cost = tf.layers.conv2d(
                reg_cost3, 1, 3, padding='same', reuse=tf.AUTO_REUSE, name='prob_conv')
            depth_costs.append(reg_cost)

        prob_volume = tf.stack(depth_costs, axis=1)
        prob_volume = tf.nn.softmax(prob_volume, axis=1, name='prob_volume')

    return prob_volume


def inference_winner_take_all(images, cams, depth_num, depth_start, depth_end,
                              is_master_gpu=True, reg_type='GRU', inverse_depth=False):
    """ infer disparity image from stereo images and cameras """

    if not inverse_depth:
        depth_interval = (depth_end - depth_start) / (tf.cast(depth_num, tf.float32) - 1)

    # reference image
    ref_image = tf.squeeze(tf.slice(images, [0, 0, 0, 0, 0], [-1, 1, -1, -1, 3]), axis=1)
    ref_cam = tf.squeeze(tf.slice(cams, [0, 0, 0, 0, 0], [-1, 1, 2, 4, 4]), axis=1)

    # image feature extraction    
    if is_master_gpu:
        ref_tower = UNetDS2GN({'data': ref_image}, is_training=True, reuse=False)
    else:
        ref_tower = UNetDS2GN({'data': ref_image}, is_training=True, reuse=True)
    view_towers = []
    for view in range(1, FLAGS.view_num):
        view_image = tf.squeeze(tf.slice(images, [0, view, 0, 0, 0], [-1, 1, -1, -1, -1]), axis=1)
        view_tower = UNetDS2GN({'data': view_image}, is_training=True, reuse=True)
        view_towers.append(view_tower)

    # get all homographies
    view_homographies = []
    for view in range(1, FLAGS.view_num):
        view_cam = tf.squeeze(tf.slice(cams, [0, view, 0, 0, 0], [-1, 1, 2, 4, 4]), axis=1)
        if inverse_depth:
            homographies = get_homographies_inv_depth(ref_cam, view_cam, depth_num=depth_num,
                                                      depth_start=depth_start, depth_end=depth_end)
        else:
            homographies = get_homographies(ref_cam, view_cam, depth_num=depth_num,
                                            depth_start=depth_start, depth_interval=depth_interval)
        view_homographies.append(homographies)

    # gru unit
    gru1_filters = 16
    gru2_filters = 4
    gru3_filters = 2
    feature_shape = [FLAGS.batch_size, FLAGS.max_h / 4, FLAGS.max_w / 4, 32]
    gru_input_shape = [feature_shape[1], feature_shape[2]]
    state1 = tf.zeros([FLAGS.batch_size, feature_shape[1], feature_shape[2], gru1_filters])
    state2 = tf.zeros([FLAGS.batch_size, feature_shape[1], feature_shape[2], gru2_filters])
    state3 = tf.zeros([FLAGS.batch_size, feature_shape[1], feature_shape[2], gru3_filters])
    conv_gru1 = ConvGRUCell(shape=gru_input_shape, kernel=[3, 3], filters=gru1_filters)
    conv_gru2 = ConvGRUCell(shape=gru_input_shape, kernel=[3, 3], filters=gru2_filters)
    conv_gru3 = ConvGRUCell(shape=gru_input_shape, kernel=[3, 3], filters=gru3_filters)

    # initialize variables
    exp_sum = tf.Variable(tf.zeros(
        [FLAGS.batch_size, feature_shape[1], feature_shape[2], 1]),
        name='exp_sum', trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
    depth_image = tf.Variable(tf.zeros(
        [FLAGS.batch_size, feature_shape[1], feature_shape[2], 1]),
        name='depth_image', trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
    max_prob_image = tf.Variable(tf.zeros(
        [FLAGS.batch_size, feature_shape[1], feature_shape[2], 1]),
        name='max_prob_image', trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
    init_map = tf.zeros([FLAGS.batch_size, feature_shape[1], feature_shape[2], 1])

    # define winner take all loop
    def body(depth_index, state1, state2, state3, depth_image, max_prob_image, exp_sum, incre):
        """Loop body."""

        # calculate cost 
        ave_feature = ref_tower.get_output()
        ave_feature2 = tf.square(ref_tower.get_output())
        for view in range(0, FLAGS.view_num - 1):
            homographies = view_homographies[view]
            homographies = tf.transpose(homographies, perm=[1, 0, 2, 3])
            homography = homographies[depth_index]
            # warped_view_feature = homography_warping(view_towers[view].get_output(), homography)
            warped_view_feature = tf_transform_homography(view_towers[view].get_output(), homography)
            ave_feature = ave_feature + warped_view_feature
            ave_feature2 = ave_feature2 + tf.square(warped_view_feature)
        ave_feature = ave_feature / FLAGS.view_num
        ave_feature2 = ave_feature2 / FLAGS.view_num
        cost = ave_feature2 - tf.square(ave_feature)
        cost.set_shape([FLAGS.batch_size, feature_shape[1], feature_shape[2], 32])

        # gru
        reg_cost1, state1 = conv_gru1(-cost, state1, scope='conv_gru1')
        reg_cost2, state2 = conv_gru2(reg_cost1, state2, scope='conv_gru2')
        reg_cost3, state3 = conv_gru3(reg_cost2, state3, scope='conv_gru3')
        reg_cost = tf.layers.conv2d(
            reg_cost3, 1, 3, padding='same', reuse=tf.AUTO_REUSE, name='prob_conv')
        prob = tf.exp(reg_cost)

        # index
        d_idx = tf.cast(depth_index, tf.float32)
        if inverse_depth:
            inv_depth_start = tf.div(1.0, depth_start)
            inv_depth_end = tf.div(1.0, depth_end)
            inv_interval = (inv_depth_start - inv_depth_end) / (tf.cast(depth_num, 'float32') - 1)
            inv_depth = inv_depth_start - d_idx * inv_interval
            depth = tf.div(1.0, inv_depth)
        else:
            depth = depth_start + d_idx * depth_interval
        temp_depth_image = tf.reshape(depth, [FLAGS.batch_size, 1, 1, 1])
        temp_depth_image = tf.tile(
            temp_depth_image, [1, feature_shape[1], feature_shape[2], 1])

        # update the best
        update_flag_image = tf.cast(tf.less(max_prob_image, prob), dtype='float32')
        new_max_prob_image = update_flag_image * prob + (1 - update_flag_image) * max_prob_image
        new_depth_image = update_flag_image * temp_depth_image + (1 - update_flag_image) * depth_image
        max_prob_image = tf.assign(max_prob_image, new_max_prob_image)
        depth_image = tf.assign(depth_image, new_depth_image)

        # update counter
        exp_sum = tf.assign_add(exp_sum, prob)
        depth_index = tf.add(depth_index, incre)

        return depth_index, state1, state2, state3, depth_image, max_prob_image, exp_sum, incre

    # run forward loop
    exp_sum = tf.assign(exp_sum, init_map)
    depth_image = tf.assign(depth_image, init_map)
    max_prob_image = tf.assign(max_prob_image, init_map)
    depth_index = tf.constant(0)
    incre = tf.constant(1)
    cond = lambda depth_index, *_: tf.less(depth_index, depth_num)
    _, state1, state2, state3, depth_image, max_prob_image, exp_sum, incre = tf.while_loop(
        cond, body
        , [depth_index, state1, state2, state3, depth_image, max_prob_image, exp_sum, incre]
        , back_prop=False, parallel_iterations=1)

    # get output
    forward_exp_sum = exp_sum + 1e-7
    forward_depth_map = depth_image
    return forward_depth_map, max_prob_image / forward_exp_sum


def depth_refine(init_depth_map, image, depth_num, depth_start, depth_interval, colmap_image=None, depth_start_colmap=None, prob_image=None,
 is_master_gpu=True):
    """ refine depth image with the image """

    # normalization parameters
    depth_shape = tf.shape(init_depth_map)
    depth_end = depth_start + (tf.cast(depth_num, tf.float32) - 1) * depth_interval
    depth_start_mat = tf.tile(tf.reshape(
        depth_start, [depth_shape[0], 1, 1, 1]), [1, depth_shape[1], depth_shape[2], 1])
    depth_end_mat = tf.tile(tf.reshape(
        depth_end, [depth_shape[0], 1, 1, 1]), [1, depth_shape[1], depth_shape[2], 1])
    depth_scale_mat = depth_end_mat - depth_start_mat

    # normalize depth map (to 0~1)
    init_norm_depth_map = tf.div(init_depth_map - depth_start_mat, depth_scale_mat)

    # resize normalized image to the same size of depth image
    resized_image = tf.image.resize_bilinear(image, [depth_shape[1], depth_shape[2]])

    if colmap_image is not None:
        # resize normalized image to the same size of depth image
        resized_colmap_image = tf.image.resize_nearest_neighbor(colmap_image, [depth_shape[1], depth_shape[2]])
        resized_prob_image = tf.image.resize_nearest_neighbor(prob_image, [depth_shape[1], depth_shape[2]])

        # normalization parameters COLMAP
        depth_end_colmap = depth_start_colmap + (tf.cast(depth_num, tf.float32) - 1) * depth_interval
        depth_start_mat_colmap = tf.tile(tf.reshape(
            depth_start_colmap, [depth_shape[0], 1, 1, 1]), [1, depth_shape[1], depth_shape[2], 1])
        depth_end_mat_colmap = tf.tile(tf.reshape(
            depth_end_colmap, [depth_shape[0], 1, 1, 1]), [1, depth_shape[1], depth_shape[2], 1])
        depth_scale_mat_colmap = depth_end_mat_colmap - depth_start_mat_colmap
        # normalize depth map (to 0~1)
        norm_colmap_image = tf.div(resized_colmap_image - depth_start_mat_colmap, depth_scale_mat_colmap)


    # refinement network
    if is_master_gpu:

        if colmap_image is not None:
            norm_depth_tower = RefineNet({'color_image': resized_image, 'depth_image': init_norm_depth_map,
                                          'colmap_image': norm_colmap_image, 'prob_image': resized_prob_image
                                          },
                                         is_training=True, reuse=False)
        else:
            norm_depth_tower = RefineNet({'color_image': resized_image, 'depth_image': init_norm_depth_map},
                                         is_training=True, reuse=False)
    else:
        norm_depth_tower = RefineNet({'color_image': resized_image, 'depth_image': init_norm_depth_map},
                                     is_training=True, reuse=True)
    norm_depth_map = norm_depth_tower.get_output()

    # denormalize depth map
    refined_depth_map = tf.multiply(norm_depth_map, depth_scale_mat) + depth_start_mat

    return refined_depth_map
