"""
    PointNet++ Model for point clouds classification
"""

import os
import sys
import tensorflow as tf
import tf_util
from pointnet_util import pointnet_sa_module

BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))


def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    attachment_labels_pl = tf.placeholder(tf.int32, shape=(batch_size, 128))
    type_labels_pl = tf.placeholder(tf.int32, shape=(batch_size, 128))
    orientation_labels_pl = tf.placeholder(tf.int32, shape=(batch_size, 128))
    surface_labels_pl = tf.placeholder(tf.int32, shape=(batch_size, 128))
    startstage_labels_pl = tf.placeholder(tf.int32, shape=(batch_size, 128))

    return pointclouds_pl, attachment_labels_pl,type_labels_pl,orientation_labels_pl,surface_labels_pl,startstage_labels_pl


def get_model(point_cloud, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    l0_xyz = point_cloud
    l0_points = None
    end_points['l0_xyz'] = l0_xyz

    # Set abstraction layers
    # Note: When using NCHW for layer 2, we see increased GPU memory usage (in TF1.4).
    # So we only use NCHW for layer 1 until this issue can be resolved.

    l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=4096, radius=0.2, nsample=32, mlp=[64, 64, 128], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer1', use_nchw=True)
    l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=1024, radius=0.4, nsample=64, mlp=[128, 128, 256], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer2')
    l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=None, radius=None, nsample=None, mlp=[256, 512, 1024], mlp2=None, group_all=True, is_training=is_training, bn_decay=bn_decay, scope='layer3')
    
    # Fully connected layers
    net = tf.reshape(l3_points, [batch_size, -1])
    net = tf_util.fully_connected(net, 12800, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1')
    net = tf_util.fully_connected(net, 6400, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp2')
    print("before label net\n",net)
    attachment_net = tf_util.fully_connected(net, 128*40, activation_fn=None, scope='fc3')
    type_net = tf_util.fully_connected(net, 128*4, activation_fn=None, scope='fc4')
    orientation_net = tf_util.fully_connected(net, 128*4, activation_fn=None, scope='fc5')
    surface_net = tf_util.fully_connected(net, 128*4, activation_fn=None,scope='fc6')
    startstage_net = tf_util.fully_connected(net, 128*100, activation_fn=None, scope='fc7')
    print("atttachment\n")
    print(attachment_net)
    print("type\n")
    print(type_net)
    print("stage\n")
    print(startstage_net)
    attachment_net = tf.reshape(attachment_net, [batch_size,128,-1])
    type_net = tf.reshape(type_net,[batch_size,128,-1])
    orientation_net = tf.reshape(orientation_net,[batch_size,128,-1])
    surface_net = tf.reshape(surface_net,[batch_size,128,-1])
    startstage_net = tf.reshape(startstage_net,[batch_size,128,-1])
    print("after resahpe\n")
    print(attachment_net)
    print("another\n")
    print(surface_net)

    return attachment_net, type_net, orientation_net, surface_net, startstage_net, end_points


def get_loss(attachment_pred, type_pred, orientation_pred, surface_pred, startstage_pred,
            attachment_label, type_label, orientation_label, surface_label, startstage_label, end_points):
    """ pred: B*NUM_CLASSES,
        label: B, """
    
    attachment_loss =tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=attachment_pred, labels=attachment_label))
    type_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=type_pred, labels=type_label))
    orientation_loss =tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=orientation_pred, labels=orientation_label))
    surface_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=surface_pred, labels=surface_label))
    startstage_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=startstage_pred, labels=startstage_label))

    tf.add_to_collection('losses',attachment_loss)
    tf.add_to_collection('losses',type_loss)
    tf.add_to_collection('losses',orientation_loss)
    tf.add_to_collection('losses',surface_loss)
    tf.add_to_collection('losses',startstage_loss)
    tf.summary.scalar('surface_losses', surface_loss)
  

if __name__ == '__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((16, 10000, 3))
        output, _ , _ , _ , _ , _ = get_model(inputs, tf.constant(True))
        print(output)

