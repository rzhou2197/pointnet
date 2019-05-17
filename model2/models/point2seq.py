import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tensorflow as tf
import numpy as np
import tf_util
from pointnet_util import pointnet_sa_module, pointnet_sa_module_msg
from tensorflow.contrib.seq2seq import *
from tensorflow.python.layers.core import Dense


def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size, 95))
    return pointclouds_pl, labels_pl


def get_model(point_cloud, label, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx95x51 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}

    l0_xyz = point_cloud
    l0_points = None

    # Set abstraction layers
    l1_xyz, l1_points = pointnet_sa_module_msg(l0_xyz, l0_points, 4096, [0.1,0.2,0.4], [16,32,128], [[32,32,64], [64,64,128], [64,96,128]], is_training, bn_decay, scope='layer1', use_nchw=True)
    l2_xyz, l2_points = pointnet_sa_module_msg(l1_xyz, l1_points, 512, [0.2,0.4,0.8], [32,64,128], [[64,64,128], [128,128,256], [128,128,256]], is_training, bn_decay, scope='layer2')
    l3_xyz, l3_points, _ = pointnet_sa_module(l2_xyz, l2_points, npoint=None, radius=None, nsample=None, mlp=[256,512,1024], mlp2=None, group_all=True, is_training=is_training, bn_decay=bn_decay, scope='layer3')
  
    # Encoder-Decoder
    net = tf.reshape(l3_points, [batch_size, -1])
    net = tf.expand_dims(net, axis=1)

    # Define LSTM encoder
    with tf.variable_scope("encoder"):
       encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
              tf.contrib.rnn.LSTMCell(128), net, dtype = tf.float32)

    # Define helper
    d_embedding = tf.get_variable("d_embedding",
                             shape=[51, 51])

    target = tf.nn.embedding_lookup(d_embedding, label)

    seq_length = [95 for i in range(batch_size)]
    start = [0 for i in range(batch_size)]

    helper1 = GreedyEmbeddingHelper(d_embedding, start, 50)
    helper2 = TrainingHelper(target, seq_length)


    # Define decoder
    fc_layer = Dense(51)
    decoder_cell = tf.contrib.rnn.LSTMCell(128)

    with tf.variable_scope("decoder"):
       train_decoder=BasicDecoder(decoder_cell, helper2,
                             encoder_state, fc_layer)

    net1, final_state1, final_sequence_lengths1 = dynamic_decode(train_decoder)

    with tf.variable_scope("decoder"):
       infer_decoder=BasicDecoder(decoder_cell, helper1,
                             encoder_state, fc_layer)
       
    net2, final_state2, final_sequence_lengths2 = dynamic_decode(infer_decoder)
   
    return net1.rnn_output, net2.rnn_output, end_points


def get_loss(pred, label, end_points):
    """ pred: B*95*51,
        label: B*95, """
    targets = tf.reshape(label, [-1])
    logits_flat = tf.reshape(pred, [-1, 51])
    loss =  tf.losses.sparse_softmax_cross_entropy(targets, logits_flat)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classifyloss', classify_loss)
    tf.add_to_collection('losses', classify_loss)
    return classify_loss


if __name__ == '__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32, 1024, 3))
        net, _ = get_model(inputs, tf.constant(True))
        print(net)
