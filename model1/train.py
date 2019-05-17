'''
    Single-GPU training.
    Will use H5 dataset in default. If using normal, will shift to the normal dataset.
'''
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import argparse
from datetime import datetime
import numpy as np
import tensorflow as tf
import socket
import importlib
import modelnet_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet2_cls_ssg', help='Model name [default: pointnet2_cls_ssg]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=10000, help='Point Number [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=50, help='Epoch to run [default: 251]')
parser.add_argument('--batch_size', type=int, default=2, help='Batch Size during training [default: 16]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--normal', action='store_true', default=True, help='Whether to use normal information')
FLAGS = parser.parse_args()

EPOCH_CNT = 0

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
MODEL = importlib.import_module(FLAGS.model)  # import network module
MODEL_FILE = os.path.join(ROOT_DIR, 'models', FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR))  # bkp of model def
os.system('cp train.py %s' % (LOG_DIR))  # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

NUM_CLASSES = 40

if FLAGS.normal:
    assert(NUM_POINT <= 10000)
    DATA_PATH='alldata/'
    TRAIN_DATASET = modelnet_dataset.ModelNetDataset(root=DATA_PATH, npoints=NUM_POINT, split='train', batch_size=BATCH_SIZE)
    TEST_DATASET = modelnet_dataset.ModelNetDataset(root=DATA_PATH, npoints=NUM_POINT, split='test', batch_size=BATCH_SIZE)


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001)  # CLIP THE LEARNING RATE!
    return learning_rate


def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay
def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, attachment_labels_pl, type_labels_pl, orientation_labels_pl, surface_labels_pl, startstage_labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())

            batch = tf.get_variable('batch', [], initializer=tf.constant_initializer(0), trainable=False)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss
            attachment_pred, type_pred, orientation_pred, surface_pred, startstage_pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl, bn_decay=bn_decay)

            MODEL.get_loss(attachment_pred, type_pred, orientation_pred, surface_pred, startstage_pred,
                           attachment_labels_pl, type_labels_pl, orientation_labels_pl, surface_labels_pl, startstage_labels_pl, end_points)
            losses = tf.get_collection('losses')
            print('############losses############')
            print(losses)
            total_loss = tf.add_n(losses, name='total_loss')
            tf.summary.scalar('total_loss', total_loss)
            for l in losses + [total_loss]:
                tf.summary.scalar(l.op.name, l)

            attachment_correct = tf.equal(tf.argmax(attachment_pred, 2), tf.to_int64(attachment_labels_pl))
            type_correct = tf.equal(tf.argmax(type_pred, 2), tf.to_int64(type_labels_pl))
            orientation_correct = tf.equal(tf.argmax(orientation_pred, 2), tf.to_int64(orientation_labels_pl))
            surface_correct = tf.equal(tf.argmax(surface_pred, 2), tf.to_int64(surface_labels_pl))
            startstage_correct = tf.equal(tf.argmax(startstage_pred, 2), tf.to_int64(startstage_labels_pl))

            attachment_accuracy = tf.reduce_mean(tf.cast(attachment_correct, tf.float32))
            type_accuracy = tf.reduce_mean(tf.cast(type_correct, tf.float32))
            orientation_accuracy = tf.reduce_mean(tf.cast(orientation_correct, tf.float32))
            surface_accuracy = tf.reduce_mean(tf.cast(surface_correct, tf.float32))
            startstage_accuracy = tf.reduce_mean(tf.cast(startstage_correct, tf.float32))

            # never write again
            tf.summary.scalar('attachment_accuracy', attachment_accuracy)
            tf.summary.scalar('type_accuracy', type_accuracy)
            tf.summary.scalar('orientation_accuracy', orientation_accuracy)
            tf.summary.scalar('surface_accuracy', surface_accuracy)
            tf.summary.scalar('startstage_accuracy', startstage_accuracy)

            print ("--- Get training operator")
            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(total_loss, global_step=batch)

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init)

        ops = {'pointclouds_pl': pointclouds_pl,
               'attachment_labels_pl': attachment_labels_pl,
               'type_labels_pl': type_labels_pl,
               'orientation_labels_pl': orientation_labels_pl,
               'surface_labels_pl': surface_labels_pl,
               'startstage_labels_pl': startstage_labels_pl,
               'is_training_pl': is_training_pl,
               'attachment_pred': attachment_pred,
               'type_pred': type_pred,
               'orientation_pred': orientation_pred,
               'surface_pred': surface_pred,
               'startstage_pred': startstage_pred,
               'loss': total_loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'end_points': end_points}

        best_acc = -1
        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()

            train_one_epoch(sess, ops, train_writer)
            eval_one_epoch(sess, ops, test_writer)

            # Save the variables to disk.
            if epoch % 10 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)

def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True

    log_string(str(datetime.now()))

    # Make sure batch data is of same size
    cur_batch_data = np.zeros((BATCH_SIZE, NUM_POINT, TRAIN_DATASET.num_channel()))
    cur_batch_attachment_label = np.zeros((BATCH_SIZE, 128), dtype=np.int32)
    cur_batch_type_label = np.zeros((BATCH_SIZE, 128), dtype=np.int32)
    cur_batch_orientation_label = np.zeros((BATCH_SIZE, 128), dtype=np.int32)
    cur_batch_surface_label = np.zeros((BATCH_SIZE, 128), dtype=np.int32)
    cur_batch_startstage_label = np.zeros((BATCH_SIZE, 128), dtype=np.int32)

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    batch_idx = 0
    losses = tf.get_collection('losses')
    while TRAIN_DATASET.has_next_batch():
        batch_data, batch_attachment_label, batch_type_label, batch_orientation_label, batch_surface_label, batch_startstage_label = TRAIN_DATASET.next_batch(augment=True)
        # batch_data = provider.random_point_dropout(batch_data)
        bsize = 4
        cur_batch_data[0:bsize, ...] = batch_data
        cur_batch_attachment_label[0:bsize, ...] = batch_attachment_label
        cur_batch_type_label[0:bsize, ...] = batch_type_label
        cur_batch_orientation_label[0:bsize, ...] = batch_orientation_label
        cur_batch_surface_label[0:bsize, ...] = batch_surface_label
        cur_batch_startstage_label[0:bsize, ...] = batch_startstage_label

        feed_dict = {ops['pointclouds_pl']: cur_batch_data,
                     ops['attachment_labels_pl']: cur_batch_attachment_label,
                     ops['type_labels_pl']: cur_batch_type_label,
                     ops['orientation_labels_pl']: cur_batch_orientation_label,
                     ops['surface_labels_pl']: cur_batch_surface_label,
                     ops['startstage_labels_pl']: cur_batch_startstage_label,
                     ops['is_training_pl']: is_training}
        a, b, c, d, e, summary, step, _, loss_val, attachment_pred_val, type_pred_val, orientation_pred_val, surface_pred_val, startstage_pred_val = sess.run([losses[0], losses[1], losses[2], losses[3],losses[4], ops['merged'], ops['step'],
            ops['train_op'], ops['loss'], ops['attachment_pred'], ops['type_pred'], ops['orientation_pred'], ops['surface_pred'], ops['startstage_pred']], feed_dict=feed_dict)
        print ('==================', a,b, c, d, e)
        train_writer.add_summary(summary, step)
        attachment_pred_val = np.argmax(attachment_pred_val, 2)
        type_pred_val = np.argmax(type_pred_val, 2)
        orientation_pred_val = np.argmax(orientation_pred_val, 2)
        surface_pred_val = np.argmax(surface_pred_val, 2)
        startstage_pred_val = np.argmax(startstage_pred_val, 2)
        correct = np.sum(attachment_pred_val[0:bsize] == batch_attachment_label[0:bsize])+np.sum(type_pred_val[0:bsize] == batch_type_label[0:bsize])+np.sum(
            orientation_pred_val[0:bsize] == batch_orientation_label[0:bsize])+np.sum(surface_pred_val[0:bsize] == batch_surface_label[0:bsize])+np.sum(startstage_pred_val[0:bsize] == batch_startstage_label[0:bsize])
        total_correct += correct
        total_seen += bsize*128*5
        loss_sum += loss_val
        if True:
            log_string(' ---- batch: %03d ----' % (batch_idx+1))
            log_string('mean loss: %f' % (loss_sum))
            log_string('accuracy: %f' % (total_correct / float(total_seen)))
            total_correct = 0
            total_seen = 0
            loss_sum = 0
        batch_idx += 1

    TRAIN_DATASET.reset()


def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    global EPOCH_CNT
    is_training = False
    
    
    cur_batch_data = np.zeros((BATCH_SIZE, NUM_POINT, TRAIN_DATASET.num_channel()))
    cur_batch_attachment_label = np.zeros((BATCH_SIZE, 128), dtype=np.int32)
    cur_batch_type_label = np.zeros((BATCH_SIZE, 128), dtype=np.int32)
    cur_batch_orientation_label = np.zeros((BATCH_SIZE, 128), dtype=np.int32)
    cur_batch_surface_label = np.zeros((BATCH_SIZE, 128), dtype=np.int32)
    cur_batch_startstage_label = np.zeros((BATCH_SIZE, 128), dtype=np.int32)

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    batch_idx = 0
    losses=tf.get_collection('losses')
    log_string(str(datetime.now()))
    log_string('----EPOCH %03d EVALUATION----'%(EPOCH_CNT))

    while TEST_DATASET.has_next_batch():
        batch_data, batch_attachment_label, batch_type_label, batch_orientation_label, batch_surface_label, batch_startstage_label = TEST_DATASET.next_batch(augment=True)
        # batch_data = provider.random_point_dropout(batch_data)
        bsize = 4
        cur_batch_data[0:bsize, ...] = batch_data
        cur_batch_attachment_label[0:bsize, ...] = batch_attachment_label
        cur_batch_type_label[0:bsize, ...] = batch_type_label
        cur_batch_orientation_label[0:bsize, ...] = batch_orientation_label
        cur_batch_surface_label[0:bsize, ...] = batch_surface_label
        cur_batch_startstage_label[0:bsize, ...] = batch_startstage_label

        feed_dict = {ops['pointclouds_pl']: cur_batch_data,
                     ops['attachment_labels_pl']: cur_batch_attachment_label,
                     ops['type_labels_pl']: cur_batch_type_label,
                     ops['orientation_labels_pl']: cur_batch_orientation_label,
                     ops['surface_labels_pl']: cur_batch_surface_label,
                     ops['startstage_labels_pl']: cur_batch_startstage_label,
                     ops['is_training_pl']: is_training}
        a, b, c, d, e, summary, step, _, loss_val, attachment_pred_val, type_pred_val, orientation_pred_val, surface_pred_val, startstage_pred_val = sess.run([losses[0], losses[1], losses[2], losses[3], ops['merged'], ops['step'],
            ops['train_op'], ops['loss'], ops['attachment_pred'], ops['type_pred'], ops['orientation_pred'], ops['surface_pred'], ops['startstage_pred']], feed_dict=feed_dict)
        print ('==================', a, b, c, d, e)
        test_writer.add_summary(summary, step)
        attachment_pred_val = np.argmax(attachment_pred_val, 2)
        type_pred_val = np.argmax(type_pred_val, 2)
        orientation_pred_val = np.argmax(orientation_pred_val, 2)
        surface_pred_val = np.argmax(surface_pred_val, 2)
        startstage_pred_val = np.argmax(startstage_pred_val, 2)
        correct = np.sum(attachment_pred_val[0:bsize] == batch_attachment_label[0:bsize])+np.sum(type_pred_val[0:bsize] == batch_type_label[0:bsize])+np.sum(
            orientation_pred_val[0:bsize] == batch_orientation_label[0:bsize])+np.sum(surface_pred_val[0:bsize] == batch_surface_label[0:bsize])+np.sum(startstage_pred_val[0:bsize] == batch_startstage_label[0:bsize])
        total_correct += correct
        total_seen += bsize*128*5
        loss_sum += loss_val
        log_string(' ---- batch: %03d ----' % (batch_idx+1))
        log_string('mean loss: %f' % (loss_sum))
        log_string('accuracy: %f' % (total_correct / float(total_seen)))
         
    EPOCH_CNT+=1
    batch_idx+=1
    TEST_DATASET.reset()
    return total_correct/float(total_seen)

if __name__ == "__main__":
    log_string('pid: %s' % (str(os.getpid())))
    train()
    LOG_FOUT.close()

