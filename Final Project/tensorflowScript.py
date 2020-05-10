from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import os
import json
import argparse
from tensorflow.python.platform import tf_logging
import logging as _logging
import sys as _sys
from tensorflow.python.estimator.model_fn import ModeKeys as Modes

tf.logging.set_verbosity(tf.logging.INFO)

INPUT_TENSOR_NAME = 'inputs'
SIGNATURE_NAME = 'predictions'

def _model_fn(features, labels, mode, params):
    # Input Layer
    input_layer = tf.reshape(features[INPUT_TENSOR_NAME], [-1, 50, 50, 3])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu
    )

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 12 * 12 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=(mode == Modes.TRAIN))

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=params['num_classes'])

    # Define operations
    if mode in (Modes.PREDICT, Modes.EVAL):
        predicted_indices = tf.argmax(input=logits, axis=1)
        probabilities = tf.nn.softmax(logits, name='softmax_tensor')

    if mode in (Modes.TRAIN, Modes.EVAL):
        global_step = tf.train.get_or_create_global_step()
        label_indices = tf.cast(labels, tf.int32)
        loss = tf.losses.softmax_cross_entropy(
            onehot_labels=tf.one_hot(label_indices, depth=params['num_classes']), logits=logits)
        tf.summary.scalar('OptimizeLoss', loss)

    if mode == Modes.PREDICT:
        predictions = {
            'classes': predicted_indices,
            'probabilities': probabilities
        }
        export_outputs = {
            SIGNATURE_NAME: tf.estimator.export.PredictOutput(predictions)
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)

    if mode == Modes.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
        train_op = optimizer.minimize(loss, global_step=global_step)
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    if mode == Modes.EVAL:
        eval_metric_ops = {
            'accuracy': tf.metrics.accuracy(label_indices, predicted_indices)
        }
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops)

def _read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image.set_shape([50 * 50 * 3])
    image = tf.cast(image, tf.float32) * (1. / 255)
    label = tf.cast(features['label'], tf.int32)

    return image, label

def _input_fn(input_dir, input_filename, mini_batch_size):
    file_path = os.path.join(input_dir, input_filename)
    filename_queue = tf.train.string_input_producer([file_path])

    image, label = _read_and_decode(filename_queue)
    images, labels = tf.train.batch([image, label], batch_size=mini_batch_size, capacity=1000 + 3 * mini_batch_size)
    return {INPUT_TENSOR_NAME: images}, labels

def _train_input_fn(training_dir, mini_batch_size):
    def train_input():
        return _input_fn(training_dir, 'images_train.tfrecords', mini_batch_size)
    return train_input


def _eval_input_fn(eval_dir, mini_batch_size):
    def eval_input():
        return _input_fn(eval_dir, 'images_test.tfrecords', mini_batch_size)
    return eval_input

def _serving_input_fn():
    inputs = {INPUT_TENSOR_NAME: tf.placeholder(tf.float32, [None, 50 * 50 * 3])}
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)

def _parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str)
    parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ.get('SM_HOSTS')))
    parser.add_argument('--current-host', type=str, default=os.environ.get('SM_CURRENT_HOST'))
    parser.add_argument('--num-classes', type=int, default=2)
    parser.add_argument('--mini-batch-size', type=int, default=100)
    parser.add_argument('--max-steps', type=int, default=100)
    parser.add_argument('--learning-rate', type=float, default=0.1)
    
    return parser.parse_known_args()

if __name__ == '__main__':
    args, unknown = _parse_args()
    
    # Create the Estimator
    classifier = tf.estimator.Estimator(
        model_fn=_model_fn, model_dir=args.model_dir, params = {
            'num_classes': args.num_classes,
            'learning_rate': args.learning_rate
        }
    )
    
    train_spec = tf.estimator.TrainSpec(_train_input_fn(args.train, args.mini_batch_size), max_steps=args.max_steps)
    eval_spec = tf.estimator.EvalSpec(_eval_input_fn(args.test, args.mini_batch_size), steps=args.max_steps)
    tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)

    if args.current_host == args.hosts[0]:
        classifier.export_savedmodel(args.sm_model_dir, _serving_input_fn)