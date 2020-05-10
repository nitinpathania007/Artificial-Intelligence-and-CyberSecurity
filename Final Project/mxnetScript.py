from __future__ import print_function

import argparse
import gzip
import json
import logging
import os
import struct
import time

import mxnet as mx
import numpy as np
import mxnet as mx
from mxnet import autograd as ag
from mxnet import gluon
from mxnet.gluon.model_zoo import vision as models

def _train(batch_size, epochs, learning_rate, momentum, wd, log_interval, num_gpus, hosts, current_host,
          num_cpus, model_dir, train_data_dir, test_data_dir, **kwargs):
    logging.getLogger().setLevel(logging.DEBUG)

    # Initialize neural network
    ctx = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
    net = models.get_model('resnet34_v2', ctx=ctx, pretrained=False, classes=2)
    net.initialize(mx.init.Xavier(magnitude=2), ctx=ctx)
    net.hybridize()
    batch_size *= max(1, len(ctx))

    # Load train and test data
    part_index = 0
    for i, host in enumerate(hosts):
        if host == current_host:
            part_index = i
            break

    train_data = _get_train_data(num_cpus, train_data_dir, batch_size, (3, 224, 224), resize=224)
    test_data = _get_test_data(num_cpus, test_data_dir, batch_size, (3, 224, 224), resize=224)
    
    # Create a trainer
    if len(hosts) == 1:
        kvstore = 'device' if num_gpus > 0 else 'local'
    else:
        kvstore = 'dist_device_sync'
        
    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                            optimizer_params={'learning_rate': learning_rate, 'momentum': momentum, 'wd': wd},
                            kvstore=kvstore)
    
    # Initialize a metric variable for measuring accuracy
    metric = mx.metric.Accuracy()
    
    # Define the loss function
    loss = gluon.loss.SoftmaxCrossEntropyLoss()

    # Train the neural network
    best_accuracy = 0.0
    for epoch in range(epochs):
        train_data.reset()
        tic = time.time()
        metric.reset()
        btic = time.time()

        for i, batch in enumerate(train_data):
            data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
            outputs = []
            Ls = []
            with ag.record():
                for x, y in zip(data, label):
                    z = net(x)
                    L = loss(z, y)
                    Ls.append(L)
                    outputs.append(z)
                for L in Ls:
                    L.backward()
            trainer.step(batch.data[0].shape[0])
            metric.update(label, outputs)
            if i % log_interval == 0 and i > 0:
                name, acc = metric.get()
                logging.info('Epoch [%d] Batch [%d]\tSpeed: %f samples/sec\t%s=%f' %
                             (epoch, i, batch_size / (time.time() - btic), name, acc))
            btic = time.time()

        name, acc = metric.get()
        logging.info('[Epoch %d] training: %s=%f' % (epoch, name, acc))
        logging.info('[Epoch %d] time cost: %f' % (epoch, time.time() - tic))

        name, val_acc = _test(ctx, net, test_data)
        logging.info('[Epoch %d] validation: %s=%f' % (epoch, name, val_acc))

        # Only save params on primary host
        if current_host == hosts[0]:
            if val_acc > best_accuracy:
                _save(net, model_dir)
                best_accuracy = val_acc

    return net

def _get_data(path, augment, num_cpus, batch_size, data_shape, resize=-1, num_parts=1, part_index=0):
    return mx.io.ImageRecordIter(
        path_imgrec=path,
        resize=resize,
        data_shape=data_shape,
        batch_size=batch_size,
        rand_crop=augment,
        rand_mirror=augment,
        preprocess_threads=num_cpus,
        num_parts=num_parts,
        part_index=part_index
    )

def _get_train_data(num_cpus, data_dir, batch_size, data_shape, resize=-1):
    return _get_data(os.path.join(data_dir, 'images_train.rec'), True, num_cpus, batch_size, data_shape, resize)

def _get_test_data(num_cpus, data_dir, batch_size, data_shape, resize=-1):
    return _get_data(os.path.join(data_dir, 'images_test.rec'), False, num_cpus, batch_size, data_shape, resize)

def _test(ctx, net, test_data):
    test_data.reset()
    metric = mx.metric.Accuracy()

    for i, batch in enumerate(test_data):
        data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
        outputs = []
        for x in data:
            outputs.append(net(x))
        metric.update(label, outputs)
    return metric.get()

def _save(net, model_dir):
    net.export('%s/model'% model_dir)
    
def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    
    parser.add_argument('--num-classes', type=int, default=2)
    parser.add_argument('--mini-batch-size', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning-rate', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--wd', type=float, default=0.0001)
    
    return parser.parse_args()

# ------------------------------------------------------------ #
# Hosting functions                                            #
# ------------------------------------------------------------ #

def model_fn(model_dir):
    net = gluon.SymbolBlock.imports(
        '%s/model-symbol.json' % model_dir,
        ['data'],
        '%s/model-0000.params' % model_dir,        
    )
    return net


def transform_fn(net, data, input_content_type, output_content_type):
    parsed = json.loads(data)
    nda = mx.nd.array(parsed)
    output = net(nda)
    prediction = mx.nd.argmax(output, axis=1)
    response_body = json.dumps(prediction.asnumpy().tolist()[0])
    return response_body, output_content_type

if __name__ == '__main__':
    args = _parse_args()
    num_gpus = int(os.environ['SM_NUM_GPUS'])
    num_cpus = int(os.environ['SM_NUM_CPUS'])
    log_interval = 1
    
    _train(args.mini_batch_size, args.epochs, args.learning_rate, args.momentum, args.wd,
          log_interval, num_gpus, args.hosts, args.current_host, num_cpus, args.model_dir, args.train, args.test)