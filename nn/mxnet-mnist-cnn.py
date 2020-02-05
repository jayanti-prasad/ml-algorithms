import mxnet as mx
import mxnet as mx
from mxnet import nd, gluon, autograd
from mxnet.gluon import nn
import matplotlib.pyplot as plt
import numpy as np


def get_params():
    P = {'random_seed': 42,
         'num_epochs': 10,
         'batch_size': 100}
    return P


def data_xform(data):
    """Move channel axis to the beginning, cast to float32, and normalize to [0, 1]."""
    return nd.moveaxis(data, 2, 0).astype('float32') / 255


def get_data(P):
    train_data = mx.gluon.data.vision.MNIST(train=True).transform_first(data_xform)
    val_data = mx.gluon.data.vision.MNIST(train=False).transform_first(data_xform)

    train_loader = mx.gluon.data.DataLoader(train_data, shuffle=True, batch_size=P['batch_size'])
    val_loader = mx.gluon.data.DataLoader(val_data, shuffle=False, batch_size=P['batch_size'])

    return  train_data , val_data , train_loader, val_loader 

def get_model (P):

    lenet = nn.HybridSequential(prefix='LeNet_')
    with lenet.name_scope():
        lenet.add(
        nn.Conv2D(channels=20, kernel_size=(5, 5), activation='tanh'),
        nn.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
        nn.Conv2D(channels=50, kernel_size=(5, 5), activation='tanh'),
        nn.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
        nn.Flatten(),
        nn.Dense(500, activation='tanh'),
        nn.Dense(10, activation=None),
    )

    return lenet 



if __name__ == "__main__":

    P = get_params()

    train_data , val_data , train_loader, val_loader =  get_data(P)

    mx.random.seed(P['random_seed'])

    net = get_model(P)

    ctx = mx.gpu(0) if mx.context.num_gpus() > 0 else mx.cpu(0)


    net.initialize(mx.init.Xavier(), ctx=ctx)
    net.summary(nd.zeros((1, 1, 28, 28), ctx=ctx))

    trainer = gluon.Trainer(params=net.collect_params(), optimizer='sgd', optimizer_params={'learning_rate': 0.04},)
    metric = mx.metric.Accuracy()

    loss_function = gluon.loss.SoftmaxCrossEntropyLoss()

    for epoch in range(P['num_epochs']):
        for inputs, labels in train_loader:
        # Possibly copy inputs and labels to the GPU
            inputs = inputs.as_in_context(ctx)
            labels = labels.as_in_context(ctx)

            # The forward pass and the loss computation need to be wrapped
            # in a `record()` scope to make sure the computational graph is
            # recorded in order to automatically compute the gradients
            # during the backward pass.
            with autograd.record():
                outputs = net(inputs)
                loss = loss_function(outputs, labels)

            # Compute gradients by backpropagation and update the evaluation
            # metric
            loss.backward()
            metric.update(labels, outputs)

            # Update the parameters by stepping the trainer; the batch size
            # is required to normalize the gradients by `1 / batch_size`.
            trainer.step(batch_size=inputs.shape[0])

        # Print the evaluation metric and reset it for the next epoch
        name, acc = metric.get()
        print('After epoch {}: {} = {}'.format(epoch + 1, name, acc))
        metric.reset()


    metric = mx.metric.Accuracy()
    for inputs, labels in val_loader:
        # Possibly copy inputs and labels to the GPU
        inputs = inputs.as_in_context(ctx)
        labels = labels.as_in_context(ctx)
        metric.update(labels, net(inputs))
    print('Validaton: {} = {}'.format(*metric.get()))
    assert metric.get()[1] > 0.96

