# Adasum in Horovod on Azure ML

Adasum is a new scalable algorithm for distributed training available in Horovod. We have a [pull request open](https://github.com/horovod/horovod/pull/1485) for inclusion into Horovod mainline and in the meanwhile it is available as a separate pip package `horovod-adasum`. This repository contains a minimal example for submitting training jobs using Adasum onto Azure Machine Learning.

## Getting started

Follow the instructions at https://docs.microsoft.com/en-us/python/api/overview/azure/ml/install?view=azure-ml-py to install the Azure Machine Learning SDK for Python.

Fill in `config_aml.json` with your subscription ID, resource group and workspace name.

You can now submit an experiment to train MNIST with Pytorch:
```
python submit.py
```

This will submit the standard MNIST CNN model to train on a cluster of eight NC24 machines with 4 K80s each.

## Notes on using horovod-adasum

Normally Azure Machine Learning handles installation of Horovod for you. For the `horovod-adasum` package we must ensure that it installs after Pytorch. The `submit.py` script included here handles this by installing Pytorch as a Conda package, and `horovod-adasum` as a pip package, which are installed after packages from Conda.

## Using Adasum in your own model

The following lines of code set up the Horovod optimizer in `pytorch_mnist.py`.
```
optimizer = hvd.DistributedOptimizer(optimizer,
                                     named_parameters=model.named_parameters(),
                                     compression=compression,
                                     op=hvd.Adasum)
```

Adasum is enabled here by passing `op=hvd.Adasum` to `hvd.DistributedOptimizer`.

Because Adasum is an "adaptive summation" there is **no need to scale learning rate by the number of GPUs**. In contrast Horovod will by default average gradients, which makes it reasonable for small numbers of GPUs to scale up the learning rate (until diminishing returns from scaling up the global batch size kick in).

## Benchmarking Adasum

For benchmarking scaling to large numbers of GPUs with Adasum and averaging, it is important to ensure that your baseline 1 GPU learning rate and learning rate schedule are aggressive enough. **If you have a conservative baseline learning rate then averaging combined with scaling up the learning will seem to scale to large numbers of GPUs.** This is bad because the result you find at the scaling limit would also be reachable with a smaller number of GPUs coupled with a more agressive learning rate schedule.
