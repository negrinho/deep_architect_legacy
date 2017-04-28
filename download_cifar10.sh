#!/bin/bash

mkdir -p data/cifar10
cd data/cifar10
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xvf cifar-10-python.tar.gz 
cd -