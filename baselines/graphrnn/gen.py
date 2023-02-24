import pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'

from baselines.graphrnn.train import *


def gen(args, model, output, num_gen=1):
    generated = test_rnn_epoch(0, args, model, output, test_batch_size=num_gen)
    return generated
