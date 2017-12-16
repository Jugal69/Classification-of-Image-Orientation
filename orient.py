#!/usr/bin/env python

import sys
from adaboost import *
#from knn import *

phase = sys.argv[1]
phase_file = sys.argv[2]
model_file = sys.argv[3]
model = sys.argv[4]

if model=='nearest':
    if phase=='train':
        knn_train(phase_file,model_file)
    else:
        distance(phase_file,model_file)
elif model=='adaboost':
    if phase=='train':
        train_adaboost(phase_file,model_file)
    else:
        test_adaboost(phase_file,model_file)
else:
    if phase=='train':
        nnet_train(phase_file,model_file)
    else:
        nnet_test(phase_file,model_file)
