#!/bin/bash


#python z_my_train_i3d_1.py -mode 'rgb' -save_model 'tmp/tmp' 2>&1 | tee AAA.log

python z_my_train_i3d_1.py -mode 'flow' -save_model 'tmp/tmp' 2>&1 | tee AAA.log

#python z_my_train_i3d_1_attention_gaze.py -mode 'rgb' -save_model 'tmp/tmp' 2>&1 | tee AAA.log

#python z_my_train_i3d_1_attention_gaze.py -mode 'flow' -save_model 'tmp/tmp' 2>&1 | tee AAA.log


echo 'Done.'
