#!/bin/bash
# set -e 
# observe RL can increase adapter weight 
bash examples/lite/train.sh "trail0__lr1e-2_reg0_initval8e-1" "1e-2" "0" "0.8"
bash examples/lite/train.sh "trail0__lr1e-3_reg0_initval8e-1" "1e-3" "0" "0.8"
bash examples/lite/train.sh "trail0__lr1e-4_reg0_initval8e-1" "1e-4" "0" "0.8"

# observe RL can increase adapter weight with reg loss
bash examples/lite/train.sh "trail1__lr1e-2_reg1e-3_initval8e-1" "1e-2" "1e-3" "0.8"
bash examples/lite/train.sh "trail1__lr1e-3_reg1e-3_initval8e-1" "1e-3" "1e-3" "0.8"