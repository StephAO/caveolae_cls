#!/bin/bash

cd ~/
for i in 1 2 3
do
    echo "Starting data transfer of experiment ${i}"
    python -m caveolae_cls.resampler projection /home/stephane/sfu_data/DL_Exp${i}/Blobs_Exp${i}_MAT_PC3 /home/stephane/sfu_data/mil_data/negative
    python -m caveolae_cls.resampler projection /home/stephane/sfu_data/DL_Exp${i}/Blobs_Exp${i}_MAT_PC3PTRF /home/stephane/sfu_data/mil_data/positive
done