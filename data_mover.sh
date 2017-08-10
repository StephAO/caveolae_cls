#!/bin/bash

cd ~/
for i in 1 2 3 4
do
    echo "Starting data transfer of experiment ${i}"
    python -m caveolae_cls.resampler projection /home/stephane/sfu_data/DL_Exp${i}/Blobs_Exp${i}_MAT_PC3 /home/stephane/sfu_data/DL_Exp${i}/Projs_Exp${i}_MAT_PCA_PC3 0
    python -m caveolae_cls.resampler projection /home/stephane/sfu_data/DL_Exp${i}/Blobs_Exp${i}_MAT_PC3PTRF /home/stephane/sfu_data/DL_Exp${i}/Projs_Exp${i}_MAT_PCA_PC3PTRF 0
done