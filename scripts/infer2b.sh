#!/bin/bash
# Finish remaining Anopheles maps; pin each process to ONE GPU via CUDA_VISIBLE_DEVICES
# (addressed as cuda:0 inside) to avoid the Mamba/Triton cuda:1 error. Skips done maps.
cd /home/kkor/fastrho
FR=/home/kkor/venvs/fastrho/bin/python
M=/home/kkor/fastrho_data/campaign/markers
rm -f "$M/AGAM_DONE"
CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 $FR scripts/infer_agam.py /home/kkor/agam_work/haps/ cuda:0 0 2 > /home/kkor/agam_work/logs/infer0b.log 2>&1 &
P0=$!
CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 $FR scripts/infer_agam.py /home/kkor/agam_work/haps/ cuda:0 1 2 > /home/kkor/agam_work/logs/infer1b.log 2>&1 &
P1=$!
wait $P0 $P1
touch "$M/AGAM_DONE"
echo "AGAM_DONE ($(ls /home/kkor/agam_work/maps/*.npz 2>/dev/null | wc -l)/25 maps)"
