#!/bin/bash
# Stage 2 only: infer all Ag1000G maps, sharded across the two GPUs.
cd /home/kkor/agam_work
FR=/home/kkor/venvs/fastrho/bin/python
M=/home/kkor/fastrho_data/campaign/markers
mkdir -p logs maps
rm -f "$M/AGAM_DONE"
PYTHONNOUSERSITE=1 $FR infer_agam.py /home/kkor/agam_work/haps/ cuda:0 0 2 > logs/infer0.log 2>&1 &
P0=$!
PYTHONNOUSERSITE=1 $FR infer_agam.py /home/kkor/agam_work/haps/ cuda:1 1 2 > logs/infer1.log 2>&1 &
P1=$!
wait $P0 $P1
touch "$M/AGAM_DONE"
echo "AGAM_DONE ($(ls maps/*.npz 2>/dev/null | wc -l) maps)"
