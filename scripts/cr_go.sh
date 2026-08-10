#!/bin/bash
# Robust relaunch of the C. remanei build (runs detached on sesame, survives SSH drops).
pgrep -f cr_build.sh | grep -v $$ | xargs -r kill -9 2>/dev/null
pgrep -f cr_ingest.sh | xargs -r kill -9 2>/dev/null
sleep 3
rm -f /home/kkor/realdata/cr.status /home/kkor/realdata/cr/SRR*.bam /home/kkor/realdata/cr_build.log
nohup bash /home/kkor/realdata/cr_build.sh </dev/null >/dev/null 2>&1 &
sleep 1
nohup bash /home/kkor/realdata/cr_ingest.sh </dev/null >/dev/null 2>&1 &
echo "CR_GO $(date +%T)" > /home/kkor/realdata/cr_go.status
