#!/bin/bash
trap "echo 'Stopped. Exiting.'; exit 0" SIGINT
my_program() {
    #../IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py --task=Quadruped-Locomotion-Go2 --headless --resume
    ../IsaacLab/isaaclab.sh -p -m torch.distributed.run --nproc_per_node=8 scripts/rsl_rl/train.py --task=Quadruped-Locomotion-Go2 --headless --distributed --resume
}
while true; do my_program || sleep 10; done