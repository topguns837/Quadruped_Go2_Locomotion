#!/bin/bash
trap "echo 'Stopped. Exiting.'; exit 0" SIGINT
my_program() {
    ../IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py --task=Quadruped-Locomotion-Go2 --headless --resume
}
while true; do my_program || sleep 5; done

../IsaacLab/isaaclab.sh -p scripts/rsl_rl/play.py --task=Quadruped-Locomotion-Go2 --num_envs=1