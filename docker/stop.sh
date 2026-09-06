#!/usr/bin/env bash

CONTAINER_NAME=quadruped_go2_locomotion

docker stop $CONTAINER_NAME && docker rm $CONTAINER_NAME