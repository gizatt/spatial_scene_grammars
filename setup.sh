#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:$DIR/packages
