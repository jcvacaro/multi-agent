#!/bin/bash

PYT_IMAGE="pyt:0.1"

X11_DISPLAY=$DISPLAY

X11_DOCKER="docker run -it \
    --runtime=nvidia \
    --privileged \
    -v $HOME/.Xauthority:/root/.Xauthority \
    -v /dev/dri:/dev/dri \
    -v /dev/shm:/dev/shm \
    -e DISPLAY=$X11_DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw" 

pyt_run_base() {
    xhost +
    $X11_DOCKER --rm --name="$1" \
        -v $PWD:/workspace \
        -w /workspace \
        $PYT_IMAGE "$2"
}

pyt_run_base "pyt" "/bin/bash"
