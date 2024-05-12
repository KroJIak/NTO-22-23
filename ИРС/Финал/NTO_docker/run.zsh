#!/bin/zsh
xhost +local:docker > /dev/null || true
sudo docker run -d -it --rm --gpus all \
    -v /etc/localtime:/etc/localtime:ro \
    -e "DISPLAY" \
    -e "QT_X11_NO_MITSHM=1" \
    -e XAUTHORITY \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v ./workspace/:/workspace/ \
    --net=host \
    --privileged \
    -v /dev:/dev \
    -v .:/app \
    --name blth bluetooth
