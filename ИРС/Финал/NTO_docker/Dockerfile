FROM ubuntu:22.04

ENV DEBIAN_FRONTEND noninteractive

ENV TZ=Europe/Moscow
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get -y update && apt-get upgrade -y

RUN apt-get install -y git \
                        wget \
                        cmake \
                        clang \
                        python3 \
                        python3-dev \
                        python3-pip \
                        python3-distutils \
                        libpython3-dev \
                        libeigen3-dev \
                        python3-bluez
         
RUN apt-get install -y zsh \
                        tmux \
                        vim \
                        nano \
                        htop \
                        kmod \
                        kbd \
                        gedit



RUN sh -c "$(wget https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh -O -)" -y
RUN chsh -s $(which zsh)

RUN pip3 install opencv-python
RUN pip3 install opencv-contrib-python

CMD ["/bin/zsh"]
