FROM python:3.9-slim

ARG WANDB_API_KEY
ENV WANDB_API_KEY=${WANDB_API_KEY}

RUN apt-get update && \
    apt-get install -y git nano vim python3-pip\
    libpcap0.8 libpcap0.8-dev wget\
    adduser libfontconfig1 musl\
    net-tools iputils-ping  tcpdump socat && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip

RUN pip3 install torch --index-url https://download.pytorch.org/whl/cpu

RUN git clone https://github.com/DISTA-IoT/pox /pox

WORKDIR /pox

RUN rm -r .git

ARG CACHE_BUST=1
RUN git clone https://github.com/DISTA-IoT/smartville-controller.git pox/smartController -b new_smartville

WORKDIR /pox/pox/smartController

RUN pip install -r requirements.txt


