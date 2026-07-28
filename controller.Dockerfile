FROM python:3.9-slim

ARG WANDB_API_KEY
ENV WANDB_API_KEY=${WANDB_API_KEY}

RUN apt-get update && \
    apt-get install -y git nano vim python3-pip\
    libpcap0.8 libpcap0.8-dev wget\
    adduser libfontconfig1 musl\
    net-tools iputils-ping  tcpdump socat && \
    rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/DISTA-IoT/pox /pox

WORKDIR /pox

RUN rm -r .git

# Full rebuild bust: pass CACHE_BUST=<timestamp> to re-run pip installs
ARG CACHE_BUST=1

RUN pip3 install --upgrade pip

RUN pip3 install torch --index-url https://download.pytorch.org/whl/cpu

# Install requirements before cloning to cache them
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# Code-only bust: pass CODE_BUST=<timestamp> to re-run only the git clone, keeping pip cached.
ARG CODE_BUST=1

ARG CODE_BRANCH=tiger_second_round

RUN git clone --branch ${CODE_BRANCH} https://github.com/DISTA-IoT/smartville-controller.git pox/smartController

WORKDIR /pox/pox/smartController
