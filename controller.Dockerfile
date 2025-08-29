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

RUN git clone https://github.com/noxrepo/pox.git /pox

# Download and install Prometheus
RUN wget https://github.com/prometheus/prometheus/releases/download/v2.49.1/prometheus-2.49.1.linux-amd64.tar.gz && \
    tar -xvf prometheus-2.49.1.linux-amd64.tar.gz && \
    cp prometheus-2.49.1.linux-amd64/prometheus /usr/local/bin/ && \
    cp prometheus-2.49.1.linux-amd64/promtool /usr/local/bin/


# Download and install Grafana
RUN wget https://dl.grafana.com/oss/release/grafana_10.3.1_amd64.deb && \
    dpkg -i grafana_10.3.1_amd64.deb
    
# Expose the Prometheus port
EXPOSE 9090
# Expose Grafana port
EXPOSE 3000


# Clean up unnecessary files
RUN rm -rf prometheus-2.34.0.linux-amd64.tar.gz prometheus-2.34.0.linux-amd64

WORKDIR /pox

RUN git checkout gar-experimental 

RUN rm -r .git

ARG CACHE_BUST=1
RUN git clone https://github.com/DISTA-IoT/smartville-controller.git pox/smartController -b new_smartville

WORKDIR /pox/pox/smartController

RUN pip install -r requirements.txt


