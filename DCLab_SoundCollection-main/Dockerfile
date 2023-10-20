FROM ubuntu:20.04


RUN apt update -y && \
    apt install software-properties-common -y && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt install python3.9 -y && \
    git clone https://github.com/Lfollow-CS/DCLab_SoundCollection.git && \
    pip install -r requirement.txt