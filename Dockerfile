FROM nvidia/cuda:12.1.0-devel-ubuntu18.04

RUN apt-get update
RUN apt-get install -y gcc make python3.8 python3-pip

COPY . .

RUN python3.8 -m pip install -U pip
RUN python3.8 -m pip install poetry
RUN python3.8 -m poetry install

EXPOSE 5000
