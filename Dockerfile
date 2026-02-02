FROM pytorch/pytorch:2.5.1-cuda11.8-cudnn9-devel

RUN apt-get update \
    && apt-get install -y \
        apt-transport-https \
        libtcmalloc-minimal4 \
        libomp-dev \
        sox \
        git \
        gcc \
        g++ \
        python3-dev \
        ffmpeg \
        libsm6 \
        libxext6 \
    && apt-get clean

RUN ln -snf /usr/share/zoneinfo/Etc/UTC /etc/localtime \
    && echo "Etc/UTC" > /etc/timezone \
    && apt-get update \
    && apt-get upgrade -y \
    && apt-get install texlive-latex-base texlive-latex-extra texlive-fonts-recommended xzdec dvipng cm-super -y \
    && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y libglib2.0-0 libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

WORKDIR /workspace

RUN mkdir /assets

COPY requirements.txt /assets/requirements.txt
RUN pip install -r /assets/requirements.txt --upgrade --no-cache-dir

# perception encoder package from meta
RUN cd /home/ && git clone https://github.com/facebookresearch/perception_models.git && cd perception_models && pip install -e .
RUN cd /workspace

COPY . /workspace/
RUN git config --global --add safe.directory '*'

ENV TOKENIZERS_PARALLELISM false
ENV ACCELERATE_LOG_LEVEL=ERROR
ENV TRANSFORMERS_VERBOSITY=error
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV PYTHONPATH "$PYTHONPATH:./"
