FROM nvidia/cuda:11.3.0-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteraction
ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# update and install
RUN apt-get update -y && apt-get upgrade -y && apt-get install -y \
    wget \
    curl \
    git \
    python-openssl \
    xvfb \
    python-opengl \
    node-gyp \
    icewm \
    python3 \
    python3-pip \
    htop

# install jupyter lab
RUN apt-get update -y && apt-get install -y \
    libgl1-mesa-glx wget curl git tmux imagemagick htop libsndfile1\
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
RUN pip install jupyterlab

# install nodejs
RUN wget https://raw.githubusercontent.com/nvm-sh/nvm/v0.38.0/install.sh | bash && \
    exec bash && \
    . ~/.bashrc && \
    nvm install v14.17.0

# install code server
# https://github.com/jupyterhub/jupyter-server-proxy
RUN pip install jupyter-server-proxy
RUN pip install jupyter-vscode-proxy
RUN curl -fsSL https://code-server.dev/install.sh | sh
# # RUN curl -fOL https://github.com/cdr/code-server/releases/download/v3.4.1/code-server_3.10.2_amd64.deb
# # RUN dpkg -i code-server_3.4.1_amd64.deb
# # RUN rm -r code-server_3.4.1_amd64.deb
RUN code-server \
    --install-extension ms-python.python \
    --install-extension ms-ceintl.vscode-language-pack-ja

# install python library
RUN pip install \
    opencv-python \
    numpy \
    pandas \
    matplotlib \
    scipy \
    scikit-learn \
    tqdm \
    torchvision \
    h5py \
    tifffile \
    tensorboardX \
    scikit-image

# install pytorch
RUN pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# matplotlib日本語化
RUN pip install japanize-matplotlib

RUN apt-get update -y && \
    apt-get install -y \
    screen

ARG USER_ID=1008
ARG GROUP_ID=1009
ENV USER_NAME=jovyan
RUN groupadd -g ${GROUP_ID} ${USER_NAME} && \
    useradd -d /home/${USER_NAME} -m -s /bin/bash -u ${USER_ID} -g ${GROUP_ID} ${USER_NAME}
WORKDIR /home/${USER_NAME}

USER ${USER_NAME}
ENV HOME /home/${USER_NAME}

RUN mkdir workdir

USER jovyan

ENV NB_PREFIX /
ENV PYTHONPATH "/home/${USER_NAME}/workdir:$PYTHONPATH"
ENV PYTHONPATH "/home/${USER_NAME}/workdir/pnp:$PYTHONPATH"
ENV DISPLAY=":1.0"

CMD ["sh","-c", "jupyter lab --notebook-dir=/home/jovyan --ip=0.0.0.0 --no-browser --allow-root --port=8888 --NotebookApp.token='' --NotebookApp.password='' --NotebookApp.allow_origin='*' --NotebookApp.base_url=${NB_PREFIX}"]
