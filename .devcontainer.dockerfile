FROM graphcore/pytorch:3.3.0-ubuntu-20.04-20230703

RUN apt-get update \
    && apt-get install -y \
        clang-format \
        sudo \
    && apt-get install -y \
        git \
        sudo \
    && apt-get clean

# Snippet from https://code.visualstudio.com/remote/advancedcontainers/add-nonroot-user
ARG USERNAME=user-name-goes-here
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME
USER $USERNAME

ADD . /tmp/besskge
RUN pip install -r /tmp/besskge/requirements-dev.txt \
    jupyterlab
