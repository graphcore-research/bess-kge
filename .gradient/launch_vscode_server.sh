#!/bin/bash
# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

cd /notebooks

CODE_FILE=/storage/.vscode/code
if [ ! -f "$CODE_FILE" ]; then
    mkdir -p /storage/.vscode/server
    curl -Lk 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64' --output /storage/vscode_cli.tar.gz
    tar -xf /storage/vscode_cli.tar.gz -C /storage/.vscode
fi

ln -s /storage/.vscode/server $HOME/.vscode-server

pip install -r requirements-dev.txt

$CODE_FILE tunnel --accept-server-license-terms --name=${1:-ipu-paperspace}