# hash:sha256:aae7d23e3b1c8e52172cd8c977a796cf79b8e04384a1207e313c5f90da1d97b0
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.0.9-miniconda4.9.2-python3.8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

ENV PROCESS_NAME=aind-fip-qc-raw

RUN pip install -U --no-cache-dir \
    aind-data-schema==1.2.0 \
    aind-data-schema-models==0.7.0 \
    aws-requests-auth==0.4.3 \
    boto3==1.35.81 \
    kachery-cloud==0.4.10 \
    matplotlib==3.7.5 \
    numpy==1.24.4 \
    pandas==2.0.3 \
    pyyaml==6.0.3 \
    watchtower==3.4.0

ADD "https://github.com/coder/code-server/releases/download/v4.95.3/code-server-4.95.3-linux-amd64.tar.gz" /.code-server/code-server.tar.gz

RUN cd /.code-server \
    && tar -xvf code-server.tar.gz \
    && rm code-server.tar.gz \
    && ln -s /.code-server/code-server-4.95.3-linux-amd64/bin/code-server /usr/bin/code-server \
    && mkdir -p /.vscode/extensions

RUN code-server --extensions-dir="/.vscode/extensions" --install-extension continue.continue \
    && code-server --extensions-dir="/.vscode/extensions" --install-extension ms-python.python \
    && code-server --extensions-dir="/.vscode/extensions" --install-extension ms-toolsai.jupyter \
    && code-server --extensions-dir="/.vscode/extensions" --install-extension reageyao.biosyntax \
    && code-server --extensions-dir="/.vscode/extensions" --install-extension reditorsupport.r \
    && code-server --extensions-dir="/.vscode/extensions" --install-extension saoudrizwan.claude-dev

COPY postInstall /
RUN --mount=type=secret,id=secrets . /run/secrets/secrets \
    && /postInstall
