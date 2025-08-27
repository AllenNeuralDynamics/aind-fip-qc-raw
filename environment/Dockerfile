# hash:sha256:1de6362a1e9c3284f648b0869a4bb0e423e34380640014a5999ea37229dbded5
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:24.5.0-0-python3.12.4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    aind-data-access-api[docdb]==0.16.1 \
    aind-data-schema==2.0.1 \
    aind-log-utils==0.2.4 \
    aws-requests-auth==0.4.3 \
    boto3==1.35.81 \
    kachery-cloud==0.4.10 \
    matplotlib==3.10.5 \
    numpy==2.3.2 \
    pandas==2.3.2

ADD "https://github.com/coder/code-server/releases/download/v4.100.3/code-server-4.100.3-linux-amd64.tar.gz" /.code-server/code-server.tar.gz
	
RUN cd /.code-server \
    && tar -xvf code-server.tar.gz \
    && rm code-server.tar.gz \
    && ln -s /.code-server/code-server-4.100.3-linux-amd64/bin/code-server /usr/bin/code-server \
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
