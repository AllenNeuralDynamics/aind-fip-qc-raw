# hash:sha256:6b57a5906fe75cb59826af7240e496290c61665b0c9db8398fa9ae2bbb4f1c84
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.0.9-miniconda4.9.2-python3.8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    aind-data-access-api[docdb]==0.16.1 \
    aind-data-schema==1.2.0 \
    aind-data-schema-models==0.7.0 \
    aind-log-utils==0.2.1 \
    aws-requests-auth==0.4.3 \
    boto3==1.35.81 \
    kachery-cloud==0.4.10 \
    matplotlib==3.7.5 \
    numpy==1.24.4 \
    pandas==2.0.3

COPY postInstall /
RUN /postInstall
