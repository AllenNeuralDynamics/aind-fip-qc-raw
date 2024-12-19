# hash:sha256:97e59fb1dec6bbb434ed8d4a126f319eb84df3522e42724401c13d86fa209e2b
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.0.9-miniconda4.9.2-python3.8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG AWS_ACCESS_KEY_ID
ARG AWS_DEFAULT_REGION
ARG AWS_SECRET_ACCESS_KEY

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

ADD "https://github.com/coder/code-server/releases/download/v4.9.0/code-server-4.9.0-linux-amd64.tar.gz" /.code-server/code-server.tar.gz
	
RUN cd /.code-server \
	&& tar -xvf code-server.tar.gz \
	&& rm code-server.tar.gz \
	&& ln -s /.code-server/code-server-4.9.0-linux-amd64/bin/code-server  /usr/bin/code-server
