# hash:sha256:36bc08be7d7bccbcb14660bf7930d0f6978b62e5f0cf00df511cd29a0a1fb2cb
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/swdb:Python3.12code-server4.133bedrock-auth0.5.0

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential=12.9ubuntu3 \
    && rm -rf /var/lib/apt/lists/*

RUN conda install -y \
        jupyterlab==4.6.3 \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    aind-data-access-api==1.10.0 \
    dynamax[notebooks]==1.0.2 \
    hdmf-zarr==0.13.0 \
    ipykernel==7.3.0 \
    matplotlib==3.11.1 \
    numpy==2.5.2 \
    pandas==3.0.5 \
    pynwb[zarr]==4.0.0 \
    scipy==1.18.1
