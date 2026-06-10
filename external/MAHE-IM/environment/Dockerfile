# hash:sha256:f3a796153ed514655469c711f1542cb158b6d962d728ce0dc3677db4f0cfdd22
FROM registry.codeocean.com/codeocean/miniconda3:4.5.11-python3.7-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN conda install -y \
        gensim==3.8.0 \
        joblib==0.14.1 \
        networkx==1.11 \
        pandas==1.0.1 \
        tensorflow==1.13.1 \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    numpy==1.19.4 \
    torch==1.7.1 \
    torchvision==0.8.2

COPY postInstall /
RUN /postInstall
