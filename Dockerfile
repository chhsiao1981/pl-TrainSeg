# Python version can be changed, e.g.
# FROM python:3.8
# FROM ghcr.io/mamba-org/micromamba:1.5.1-focal-cuda-11.3.1
FROM tensorflow/tensorflow:2.14.0-gpu

LABEL org.opencontainers.image.authors="FNNDSC <alan.rivasmunoz@childrens.harvard.edu>" \
      org.opencontainers.image.title="My ChRIS Plugin" \
      org.opencontainers.image.description="A ChRIS plugin to do something awesome"

ARG SRCDIR=/usr/local/src/pl-TrainSeg
WORKDIR ${SRCDIR}

COPY requirements.txt .
RUN --mount=type=cache,sharing=private,target=/root/.cache/pip pip install -r requirements.txt

COPY . .
ARG extras_require=none
RUN pip install ".[${extras_require}]" \
    && cd / && rm -rf ${SRCDIR}
WORKDIR /

CMD ["SegTrain"]
