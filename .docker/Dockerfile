FROM python:3.11-slim

ARG UID=1000
ARG GID=1000

RUN apt-get update && apt-get install -y --no-install-recommends \
    fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

RUN (getent group ${GID} || groupadd -g ${GID} app) \
    && useradd -m -u ${UID} -g ${GID} app
USER app
WORKDIR /work
