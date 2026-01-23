FROM continuumio/miniconda3

WORKDIR /usr

# # Install R (required for rpy2)
# RUN apt-get update && apt-get install -y \
#     r-base \
#     r-base-dev \
#     && rm -rf /var/lib/apt/lists/*

COPY environment.yml .
COPY packages/ ./packages/

RUN conda env create -f environment.yml

SHELL ["conda", "run", "-n", "iTS", "/bin/bash", "-c"]

RUN mkdir -p /opt/trained_models
COPY data/ ./data/
COPY src/ ./src/

ENV PATH=/opt/conda/envs/iTS/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/conda/envs/iTS/lib
ENV LD_PRELOAD=/opt/conda/envs/iTS/lib/libstdc++.so.6
ENV IS_DOCKER=1

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "iTS"]