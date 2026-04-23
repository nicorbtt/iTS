FROM continuumio/miniconda3

WORKDIR /usr

# # Install R (required for rpy2)
# RUN apt-get update && apt-get install -y \
#     r-base \
#     r-base-dev \
#     && rm -rf /var/lib/apt/lists/*

COPY environment.yml .
COPY packages/ ./packages/

RUN conda install -n base -c conda-forge -y libgcc-ng libstdcxx-ng && \
	conda clean -afy

RUN conda env create -f environment.yml

RUN conda run -n ilocglob Rscript -e "install.packages(c('smooth', 'forecast', 'nloptr'), repos='https://cloud.r-project.org')"

SHELL ["conda", "run", "-n", "ilocglob", "/bin/bash", "-c"]

RUN mkdir -p /opt/trained_models
COPY data/ ./data/
COPY src/ ./src/

ENV PATH=/opt/conda/envs/ilocglob/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/conda/envs/ilocglob/lib
ENV LD_PRELOAD=/opt/conda/envs/ilocglob/lib/libstdc++.so.6
ENV IS_DOCKER=1

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "ilocglob"]