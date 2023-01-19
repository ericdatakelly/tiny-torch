# Make base container to run various files

FROM continuumio/miniconda3

# Get the modeling repo
RUN git clone https://github.com/ericdatakelly/tiny-torch.git && cd tiny-torch

# Make a volume mountable
# Remember to use `-v host/path:container/path` with `docker run ...`
VOLUME /tiny-torch/logs

# Set the working directory
WORKDIR /tiny-torch

## Create the environment:
RUN conda install -c conda-forge mamba
RUN mamba env create -f environment.yaml

# Make RUN commands use the new environment:
RUN echo "conda activate ignite-env" >> ~/.bashrc
SHELL ["conda", "run", "-n", "ignite-env", "/bin/bash", "--login", "-c"]

# Install the local project into the conda env
RUN pip install -e .
