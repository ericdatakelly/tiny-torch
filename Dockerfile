# Make base container to run various files

#FROM continuumio/miniconda3
FROM python:3.9

# Make the directory to ensure permissions for current user
RUN mkdir tiny-torch

# Get the modeling repo
RUN git clone https://github.com/ericdatakelly/tiny-torch.git && cd tiny-torch

RUN mkdir tiny-torch/logs

# Make a volume mountable
# Remember to use `-v host/path:container/path` with `docker run ...`
VOLUME /tiny-torch/logs

# Set the working directory
WORKDIR /tiny-torch

# Install required packages
RUN pip install -r requirements.txt

# Install this package too
RUN pip install .
