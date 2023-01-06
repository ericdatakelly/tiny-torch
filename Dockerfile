FROM condaforge/mambaforge

#VOLUME /tiny-torch

WORKDIR /app

# Create the environment:
COPY environment.yaml .
RUN mamba env create -f environment.yaml

# Make RUN commands use the new environment:
RUN echo "conda activate ignite-env" >> ~/.bashrc
SHELL ["conda", "run", "-n", "ignite-env", "/bin/bash", "--login", "-c"]

## Test that this is working
#RUN python -c "import yaml"

RUN git clone https://github.com/ericdatakelly/tiny-torch.git && cd tiny-torch


# The code to run when container is started:
COPY entrypoint.sh ./
ENTRYPOINT ["./entrypoint.sh"]

#RUN python tiny_torch/main.py
