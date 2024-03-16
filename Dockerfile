


#######################################################################################################
FROM nvcr.io/nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04
LABEL maintainer="bigscience-workshop"
LABEL repository="petals"

WORKDIR /home
# Set en_US.UTF-8 locale by default
RUN echo "LC_ALL=en_US.UTF-8" >> /etc/environment

# Install curl, gnupg2, and other dependencies for adding the NVIDIA repository and toolkit
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gnupg2 \
    build-essential \
    wget \
    git \
    && apt-get clean autoclean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Add the NVIDIA container toolkit repository
RUN curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
    && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
       sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
       tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Update the package list
RUN apt-get update

# Install the NVIDIA Container Toolkit
RUN apt-get install -y nvidia-container-toolkit

# Continue with your original setup
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O install_miniconda.sh && \
    bash install_miniconda.sh -b -p /opt/conda && rm install_miniconda.sh
ENV PATH="/opt/conda/bin:${PATH}"

RUN conda install python~=3.10.12 pip && \
    pip install --no-cache-dir "torch>=1.12" && \
    conda clean --all && rm -rf ~/.cache/pip

VOLUME /cache
ENV PETALS_CACHE=/cache

COPY . petals/
RUN pip install petals

WORKDIR /home/petals/

# Replace the previous CMD with the new command
CMD ["sh", "-c", "python -m petals.cli.run_server deepseek-ai/deepseek-coder-7b-instruct-v1.5 --port 31337 --initial_peers /ip4/45.79.153.218/tcp/31337/p2p/QmXfANcrDYnt5LTXKwtBP5nsTMLQdgxJHbK3L1hZdFN8km  --num_blocks 10 --stats_report_interval 2"]
# CMD bash
#incase of fatal error uncomment the CMD BASH and comment out the CMD above