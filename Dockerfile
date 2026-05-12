FROM jupyter/base-notebook:x86_64-ubuntu-22.04

# Install dependencies as root
USER root

# Optional: set Python version inside conda (base or custom)
ARG PYTHON_VERSION=3.11.6

# Activate base environment and install python version
RUN conda install -y python=${PYTHON_VERSION}

# Copy environment file first (so Docker caches installations)
COPY environment.yml /tmp/environment.yml

# Update env BEFORE copying project files
RUN conda env update -n base -f /tmp/environment.yml --prune

# Install PyTorch CUDA version
RUN pip install --no-cache-dir torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# Now copy the project code (last layer, so dependency caching is preserved)
COPY . /home/jovyan/work/

# Return to notebook user
USER $NB_UID