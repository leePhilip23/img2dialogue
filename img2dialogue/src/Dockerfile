FROM nvidia/cuda:12.2.0-base-ubuntu22.04
# The installer requires curl (and certificates) to download the release archive
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

RUN uv venv --python 3.11
#RUN source .venv/bin/activate
RUN uv pip install --python 3.11 --index-url https://download.pytorch.org/whl/cu124 \
    torch torchvision torchaudio