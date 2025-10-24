#!/bin/bash

# RunPod Instance Setup Script with Python 3.11 and uv
# This script installs Python 3.11, uv, and uses uv to install Python requirements

set -e  # Exit on error

echo "=== Starting RunPod Instance Setup ==="

# Update system packages
echo "Updating system packages..."
apt-get update
apt-get upgrade -y

# Install dependencies
echo "Installing dependencies..."
apt-get install -y software-properties-common curl

# Add deadsnakes PPA for Python 3.11
echo "Adding deadsnakes PPA..."
add-apt-repository ppa:deadsnakes/ppa -y
apt-get update

# Install Python 3.11
echo "Installing Python 3.11..."
apt-get install -y python3.11 python3.11-venv python3.11-dev

# Set Python 3.11 as default (optional)
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
update-alternatives --set python3 /usr/bin/python3.11

# Verify Python installation
echo "Verifying Python installation..."
python3 --version

## Install uv
#echo "Installing uv..."
#curl -LsSf https://astral.sh/uv/install.sh | sh
#
## Add uv to PATH for current session
#export PATH="$HOME/.cargo/bin:$PATH"

# Verify uv installation
echo "Verifying uv installation..."
uv --version

# Install Python requirements
if [ -f "requirements.txt" ]; then
    echo "Installing requirements from requirements.txt..."
    uv pip install --python python3.11 -r requirements.txt
else
    echo "Warning: requirements.txt not found in current directory"
    echo "Please ensure requirements.txt is present or modify the script"
fi

echo "=== Setup Complete ==="
echo "Python version: $(python3 --version)"
echo "uv version: $(uv --version)"

echo "Login into wandb and huggingface!"