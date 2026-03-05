#!/bin/bash
# Setup environment for AutoDL

echo "Setting up Python environment..."

# Install pip
curl https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
python3 /tmp/get-pip.py --user

# Add to PATH
export PATH=$HOME/.local/bin:$PATH
echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install openai-clip transformers scipy tqdm pillow

echo "Setup complete! Please run: source ~/.bashrc"
