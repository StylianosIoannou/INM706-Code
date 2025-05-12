#shell script to create a virtual environment and install dependencies
#!/bin/bash
# Create virtual environment
python -m venv venv
# Activate environment
source venv/bin/activate
# Upgrade pip
pip install --upgrade pip
# Install required packages
pip install -r requirements.txt