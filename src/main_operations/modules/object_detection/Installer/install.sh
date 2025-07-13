#!/bin/bash
set -e  # Exit immediately if any command fails

echo "Updating and upgrading the system..."
apt update && apt upgrade -y

echo "Installing libgl1 (required for YOLO)..."
apt install -y libgl1

echo "Installing Python3 and pip..."
apt install python3-pip -y

echo "Installing python venv..."
apt install python3-venv -y

echo "Installing git..."
apt install git -y

echo "Installing curl..."
apt install curl -y

REPO_PATH="/root/EagleEye-Object-Detection"

if [ -d "$REPO_PATH" ]; then
    echo "Removing existing repository..."
    rm -rf "$REPO_PATH"
fi

echo "Fetching latest release tag from GitHub..."
# Query the GitHub API and extract the tag name
LATEST_TAG=$(curl --silent "https://api.github.com/repos/frc3322/EagleEye-Object-Detection/releases/latest" | \
             grep '"tag_name":' | sed -E 's/.*"([^"]+)".*/\1/')
if [ -z "$LATEST_TAG" ]; then
    echo "Error: Could not determine the latest release tag."
    exit 1
fi
echo "Latest release tag is: $LATEST_TAG"

echo "Cloning Repo from tag $LATEST_TAG..."
git clone --branch "$LATEST_TAG" --single-branch "https://github.com/frc3322/EagleEye-Object-Detection.git" "$REPO_PATH"

echo "Navigating to EagleEye-Object-Detection directory..."
cd "$REPO_PATH" || { echo "Failed to change directory to $REPO_PATH"; exit 1; }

echo "Creating and activating Python virtual environment..."
python3 -m venv venv
. "$REPO_PATH/venv/bin/activate"

echo "Installing PyTorch..."
pip3 install --upgrade --force-reinstall typing-extensions
pip3 install torch torchvision torchaudio --no-cache-dir --index-url https://download.pytorch.org/whl/cpu

echo "Installing project dependencies..."
pip3 install -r requirements.txt

echo "Copying service files to systemd directory..."
cp "$REPO_PATH/Installer/EagleEye.service" /etc/systemd/system/
cp "$REPO_PATH/Installer/EagleEyeUpdater.service" /etc/systemd/system/

echo "Reloading systemd daemon..."
systemctl daemon-reload

echo "Enabling services..."
systemctl enable EagleEye.service
systemctl enable EagleEyeUpdater.service

echo "Starting services..."
systemctl start EagleEye.service
systemctl start EagleEyeUpdater.service

echo "Setup complete."
