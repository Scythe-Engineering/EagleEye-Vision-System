#!/bin/bash
set -e  # Exit immediately if any command fails

ARCH=$(uname -m)

# Map system architecture to the correct package type
case "$ARCH" in
    "x86_64")
        PLATFORM="amd64"
        ;;
    "aarch64")
        PLATFORM="arm64"
        ;;
    "armv7l")
        PLATFORM="armhf"
        ;;
    *)
        echo "Unsupported architecture: $ARCH"
        exit 1
        ;;
esac

# Set the EULA to be automatically accepted
debconf-set-selections <<< "libedgetpu libedgetpu/accepted-eula boolean true"

# Function to get the latest libedgetpu1-std release link
echo "Fetching latest libedgetpu1-max release link for $PLATFORM..."
LATEST_URL=$(curl -s https://api.github.com/repos/DarkEden-coding/libedgetpu-no-interact/releases/latest | 
  grep -oP '"browser_download_url": "\K(https://github.com/DarkEden-coding/libedgetpu-no-interact/releases/download/[^"]+libedgetpu1-max_[^"]+'$PLATFORM'\.deb)' | head -n 1)

if [[ -z "$LATEST_URL" ]]; then
    echo "Error: Could not find the latest libedgetpu1-max package for $PLATFORM. Check the repository manually."
    exit 1
fi

echo "Downloading $LATEST_URL..."
wget -O libedgetpu1-max.bookworm_"$PLATFORM".deb "$LATEST_URL"

# Install the package
echo "Installing libedgetpu1-max..."
dpkg -i libedgetpu1-max.bookworm_"$PLATFORM".deb

# Activate virtual environment
VENV_PATH="/root/EagleEye-Object-Detection/venv/bin/activate"
if [[ -f "$VENV_PATH" ]]; then
    echo "Activating virtual environment..."
    source "$VENV_PATH"
else
    echo "Error: Virtual environment not found at $VENV_PATH."
    exit 1
fi

# Uninstall existing TensorFlow packages
echo "Uninstalling TensorFlow..."
pip uninstall -y tensorflow tensorflow-aarch64 || true

# Install tflite-runtime
echo "Installing tflite-runtime..."
pip install -U tflite-runtime

echo "Setup complete! Your Coral TPU is ready."
