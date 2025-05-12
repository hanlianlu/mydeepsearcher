#!/bin/bash

# Update package lists and install required packages
apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 && ldconfig

# Verify if libgthread-2.0.so.0 exists
if [ -f /usr/lib/x86_64-linux-gnu/libgthread-2.0.so.0 ]; then
    echo "libgthread-2.0.so.0 exists"
else
    echo "libgthread-2.0.so.0 does not exist"
fi

# Remove any pathlib.py in site-packages to avoid overriding the standard library
find /tmp/*/antenv/lib/python3.12/site-packages -name "pathlib.py" -delete

# Install Pandoc
PANDOC_VERSION="3.6.4"  # Latest version as requested
wget https://github.com/jgm/pandoc/releases/download/$PANDOC_VERSION/pandoc-$PANDOC_VERSION-1-amd64.deb
dpkg -i pandoc-$PANDOC_VERSION-1-amd64.deb
rm pandoc-$PANDOC_VERSION-1-amd64.deb  # Clean up the downloaded file

# Verify Pandoc installation
pandoc --version

# Start the Streamlit application on port 8000
/tmp/*/antenv/bin/python -m streamlit run deep_searcher_chat/prod_app.py --server.port 8000