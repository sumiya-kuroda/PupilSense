#!/bin/bash

# Copy example.env to .env if it doesn't already exist
if [ ! -f .env ]; then
    cp example.env .env
    echo "Created .env file. Please configure it with your specific settings."
else
    echo ".env file already exists. Please ensure it has the correct settings."
fi

echo "Setup completed successfully."