#!/bin/bash
# startup.sh

echo "Starting Ollama server in the background..."
ollama serve &

# Wait for the server to start (give it a few seconds)
sleep 5

echo "Pulling gemma3:1b..."
ollama pull gemma3:1b

echo "Ollama container is ready to go!"

# Keep the container running by bringing the server to the foreground
# or using a wait command to keep the script alive
wait