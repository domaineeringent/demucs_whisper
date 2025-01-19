#!/bin/bash

# Function to stop the application
stop_app() {
    echo "Stopping application..."
    if [ -f app.pid ]; then
        pid=$(cat app.pid)
        if kill -0 $pid 2>/dev/null; then
            kill $pid
            while kill -0 $pid 2>/dev/null; do
                sleep 1
            done
        fi
        rm app.pid
    fi
}

# Function to start the application
start_app() {
    echo "Starting application..."
    python3 -u -m uvicorn app:app --host 0.0.0.0 --port 8000 &
    echo $! > app.pid
}

# Handle SIGTERM gracefully
trap 'stop_app; exit 0' SIGTERM

# If running in Docker (as PID 1)
if [ $$ -eq 1 ]; then
    start_app
    # Keep the script running to handle signals
    while true; do
        sleep 1
    done
else
    # Running manually
    stop_app
    start_app
    echo "Application restarted!"
fi 
