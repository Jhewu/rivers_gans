#!/bin/bash

# Set the PID manually within the script
PID=110932  # Replace 105041 with the actual PID you want to wait for

# Function to check if a process with a given PID is running
is_process_running() {
    ps -p $PID > /dev/null 2>&1
}

# Wait for the process with the specified PID to complete
echo "Waiting for the process with PID $PID to complete..."
while is_process_running; do
    sleep 1
done

# Execute the command after the process has completed
echo "The process with PID $PID has completed."
echo "Executing the next command..."

# Run a command in the background (e.g., sleep for 100 seconds)
python3 gan.py &

# Capture the PID of the background process
PID=$!

# Print the PID
echo "The PID of the background process is: $PID"

# Perform some action (e.g., wait for the process to complete)
wait $PID

# Print a message after the process completes
echo "The background process with PID $PID has completed."
