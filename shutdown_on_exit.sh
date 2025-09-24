if ! kill -0 "$PID" 2>/dev/null; then
    echo "Process $PID is not running."
    exit 2
fi

while kill -0 "$PID" 2>/dev/null; do
    echo "Process $PID is still running, waiting to shut down..."
    sleep 120
done
echo "Shutting down now..."
sudo shutdown -h now