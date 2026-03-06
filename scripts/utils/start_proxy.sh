#!/bin/bash
# Shadowsocks Proxy Start Script
# Usage: source scripts/start_proxy.sh

PROXY_PORT=1081
SSLOCAL_BIN="/tmp/sslocal"
SS_SERVER="UD2RdK31TK6zLVZikR.xf9pzeslxw.sbs:16001"
SS_PASSWORD="cb1d3163-2bbd-4e13-83b5-19bc75647cb1"
SS_METHOD="aes-256-gcm"

# Check if already running
if pgrep -f "sslocal.*$PROXY_PORT" > /dev/null; then
    echo "Proxy already running on port $PROXY_PORT"
    export http_proxy=http://127.0.0.1:$PROXY_PORT
    export https_proxy=http://127.0.0.1:$PROXY_PORT
    echo "Proxy environment variables set"
    return
fi

# Start shadowsocks with HTTP protocol
echo "Starting Shadowsocks proxy..."
$SSLOCAL_BIN -s "$SS_SERVER" -k "$SS_PASSWORD" -m "$SS_METHOD" \
    -b 127.0.0.1:$PROXY_PORT --protocol http > /tmp/ss.log 2>&1 &

sleep 2

# Check if started
if pgrep -f "sslocal.*$PROXY_PORT" > /dev/null; then
    echo "✓ Proxy started on port $PROXY_PORT"

    # Set environment variables
    export http_proxy=http://127.0.0.1:$PROXY_PORT
    export https_proxy=http://127.0.0.1:$PROXY_PORT

    echo "✓ Proxy environment variables set"
    echo ""
    echo "Testing connection..."
    curl -x http://127.0.0.1:$PROXY_PORT -s -o /dev/null -w "  Google: %{http_code}\n" https://www.google.com
    curl -x http://127.0.0.1:$PROXY_PORT -s -o /dev/null -w "  GitHub: %{http_code}\n" https://github.com
else
    echo "✗ Failed to start proxy"
    return 1
fi
