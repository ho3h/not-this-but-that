#!/bin/bash
# Helper for the probe daemon. Detaches the daemon so it survives
# the harness's background-task lifetime cap.
#
# Usage:
#   scripts/probe_run.sh start    # start detached; returns immediately
#   scripts/probe_run.sh status   # check if daemon is up
#   scripts/probe_run.sh stop     # graceful stop
#   scripts/probe_run.sh restart  # stop + start
#   scripts/probe_run.sh ensure   # start if not running, no-op if running

set -e
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

LOG=/tmp/probe_daemon.log
PIDFILE=/tmp/probe_daemon.pid

is_up() {
    [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null
}

start() {
    if is_up; then
        echo "daemon already running (PID $(cat $PIDFILE))"
        return 0
    fi
    HF_TOKEN=$(grep -E '^HF_TOKEN=' .env 2>/dev/null | cut -d= -f2-)
    export HF_TOKEN
    # nohup makes the process ignore SIGHUP; `&` puts it in background;
    # `disown` removes it from the shell's job table. Combined, it survives
    # the parent shell's death on macOS.
    nohup .venv/bin/python scripts/probe_daemon.py > "$LOG" 2>&1 &
    PID=$!
    disown "$PID" 2>/dev/null || true
    echo "$PID" > "$PIDFILE"
    echo "started PID $(cat $PIDFILE)"
    echo "waiting for ready signal…"
    for i in {1..60}; do
        if grep -q "probe daemon listening" "$LOG" 2>/dev/null; then
            echo "  ready after ${i}s"
            return 0
        fi
        sleep 1
    done
    echo "  TIMEOUT: daemon not ready after 60s — see $LOG"
    tail -10 "$LOG"
    return 1
}

stop() {
    if ! is_up; then
        echo "not running"
        rm -f "$PIDFILE"
        return 0
    fi
    PID=$(cat "$PIDFILE")
    # Graceful first
    curl -s -X POST -H 'Content-Type: application/json' \
         -d '{"cmd": "stop"}' http://127.0.0.1:8765/probe > /dev/null 2>&1 || true
    sleep 2
    if kill -0 "$PID" 2>/dev/null; then
        kill "$PID" 2>/dev/null || true
        sleep 1
        kill -9 "$PID" 2>/dev/null || true
    fi
    rm -f "$PIDFILE"
    echo "stopped"
}

status() {
    if is_up; then
        echo "UP   PID=$(cat $PIDFILE)"
        curl -s -X POST -H 'Content-Type: application/json' \
             -d '{"cmd": "ping"}' http://127.0.0.1:8765/probe 2>&1 | head -1
    else
        echo "DOWN"
    fi
}

case "${1:-status}" in
    start)   start ;;
    stop)    stop ;;
    restart) stop; start ;;
    status)  status ;;
    ensure)  is_up || start ;;
    *) echo "usage: $0 {start|stop|status|restart|ensure}"; exit 1 ;;
esac
