#!/usr/bin/env bash

set -e

COMMAND=${1:-"web"}

case "$COMMAND" in
 web)
   exec gunicorn -c gunicorn.py src.api.app:app
   ;;
 *)
   exec sh -c "$*"
   ;;
esac