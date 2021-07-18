"""gunicorn ASGI server configuration."""
from multiprocessing import cpu_count
from os import environ


def max_workers():
    return cpu_count()


bind = '0.0.0.0:' + environ.get('PORT', '8000')
worker_class = 'uvicorn.workers.UvicornWorker'
workers = max_workers()
keepalive = 10
threads = 20
max_requests = 1000
pidfile = "/tmp/asgi.pid"
log_level = environ.get('LOGLEVEL', 'INFO')
