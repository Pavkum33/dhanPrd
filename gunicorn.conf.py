import os

# Basic configuration for Railway
bind = f"0.0.0.0:{os.getenv('PORT', '5000')}"
workers = 1
worker_class = "eventlet"
timeout = 30