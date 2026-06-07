#!/usr/bin/env python3
"""Launcher script for the NeAR Web App."""
import os
import sys
import uvicorn

# Set environment variables
os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0")

from app_web.config import parse_args
from app_web.server import app

if __name__ == "__main__":
    args = parse_args()
    print(f"Starting NeAR Web App on {args.host}:{args.port}...", flush=True)
    uvicorn.run(app, host=args.host, port=args.port)
