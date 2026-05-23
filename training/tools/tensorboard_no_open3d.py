#!/usr/bin/env python3
"""
TensorBoard entry that skips the Open3D dynamic plugin.

Open3D registers a ``tensorboard_plugins`` entry point; importing it loads
``open3d``'s native extension and can hit Intel oneMKL / OpenMP clashes in
conda + PyTorch environments. Training logs only need the stock TensorBoard
plugins, so we drop that one plugin.
"""

from __future__ import annotations

import sys

import pkg_resources
from absl import app

from tensorboard import default
from tensorboard import main_lib
from tensorboard import program
from tensorboard.plugins import base_plugin
from tensorboard.util import tb_logging

logger = tb_logging.get_logger()


def get_plugins_without_open3d():
    static = default.get_static_plugins()
    dynamic = [
        ep.resolve()
        for ep in pkg_resources.iter_entry_points("tensorboard_plugins")
        if ep.name.lower() != "open3d"
    ]
    return static + dynamic


def run_main():
    main_lib.global_init()
    tensorboard = program.TensorBoard(plugins=get_plugins_without_open3d())
    try:
        app.run(tensorboard.main, flags_parser=tensorboard.configure)
    except base_plugin.FlagsError as e:
        print("Error: %s" % e, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    run_main()
