# --------------------------------------------------------------------------
# logger.py
# --------------------------------------------------------------------------
# This file is part of:
# SushiConverter
# https://github.com/SushiSystems/SushiConverter
# https://sushisystems.io
# --------------------------------------------------------------------------
# Copyright (c) 2026-present  Mustafa Garip & Sushi Systems
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# --------------------------------------------------------------------------

import sys

YELLOW = "\033[93m"
RED = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"

def log_info(message):
    """
    Prints an info log.
    @param message Log message.
    """
    print(f"[INFO] {message}")

def log_warning(message):
    """
    Prints a warning log in yellow.
    @param message Log message.
    """
    print(f"{YELLOW}[WARNING] {message}{RESET}")

def log_error(message):
    """
    Prints an error log in red.
    @param message Log message.
    """
    print(f"{RED}[ERROR] {message}{RESET}")

def log_success(message):
    """
    Prints a success log in green.
    @param message Log message.
    """
    print(f"{GREEN}[SUCCESS] {message}{RESET}")

def set_color_mode():
    """
    Initializes OS color support.
    """
    if sys.platform == "win32":
        import os
        os.system("color")
