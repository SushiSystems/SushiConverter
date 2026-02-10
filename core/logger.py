import sys

YELLOW = "\033[93m"
RED = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"

def log_info(message):
    print(f"[INFO] {message}")

def log_warning(message):
    print(f"{YELLOW}[WARNING] {message}{RESET}")

def log_error(message):
    print(f"{RED}[ERROR] {message}{RESET}")

def log_success(message):
    print(f"{GREEN}[SUCCESS] {message}{RESET}")

def set_color_mode():
    if sys.platform == "win32":
        import os
        os.system("color")
