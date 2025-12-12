import sys

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(text):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}")
    print(f"{text.center(60)}")
    print(f"{'='*60}{Colors.ENDC}")

def print_step(step_num, total_steps, text):
    print(f"{Colors.CYAN}[Step {step_num}/{total_steps}] {text}{Colors.ENDC}")

def print_success(text):
    print(f"{Colors.GREEN}✔ {text}{Colors.ENDC}")

def print_warning(text):
    print(f"{Colors.WARNING}⚠ {text}{Colors.ENDC}")

def print_error(text):
    print(f"{Colors.FAIL}✖ {text}{Colors.ENDC}")

def print_info(key, value):
    print(f"  -> {Colors.BLUE}{key}:{Colors.ENDC} {value}")
