#!/usr/bin/env python3
import os
import sys
import time
import signal
import subprocess
import argparse
import threading
from datetime import datetime

# ANSI color codes for pretty output
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    RESET = '\033[0m'


def colored(text, color):
    return f"{color}{text}{Colors.RESET}"


def print_banner():
    banner = """
    |---------------------------------------------------------------|
    |                                                               |
    |   ⚡ CROSS-BORDER ARBITRAGE DETECTION SYSTEM ⚡                 |
    |                                                               |
    |   C++ Engine | Redpanda Streaming | QuestDB Storage           |
    |                                                               |
    |---------------------------------------------------------------|
    """
    print(colored(banner, Colors.CYAN))


def timestamp():
    return datetime.now().strftime("%H:%M:%S")


class ProcessManager:
    def __init__(self):
        self.processes = {}
        self.running = True
        
    def start_process(self, name, command, cwd, color):
        #Start a subprocess and stream its output
        print(colored(f"[{timestamp()}] Starting {name}...", color))
        
        try:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            self.processes[name] = {
                'process': process,
                'color': color
            }
            
            # Start output reader thread
            thread = threading.Thread(
                target=self._read_output,
                args=(name, process, color),
                daemon=True
            )
            thread.start()
            
            print(colored(f"[{timestamp()}] {name} started (PID: {process.pid})", color))
            return process
            
        except Exception as e:
            print(colored(f"[{timestamp()}] Failed to start {name}: {e}", Colors.RED))
            return None
    
    def _read_output(self, name, process, color):
        """Read and display process output"""
        prefix = colored(f"[{name}]", color)
        try:
            for line in process.stdout:
                if self.running:
                    print(f"{prefix} {line.rstrip()}")
        except:
            pass
    
    def stop_all(self):
        """Stop all processes gracefully"""
        self.running = False
        print(colored(f"\n[{timestamp()}] Stopping all processes...", Colors.YELLOW))
        
        for name, info in self.processes.items():
            process = info['process']
            color = info['color']
            
            if process.poll() is None:  # Still running
                print(colored(f"[{timestamp()}] Stopping {name}...", color))
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    print(colored(f"[{timestamp()}] Force killing {name}...", Colors.RED))
                    process.kill()
                print(colored(f"[{timestamp()}] {name} stopped", color))
    
    def wait_for_all(self):
        """Wait for all processes to complete"""
        for name, info in self.processes.items():
            info['process'].wait()


def check_docker_services():

    try:
        result = subprocess.run(
            ['docker', 'ps', '--format', '{{.Names}}'],
            capture_output=True,
            text=True
        )
        running_containers = result.stdout.strip().split('\n')
        
        redpanda_running = any('redpanda' in c.lower() for c in running_containers)
        questdb_running = any('questdb' in c.lower() for c in running_containers)
        
        return redpanda_running, questdb_running
    except:
        return False, False


def start_docker_services(project_dir):
    print(colored(f"[{timestamp()}] Starting Docker services...", Colors.BLUE))
    
    try:
        subprocess.run(
            ['docker-compose', 'up', '-d'],
            cwd=project_dir,
            check=True
        )
        print(colored(f"[{timestamp()}] Docker services started", Colors.GREEN))
        
        # Wait for services to be ready
        print(colored(f"[{timestamp()}] Waiting for services to be ready...", Colors.BLUE))
        time.sleep(5)
        return True
    except subprocess.CalledProcessError as e:
        print(colored(f"[{timestamp()}] Failed to start Docker services: {e}", Colors.RED))
        return False


def rebuild_cpp_extension(engine_dir):
    print(colored(f"[{timestamp()}] Rebuilding C++ extension...", Colors.MAGENTA))
    
    try:
        subprocess.run(
            [sys.executable, 'setup.py', 'build_ext', '--inplace'],
            cwd=engine_dir,
            check=True,
            capture_output=True
        )
        print(colored(f"[{timestamp()}] C++ extension rebuilt successfully", Colors.GREEN))
        return True
    except subprocess.CalledProcessError as e:
        print(colored(f"[{timestamp()}] Failed to rebuild C++ extension: {e}", Colors.RED))
        print(colored(f"Output: {e.output.decode() if e.output else 'No output'}", Colors.RED))
        return False


def main():
    parser = argparse.ArgumentParser(description="Run Cross-Border Arbitrage System")
    parser.add_argument('--no-viz', action='store_true', help='Skip visualization')
    parser.add_argument('--viz-only', action='store_true', help='Only run visualization')
    parser.add_argument('--docker', action='store_true', help='Start Docker services')
    parser.add_argument('--rebuild', action='store_true', help='Rebuild C++ extension')
    parser.add_argument('--test', action='store_true', help='Run tests instead of live system')
    
    args = parser.parse_args()
    
    # Get project directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = script_dir  # Assuming script is in project root
    engine_dir = os.path.join(project_dir, 'engine')
    ingestion_dir = os.path.join(project_dir, 'ingestion')
    
    print_banner()
    print(colored(f"Project Directory: {project_dir}", Colors.WHITE))
    print(colored(f"Python: {sys.executable}", Colors.WHITE))
    print()
    
    # Check Docker services
    redpanda_running, questdb_running = check_docker_services()
    
    if args.docker or (not redpanda_running or not questdb_running):
        if not redpanda_running:
            print(colored(f"[{timestamp()}] ⚠ Redpanda not running", Colors.YELLOW))
        if not questdb_running:
            print(colored(f"[{timestamp()}] ⚠ QuestDB not running", Colors.YELLOW))
        
        if args.docker:
            start_docker_services(project_dir)
        else:
            print(colored(f"[{timestamp()}] Use --docker to start Docker services", Colors.YELLOW))
            if not args.viz_only and not args.test:
                print(colored(f"[{timestamp()}] Continuing anyway (may fail without Docker)...", Colors.YELLOW))
    else:
        print(colored(f"[{timestamp()}] ✓ Redpanda running", Colors.GREEN))
        print(colored(f"[{timestamp()}] ✓ QuestDB running", Colors.GREEN))
    
    # Rebuild C++ extension if requested
    if args.rebuild:
        if not rebuild_cpp_extension(engine_dir):
            print(colored("Aborting due to build failure", Colors.RED))
            return 1
    
    # Run tests mode
    if args.test:
        print(colored(f"\n[{timestamp()}] Running tests...", Colors.CYAN))
        result = subprocess.run(
            [sys.executable, 'test_workflow.py'],
            cwd=engine_dir
        )
        return result.returncode
    
    # Visualization only mode
    if args.viz_only:
        print(colored(f"\n[{timestamp()}] Starting visualization only...", Colors.CYAN))
        subprocess.run(
            [sys.executable, 'visualize_graph.py', '--demo'],
            cwd=engine_dir
        )
        return 0
    
    # Start all components
    manager = ProcessManager()
    
    # Signal handler for graceful shutdown
    def signal_handler(sig, frame):
        print(colored("\n\nReceived shutdown signal...", Colors.YELLOW))
        manager.stop_all()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print(colored(f"\n[{timestamp()}] Starting system components...", Colors.CYAN))
    print(colored("-" * 60, Colors.WHITE))
    
    # Start Ingestion (Mock Data Generator)
    ingestion_proc = manager.start_process(
        name="INGESTION",
        command=[sys.executable, 'main.py'],
        cwd=ingestion_dir,
        color=Colors.GREEN
    )
    
    # Give ingestion a moment to start
    time.sleep(1)
    
    # Start Arbitrage Engine
    engine_proc = manager.start_process(
        name="ENGINE",
        command=[sys.executable, 'main.py'],
        cwd=engine_dir,
        color=Colors.MAGENTA
    )
    
    print(colored("-" * 60, Colors.WHITE))
    print(colored(f"[{timestamp()}] ✓ System running. Press Ctrl+C to stop.", Colors.CYAN))
    print()
    
    # Wait for processes
    try:
        while True:
            # Check if any process died
            for name, info in manager.processes.items():
                if info['process'].poll() is not None:
                    print(colored(f"[{timestamp()}] ⚠ {name} exited unexpectedly", Colors.RED))
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    
    manager.stop_all()
    print(colored(f"\n[{timestamp()}] System shutdown complete.", Colors.CYAN))
    return 0


if __name__ == "__main__":
    sys.exit(main())
