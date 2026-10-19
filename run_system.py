#!/usr/bin/env python3
"""
Cross-Border Arbitrage System Runner
======================================
Orchestrates all system components:
  - Docker services (Redpanda, QuestDB, Flink)
  - Data Ingestion (Mock FX Stream)
  - C++ Arbitrage Engine (SPFA)
  - Stream Processor (OHLCV, correlations, features)
  - Strategy Engine (stat-arb, ML, risk, execution)

Usage:
    python run_system.py                  # Full system
    python run_system.py --docker         # Start Docker first
    python run_system.py --init-db        # Initialize QuestDB tables
    python run_system.py --backtest       # Replay historical data
    python run_system.py --train          # Train ML models
    python run_system.py --rebuild        # Rebuild C++ extension
    python run_system.py --viz-only       # Visualization only
"""
import os
import sys
import time
import signal
import subprocess
import argparse
import threading
from datetime import datetime


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
    |---------------------------------------------------------------+
    |                                                               |
    |    CROSS-BORDER ARBITRAGE DETECTION SYSTEM  (Phase 2)         |
    |                                                               |
    |   C++ Engine | Redpanda | QuestDB | Flink | ML Pipeline       |
    |   Stat-Arb (OU) | Graph Embeddings | Monte Carlo Risk         |
    |                                                               |
    +---------------------------------------------------------------+
    """
    print(colored(banner, Colors.CYAN))


def timestamp():
    return datetime.now().strftime("%H:%M:%S")


class ProcessManager:
    def __init__(self):
        self.processes = {}
        self.running = True

    def start_process(self, name, command, cwd, color):
        """Start a subprocess and stream its output."""
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
        """Read and display process output."""
        prefix = colored(f"[{name}]", color)
        try:
            for line in process.stdout:
                if self.running:
                    print(f"{prefix} {line.rstrip()}")
        except:
            pass

    def stop_all(self):
        """Stop all processes gracefully."""
        self.running = False
        print(colored(f"\n[{timestamp()}] Stopping all processes...", Colors.YELLOW))

        for name, info in self.processes.items():
            process = info['process']
            color = info['color']

            if process.poll() is None:
                print(colored(f"[{timestamp()}] Stopping {name}...", color))
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    print(colored(f"[{timestamp()}] Force killing {name}...", Colors.RED))
                    process.kill()
                print(colored(f"[{timestamp()}] {name} stopped", color))

    def wait_for_all(self):
        """Wait for all processes to complete."""
        for name, info in self.processes.items():
            info['process'].wait()


def check_docker_services():
    try:
        result = subprocess.run(
            ['docker', 'ps', '--format', '{{.Names}}'],
            capture_output=True,
            text=True
        )
        running = result.stdout.strip().split('\n')

        return {
            'redpanda': any('redpanda' in c.lower() for c in running),
            'questdb': any('questdb' in c.lower() for c in running),
            'flink-jm': any('flink-jobmanager' in c.lower() for c in running),
            'flink-tm': any('flink-taskmanager' in c.lower() for c in running),
        }
    except:
        return {'redpanda': False, 'questdb': False, 'flink-jm': False, 'flink-tm': False}


def start_docker_services(project_dir):
    print(colored(f"[{timestamp()}] Starting Docker services...", Colors.BLUE))

    try:
        subprocess.run(
            ['docker', 'compose', 'up', '-d'],
            cwd=project_dir,
            check=True
        )
        print(colored(f"[{timestamp()}] Docker services started", Colors.GREEN))
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


def init_questdb(project_dir):
    """Initialize QuestDB tables."""
    print(colored(f"[{timestamp()}] Initializing QuestDB tables...", Colors.BLUE))
    result = subprocess.run(
        [sys.executable, 'scripts/init_questdb.py'],
        cwd=project_dir,
    )
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Run Cross-Border Arbitrage System (Phase 2)")
    parser.add_argument('--no-viz', action='store_true', help='Skip visualization')
    parser.add_argument('--viz-only', action='store_true', help='Only run visualization')
    parser.add_argument('--docker', action='store_true', help='Start Docker services first')
    parser.add_argument('--rebuild', action='store_true', help='Rebuild C++ extension')
    parser.add_argument('--init-db', action='store_true', help='Initialize QuestDB tables')
    parser.add_argument('--test', action='store_true', help='Run tests')
    parser.add_argument('--backtest', action='store_true', help='Run backtest replay')
    parser.add_argument('--backtest-speed', type=float, default=100, help='Backtest speed multiplier')
    parser.add_argument('--train', action='store_true', help='Train ML models')
    parser.add_argument('--no-engine', action='store_true', help='Skip C++ arbitrage engine')
    parser.add_argument('--no-strategy', action='store_true', help='Skip strategy engine')

    args = parser.parse_args()

    # Get project directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = script_dir
    engine_dir = os.path.join(project_dir, 'engine')
    ingestion_dir = os.path.join(project_dir, 'ingestion')
    streaming_dir = os.path.join(project_dir, 'streaming')
    strategy_dir = os.path.join(project_dir, 'strategy')

    print_banner()
    print(colored(f"Project: {project_dir}", Colors.WHITE))
    print(colored(f"Python: {sys.executable}", Colors.WHITE))
    print()

    # Check Docker services
    services = check_docker_services()

    if args.docker or not all([services['redpanda'], services['questdb']]):
        for name, running in services.items():
            status = "✓" if running else "⚠ NOT RUNNING"
            color = Colors.GREEN if running else Colors.YELLOW
            print(colored(f"[{timestamp()}] {name}: {status}", color))

        if args.docker:
            start_docker_services(project_dir)
        elif not services['redpanda'] or not services['questdb']:
            print(colored(f"[{timestamp()}] Use --docker to start Docker services", Colors.YELLOW))
    else:
        for name, running in services.items():
            print(colored(f"[{timestamp()}] ✓ {name}", Colors.GREEN))

    # Init QuestDB
    if args.init_db:
        init_questdb(project_dir)
        return 0

    # Rebuild C++ extension
    if args.rebuild:
        if not rebuild_cpp_extension(engine_dir):
            print(colored("Aborting due to build failure", Colors.RED))
            return 1

    # Test mode
    if args.test:
        print(colored(f"\n[{timestamp()}] Running tests...", Colors.CYAN))
        result = subprocess.run(
            [sys.executable, 'test_workflow.py'],
            cwd=engine_dir
        )
        return result.returncode

    # Visualization only
    if args.viz_only:
        subprocess.run([sys.executable, 'visualize_graph.py', '--demo'], cwd=engine_dir)
        return 0

    # Backtest mode
    if args.backtest:
        print(colored(f"\n[{timestamp()}] Starting backtest replay...", Colors.CYAN))
        subprocess.run(
            [sys.executable, '-m', 'backtest.replay',
             '--synthetic', '--speed', str(args.backtest_speed)],
            cwd=project_dir
        )
        return 0

    # ─── FULL SYSTEM ───
    manager = ProcessManager()

    def signal_handler(sig, frame):
        print(colored("\n\nReceived shutdown signal...", Colors.YELLOW))
        manager.stop_all()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print(colored(f"\n[{timestamp()}] Starting system components...", Colors.CYAN))
    print(colored("─" * 60, Colors.WHITE))

    # 1. Data Ingestion (Mock FX Stream)
    manager.start_process(
        name="INGESTION",
        command=[sys.executable, 'main.py'],
        cwd=ingestion_dir,
        color=Colors.GREEN,
    )
    time.sleep(1)

    # 2. Stream Processor (OHLCV, correlations, features)
    manager.start_process(
        name="STREAMING",
        command=[sys.executable, 'processor.py'],
        cwd=streaming_dir,
        color=Colors.BLUE,
    )
    time.sleep(1)

    # 3. C++ Arbitrage Engine (Phase 1 SPFA)
    if not args.no_engine:
        manager.start_process(
            name="ENGINE",
            command=[sys.executable, 'main.py'],
            cwd=engine_dir,
            color=Colors.MAGENTA,
        )
        time.sleep(0.5)

    # 4. Strategy Engine (stat-arb, ML, risk)
    if not args.no_strategy:
        manager.start_process(
            name="STRATEGY",
            command=[sys.executable, 'main.py'],
            cwd=strategy_dir,
            color=Colors.CYAN,
        )

    print(colored("─" * 60, Colors.WHITE))
    print(colored(f"[{timestamp()}] ✓ System running. Press Ctrl+C to stop.", Colors.CYAN))
    print(colored(f"[{timestamp()}] Redpanda Console: http://localhost:8080", Colors.WHITE))
    print(colored(f"[{timestamp()}] QuestDB Console:  http://localhost:9000", Colors.WHITE))
    print(colored(f"[{timestamp()}] Flink Dashboard:  http://localhost:8081", Colors.WHITE))
    print()

    # Monitor processes
    try:
        while True:
            for name, info in manager.processes.items():
                if info['process'].poll() is not None:
                    print(colored(f"[{timestamp()}] ⚠ {name} exited unexpectedly", Colors.RED))
            time.sleep(2)
    except KeyboardInterrupt:
        pass

    manager.stop_all()
    print(colored(f"\n[{timestamp()}] System shutdown complete.", Colors.CYAN))
    return 0


if __name__ == "__main__":
    sys.exit(main())
