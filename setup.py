import os
import sys


def create_file(path, content):
    try:
        with open(path, 'w') as f:
            f.write(content.strip())
        print(f"[OK] Created file: {path}")
    except Exception as e:
        print(f" Failed to create {path}: {e}")


def create_dir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
            print(f"[OK] Directory: {path}")
        except Exception as e:
            print(f" Failed to create dir {path}: {e}")


# --- 1. Docker Compose (Fixed Volumes & Command Syntax) ---
DOCKER_COMPOSE = """
version: '3.8'

services:
  redpanda:
    image: docker.redpanda.com/redpandadata/redpanda:latest
    container_name: arb-redpanda
    command: >
      redpanda start
      --smp 1
      --memory 1G
      --reserve-memory 0M
      --overprovisioned
      --node-id 0
      --check=false
      --kafka-addr internal://0.0.0.0:9092,external://0.0.0.0:19092
      --advertise-kafka-addr internal://arb-redpanda:9092,external://localhost:19092
    ports:
      - "19092:19092"
      - "9644:9644"
    volumes:
      -./data/redpanda:/var/lib/redpanda/data
    networks:
      - arb-net

  questdb:
    image: questdb/questdb:latest
    container_name: arb-questdb
    ports:
      - "9000:9000"
      - "9009:9009"
    volumes:
      -./data/questdb:/var/lib/questdb
    networks:
      - arb-net

  console:
    image: docker.redpanda.com/redpandadata/console:latest
    container_name: arb-console
    ports:
      - "8080:8080"
    environment:
      KAFKA_BROKERS: arb-redpanda:9092
    depends_on:
      - redpanda
    networks:
      - arb-net

networks:
  arb-net:
    driver: bridge
"""

# --- 2. CMakeLists (Added Include Dirs & Debug Flags) ---
CMAKE_LISTS = """
cmake_minimum_required(VERSION 3.15)
project(GraphEngine)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python and Pybind11
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Create the Python extension module
pybind11_add_module(graph_core src/graph.cpp)

# Critical: Include Python headers so compilation doesn't fail
target_include_directories(graph_core PRIVATE ${Python3_INCLUDE_DIRS})

# Enable detailed error messages for Pybind11 (useful for debugging)
target_compile_definitions(graph_core PRIVATE PYBIND11_DETAILED_ERROR_MESSAGES)

# Optimization flags for HFT (O3 for speed, march=native for CPU specific instructions)
if(MSVC)
    target_compile_options(graph_core PRIVATE /O2)
else()
    target_compile_options(graph_core PRIVATE -O3 -march=native)
endif()
"""

# --- 3. C++ Engine (Added <queue> for SPFA) ---
CPP_SOURCE = """
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <limits>
#include <queue> // Required for SPFA
#include <unordered_map>

namespace py = pybind11;

struct Edge {
    int to;
    double weight;
};

class ArbitrageGraph {
    int num_currencies;
    std::vector<std::vector<Edge>> adj;
    std::unordered_map<std::string, int> currency_to_id;
    std::vector<std::string> id_to_currency;

public:
    ArbitrageGraph(const std::vector<std::string>& currencies) {
        num_currencies = currencies.size();
        adj.resize(num_currencies);
        id_to_currency = currencies;
        for(int i=0; i<num_currencies; ++i) {
            currency_to_id[currencies[i]] = i;
        }
    }

    void update_rate(const std::string& from, const std::string& to, double rate) {
        if (rate <= 0) return;
        if (currency_to_id.find(from) == currency_to_id.end() |

| 
            currency_to_id.find(to) == currency_to_id.end()) return;

        int u = currency_to_id[from];
        int v = currency_to_id[to];
        double weight = -std::log(rate);

        // Optimization: Update existing edge instead of pushing new one
        bool found = false;
        for (auto& edge : adj[u]) {
            if (edge.to == v) {
                edge.weight = weight;
                found = true;
                break;
            }
        }
        if (!found) {
            adj[u].push_back({v, weight});
        }
    }

    // Placeholder for SPFA - To be implemented in next phase
    std::vector<std::string> detect_arbitrage() {
        return {}; 
    }
};

PYBIND11_MODULE(graph_core, m) {
    py::class_<ArbitrageGraph>(m, "ArbitrageGraph")
      .def(py::init<const std::vector<std::string>&>())
      .def("update_rate", &ArbitrageGraph::update_rate)
      .def("detect_arbitrage", &ArbitrageGraph::detect_arbitrage, py::call_guard<py::gil_scoped_release>());
}
"""

# --- 4. Requirements (Switched to confluent-kafka) ---
PYTHON_REQUIREMENTS = """
confluent-kafka>=2.3.0
questdb>=1.0.0
requests>=2.31.0
ujson>=5.9.0
numpy>=1.24.0
pandas>=2.0.0
pybind11>=2.11.1
python-dotenv>=1.0.0
fastapi>=0.109.0
uvicorn>=0.27.0
"""

# --- 5. Environment Config ---
ENV_FILE = """
# Infrastructure
REDPANDA_BROKER=localhost:19092
QUESTDB_HTTP=http://localhost:9000
QUESTDB_INGEST=localhost:9009

# OANDA Configuration
OANDA_ENV=practice
OANDA_TOKEN=replace_with_your_token
OANDA_ACCOUNT_ID=replace_with_your_account_id

# App Settings
LOG_LEVEL=INFO
"""

# --- 6. Python Boilerplate ---
UTILS_LOGGER = """
import logging
import sys

def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
"""

INGESTION_MAIN = """
import os
from dotenv import load_dotenv
import requests
import ujson
from confluent_kafka import Producer
import sys

# Add parent dir to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_logger

load_dotenv()
logger = get_logger("Ingestion")

def main():
    logger.info("Starting OANDA Ingestion Service...")
    # TODO: Implement OANDA Streaming logic here
    pass

if __name__ == "__main__":
    main()
"""


# --- Execution ---

def main():
    root = "."

    # 1. Docker & Configs
    create_file(f"{root}/docker-compose.yml", DOCKER_COMPOSE)
    create_file(f"{root}/requirements.txt", PYTHON_REQUIREMENTS)
    create_file(f"{root}/.env", ENV_FILE)

    # 2. C++ Core Engine
    create_dir(f"{root}/engine")
    create_dir(f"{root}/engine/src")
    create_file(f"{root}/engine/CMakeLists.txt", CMAKE_LISTS)
    create_file(f"{root}/engine/src/graph.cpp", CPP_SOURCE)

    # 3. Python Utils Module
    create_dir(f"{root}/utils")
    create_file(f"{root}/utils/__init__.py", "")
    create_file(f"{root}/utils/logger.py", UTILS_LOGGER)

    # 4. Services
    create_dir(f"{root}/ingestion")
    create_file(f"{root}/ingestion/main.py", INGESTION_MAIN)

    create_dir(f"{root}/strategy")
    create_file(f"{root}/strategy/main.py", "# Strategy Consumer")

    print("\n" + "=" * 40)
    print("   PROJECT GENERATED SUCCESSFULLY")
    print("=" * 40)
    print(f"Location: {os.path.abspath(root)}")
    print("Next Steps:")
    print("1. cd fx_arbitrage_bot")
    print("2. pip install -r requirements.txt")
    print("3. docker compose up -d")
    print("4. Open directory in CLion (load CMakeLists.txt)")


if __name__ == "__main__":
    main()