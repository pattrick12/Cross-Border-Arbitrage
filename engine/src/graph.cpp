#include <cmath>
#include <iostream>
#include <limits>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <queue> // Required for SPFA
#include <string>
#include <unordered_map>
#include <vector>

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
  ArbitrageGraph(const std::vector<std::string> &currencies) {
    num_currencies = currencies.size();
    adj.resize(num_currencies);
    id_to_currency = currencies;
    for (int i = 0; i < num_currencies; ++i) {
      currency_to_id[currencies[i]] = i;
    }
  }

  void update_rate(const std::string &from, const std::string &to,
                   double rate) {
    if (rate <= 0)
      return;
    if (currency_to_id.find(from) == currency_to_id.end() ||
        currency_to_id.find(to) == currency_to_id.end())
      return;

    int u = currency_to_id[from];
    int v = currency_to_id[to];
    double weight = -std::log(rate);

    // Optimization: Update existing edge instead of pushing new one
    bool found = false;
    for (auto &edge : adj[u]) {
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

  // SPFA Algorithm to detect negative cycles
  std::vector<std::string> detect_arbitrage() {
    std::vector<double> dist(num_currencies, 0.0);
    std::vector<int> parent(num_currencies, -1);
    std::vector<int> count(num_currencies, 0);
    std::vector<bool> in_queue(num_currencies, true);
    std::queue<int> q;

    // Initialize queue with all nodes to find cycles anywhere
    for (int i = 0; i < num_currencies; ++i) {
      q.push(i);
    }

    while (!q.empty()) {
      int u = q.front();
      q.pop();
      in_queue[u] = false;

      for (const auto &edge : adj[u]) {
        int v = edge.to;
        double weight = edge.weight;

        if (dist[v] > dist[u] + weight + 1e-9) { // 1e-9 for float stability
          dist[v] = dist[u] + weight;
          parent[v] = u;

          if (!in_queue[v]) {
            q.push(v);
            in_queue[v] = true;
            count[v]++;

            if (count[v] >= num_currencies) {
              // Negative cycle detected
              return reconstruct_path(v, parent);
            }
          }
        }
      }
    }
    return {};
  }

private:
  std::vector<std::string> reconstruct_path(int start_node,
                                            const std::vector<int> &parent) {
    std::vector<std::string> path;
    std::vector<int> cycle_nodes;
    int curr = start_node;

    // Retrace back num_currencies times to ensure we are in the cycle
    for (int i = 0; i < num_currencies; ++i) {
      curr = parent[curr];
    }

    int cycle_start = curr;
    cycle_nodes.push_back(curr);
    curr = parent[curr];

    while (curr != cycle_start) {
      cycle_nodes.push_back(curr);
      curr = parent[curr];
    }
    cycle_nodes.push_back(cycle_start);

    // Reverse to get forward path
    for (auto it = cycle_nodes.rbegin(); it != cycle_nodes.rend(); ++it) {
      path.push_back(id_to_currency[*it]);
    }

    return path;
  }
};

PYBIND11_MODULE(graph_core, m) {
  py::class_<ArbitrageGraph>(m, "ArbitrageGraph")
      .def(py::init<const std::vector<std::string> &>())
      .def("update_rate", &ArbitrageGraph::update_rate)
      .def("detect_arbitrage", &ArbitrageGraph::detect_arbitrage,
           py::call_guard<py::gil_scoped_release>());
}