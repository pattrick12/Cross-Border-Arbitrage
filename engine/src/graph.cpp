#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace py = pybind11;

struct Edge {
  int to;
  double weight;
};

// Result struct for arbitrage detection
struct ArbitrageResult {
  std::vector<std::string> cycle;
  double profit_pct; // Profit percentage (e.g., 0.5 for 0.5%)
  int order;         // Number of currencies in the cycle (excluding repeat)
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

  // Get the weight (rate) between two currencies
  double get_rate(const std::string &from, const std::string &to) {
    if (currency_to_id.find(from) == currency_to_id.end() ||
        currency_to_id.find(to) == currency_to_id.end())
      return 0.0;

    int u = currency_to_id[from];
    int v = currency_to_id[to];

    for (const auto &edge : adj[u]) {
      if (edge.to == v) {
        return std::exp(-edge.weight); // Convert weight back to rate
      }
    }
    return 0.0;
  }

  // Get all current rates as a list of tuples (from, to, rate)
  std::vector<std::tuple<std::string, std::string, double>> get_all_rates() {
    std::vector<std::tuple<std::string, std::string, double>> rates;
    for (int u = 0; u < num_currencies; ++u) {
      for (const auto &edge : adj[u]) {
        double rate = std::exp(-edge.weight);
        rates.push_back({id_to_currency[u], id_to_currency[edge.to], rate});
      }
    }
    return rates;
  }

  // Calculate profit for a given cycle
  double calculate_cycle_profit(const std::vector<std::string> &cycle) {
    if (cycle.size() < 2)
      return 0.0;

    double weight_sum = 0.0;
    for (size_t i = 0; i < cycle.size() - 1; ++i) {
      auto it_from = currency_to_id.find(cycle[i]);
      auto it_to = currency_to_id.find(cycle[i + 1]);
      if (it_from == currency_to_id.end() || it_to == currency_to_id.end())
        return 0.0;

      int u = it_from->second;
      int v = it_to->second;

      bool found = false;
      for (const auto &edge : adj[u]) {
        if (edge.to == v) {
          weight_sum += edge.weight;
          found = true;
          break;
        }
      }
      if (!found)
        return 0.0;
    }

    // profit = exp(-sum) - 1, converted to percentage
    double product = std::exp(-weight_sum);
    return (product - 1.0) * 100.0;
  }

  // Original detect_arbitrage - returns just the cycle (backward compatible)
  std::vector<std::string> detect_arbitrage() {
    auto result = detect_arbitrage_with_profit();
    return result.cycle;
  }

  // New detect_arbitrage that returns cycle with profit
  ArbitrageResult detect_arbitrage_with_profit() {
    std::vector<double> dist(num_currencies, 0.0);
    std::vector<int> parent(num_currencies, -1);
    std::vector<int> count(num_currencies, 0);
    std::vector<bool> in_queue(num_currencies, true);
    std::queue<int> q;

    // Initialize queue with all nodes to find cycles anywhere
    for (int i = 0; i < num_currencies; ++i) {
      q.push(i);
    }

    int cycle_node = -1;
    while (!q.empty()) {
      int u = q.front();
      q.pop();
      in_queue[u] = false;

      for (const auto &edge : adj[u]) {
        int v = edge.to;
        double weight = edge.weight;

        if (dist[v] > dist[u] + weight + 1e-9) {
          dist[v] = dist[u] + weight;
          parent[v] = u;

          if (!in_queue[v]) {
            q.push(v);
            in_queue[v] = true;
            count[v]++;

            if (count[v] >= num_currencies) {
              cycle_node = v;
              break;
            }
          }
        }
      }
      if (cycle_node != -1)
        break;
    }

    ArbitrageResult result;
    result.order = 0;
    result.profit_pct = 0.0;

    if (cycle_node == -1) {
      return result;
    }

    result.cycle = reconstruct_path(cycle_node, parent);
    result.order = result.cycle.size() - 1; // Exclude repeated currency
    result.profit_pct = calculate_cycle_profit(result.cycle);

    return result;
  }

  std::vector<ArbitrageResult> detect_arbitrage_n(int max_order = 5,
                                                  double min_profit_pct = 0.0) {
    std::vector<ArbitrageResult> results;

    // DFS from each node to find all cycles
    for (int start = 0; start < num_currencies; ++start) {
      std::vector<int> path;
      std::vector<bool> visited(num_currencies, false);
      std::vector<double> weight_sum;

      dfs_find_cycles(start, start, path, visited, weight_sum, 0.0, max_order,
                      min_profit_pct, results);
    }

    // Remove duplicate cycles and sort by profit
    std::vector<ArbitrageResult> unique_results;
    std::unordered_set<std::string> seen;

    for (const auto &r : results) {
      // Create canonical representation (smallest currency first)
      std::vector<std::string> sorted_cycle = r.cycle;
      sorted_cycle.pop_back(); // Remove last (duplicate of first)
      auto min_it = std::min_element(sorted_cycle.begin(), sorted_cycle.end());
      std::rotate(sorted_cycle.begin(), min_it, sorted_cycle.end());

      std::string key;
      for (const auto &c : sorted_cycle)
        key += c + "-";

      if (seen.find(key) == seen.end()) {
        seen.insert(key);
        unique_results.push_back(r);
      }
    }

    // Sort by profit descending
    std::sort(unique_results.begin(), unique_results.end(),
              [](const ArbitrageResult &a, const ArbitrageResult &b) {
                return a.profit_pct > b.profit_pct;
              });

    return unique_results;
  }

private:
  void dfs_find_cycles(int start, int current, std::vector<int> &path,
                       std::vector<bool> &visited, std::vector<double> &weights,
                       double current_weight, int max_order,
                       double min_profit_pct,
                       std::vector<ArbitrageResult> &results) {
    path.push_back(current);
    visited[current] = true;

    for (const auto &edge : adj[current]) {
      int next = edge.to;
      double new_weight = current_weight + edge.weight;

      // Found a cycle back to start
      if (next == start && path.size() >= 2) {
        double profit = (std::exp(-new_weight) - 1.0) * 100.0;

        if (profit > min_profit_pct) {
          ArbitrageResult result;
          for (int node : path) {
            result.cycle.push_back(id_to_currency[node]);
          }
          result.cycle.push_back(id_to_currency[start]); // Complete the cycle
          result.order = path.size();
          result.profit_pct = profit;
          results.push_back(result);
        }
      }
      // Continue exploring if not visited and within order limit
      else if (!visited[next] && static_cast<int>(path.size()) < max_order) {
        dfs_find_cycles(start, next, path, visited, weights, new_weight,
                        max_order, min_profit_pct, results);
      }
    }

    path.pop_back();
    visited[current] = false;
  }

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
  m.doc() = "C++ Arbitrage Graph Engine with SPFA negative cycle detection";

  py::class_<ArbitrageResult>(m, "ArbitrageResult")
      .def(py::init<>())
      .def_readwrite("cycle", &ArbitrageResult::cycle)
      .def_readwrite("profit_pct", &ArbitrageResult::profit_pct)
      .def_readwrite("order", &ArbitrageResult::order)
      .def("__repr__", [](const ArbitrageResult &r) {
        std::string cycle_str;
        for (size_t i = 0; i < r.cycle.size(); ++i) {
          cycle_str += r.cycle[i];
          if (i < r.cycle.size() - 1)
            cycle_str += " -> ";
        }
        return "<ArbitrageResult order=" + std::to_string(r.order) +
               " profit=" + std::to_string(r.profit_pct) + "% path=[" +
               cycle_str + "]>";
      });

  py::class_<ArbitrageGraph>(m, "ArbitrageGraph")
      .def(py::init<const std::vector<std::string> &>())
      .def("update_rate", &ArbitrageGraph::update_rate)
      .def("get_rate", &ArbitrageGraph::get_rate)
      .def("get_all_rates", &ArbitrageGraph::get_all_rates)
      .def("calculate_cycle_profit", &ArbitrageGraph::calculate_cycle_profit)
      .def("detect_arbitrage", &ArbitrageGraph::detect_arbitrage,
           py::call_guard<py::gil_scoped_release>())
      .def("detect_arbitrage_with_profit",
           &ArbitrageGraph::detect_arbitrage_with_profit,
           py::call_guard<py::gil_scoped_release>())
      .def("detect_arbitrage_n", &ArbitrageGraph::detect_arbitrage_n,
           py::arg("max_order") = 5, py::arg("min_profit_pct") = 0.0,
           py::call_guard<py::gil_scoped_release>(),
           "Find all arbitrage cycles up to n-order with minimum profit "
           "threshold");
}