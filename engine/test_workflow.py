import sys
import os
import math

# Ensure we can import the module from current directory
sys.path.append(os.getcwd())

try:
    import graph_core
    print("Successfully imported graph_core")
except ImportError as e:
    print(f"Failed to import graph_core: {e}")
    sys.exit(1)

def test_arbitrage():
    currencies = ["USD", "EUR", "JPY"]
    graph = graph_core.ArbitrageGraph(currencies)
    
    # USD -> EUR -> JPY -> USD
    # 1 USD = 0.9 EUR
    # 1 EUR = 160 JPY
    # 1 JPY = 0.0075 USD
    
    # Cycle: 1 * 0.9 * 160 * 0.0075 = 1.08 (Profit!)
    
    print("Adding rates...")
    graph.update_rate("USD", "EUR", 0.9)
    graph.update_rate("EUR", "JPY", 160.0)
    graph.update_rate("JPY", "USD", 0.0075)
    
    print("Detecting arbitrage...")
    cycle = graph.detect_arbitrage()
    
    if cycle:
        print(f"Arbitrage detected! Cycle: {cycle}")
        # Verify cycle content - order matters for path but for set check it's fine
        # The cycle should contain all 3 currencies.
        expected_elements = {"USD", "EUR", "JPY"}
        if set(cycle) == expected_elements:
             print("Cycle verification PASSED")
        else:
             print(f"Cycle verification FAILED. Expected elements {expected_elements}, got {cycle}")
    else:
        print("No arbitrage detected (FAILED)")

if __name__ == "__main__":
    test_arbitrage()
