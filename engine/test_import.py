import sys
import os

# Add the build directory to path
# We need to find where the .so file is. It was copied to the current directory by --inplace.
sys.path.append(os.getcwd())

try:
    import graph_core
    print("Successfully imported graph_core")
    print("ArbitrageGraph is:", graph_core.ArbitrageGraph)
    
    # Try to instantiate
    g = graph_core.ArbitrageGraph(["USD", "EUR"])
    print("Successfully instantiated ArbitrageGraph")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Error: {e}")
