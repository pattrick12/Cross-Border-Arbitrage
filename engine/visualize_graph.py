#!/usr/bin/env python3


import sys
import os
import time
import argparse

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import graph_core
except ImportError:
    print("Error: graph_core module not found. Build the C++ extension first.")
    sys.exit(1)

try:
    import networkx as nx
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError:
    print("Error: Required packages not found. Install with:")
    print("  pip install networkx matplotlib")
    sys.exit(1)


class GraphVisualizer:
    def __init__(self, currencies):
        self.currencies = currencies
        self.graph = graph_core.ArbitrageGraph(currencies)
        self.fig = None
        self.ax = None
        
    def update_rate(self, from_curr, to_curr, rate):
        """Update exchange rate in the graph"""
        self.graph.update_rate(from_curr, to_curr, rate)
        
    def create_networkx_graph(self):
        """Convert internal graph to NetworkX for visualization"""
        G = nx.DiGraph()
        
        # Add nodes
        for currency in self.currencies:
            G.add_node(currency)
        
        # Add edges with rates
        rates = self.graph.get_all_rates()
        for from_curr, to_curr, rate in rates:
            G.add_edge(from_curr, to_curr, weight=rate, label=f"{rate:.4f}")
        
        return G
    
    def visualize(self, title="Currency Exchange Graph", highlight_cycles=True, save_path=None):
        """
        Visualize the currency graph
        
        Args:
            title: Graph title
            highlight_cycles: If True, detect and highlight arbitrage cycles
            save_path: If provided, save the figure to this path
        """
        G = self.create_networkx_graph()
        
        # Create figure
        self.fig, self.ax = plt.subplots(1, 1, figsize=(14, 10))
        self.ax.set_title(title, fontsize=16, fontweight='bold')
        
        # Layout - circular for currency graphs
        pos = nx.circular_layout(G)
        
        # Detect arbitrage cycles
        arbitrage_edges = set()
        arbitrage_nodes = set()
        arbitrage_info = []
        
        if highlight_cycles:
            results = self.graph.detect_arbitrage_n(max_order=6, min_profit_pct=-100)
            
            # Get top 3 profitable cycles
            for result in results[:3]:
                if result.profit_pct > 0:
                    cycle = result.cycle
                    arbitrage_info.append(result)
                    for i in range(len(cycle) - 1):
                        arbitrage_edges.add((cycle[i], cycle[i+1]))
                        arbitrage_nodes.add(cycle[i])
        
        # Draw regular edges
        regular_edges = [(u, v) for u, v in G.edges() if (u, v) not in arbitrage_edges]
        nx.draw_networkx_edges(
            G, pos,
            edgelist=regular_edges,
            edge_color='#888888',
            arrows=True,
            arrowsize=20,
            connectionstyle="arc3,rad=0.1",
            width=1.5,
            alpha=0.7,
            ax=self.ax
        )
        
        # Draw arbitrage edges (highlighted)
        if arbitrage_edges:
            nx.draw_networkx_edges(
                G, pos,
                edgelist=list(arbitrage_edges),
                edge_color='#FF4444',
                arrows=True,
                arrowsize=25,
                connectionstyle="arc3,rad=0.1",
                width=3,
                alpha=0.9,
                ax=self.ax
            )
        
        # Draw regular nodes
        regular_nodes = [n for n in G.nodes() if n not in arbitrage_nodes]
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=regular_nodes,
            node_color='#4A90D9',
            node_size=2000,
            alpha=0.9,
            ax=self.ax
        )
        
        # Draw arbitrage nodes (highlighted)
        if arbitrage_nodes:
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=list(arbitrage_nodes),
                node_color='#FF6B6B',
                node_size=2200,
                alpha=0.9,
                ax=self.ax
            )
        
        # Draw labels
        nx.draw_networkx_labels(
            G, pos,
            font_size=12,
            font_weight='bold',
            font_color='white',
            ax=self.ax
        )
        
        # Draw edge labels (rates)
        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels=edge_labels,
            font_size=8,
            font_color='#333333',
            ax=self.ax
        )
        
        # Add legend
        legend_elements = [
            mpatches.Patch(color='#4A90D9', label='Currency Node'),
            mpatches.Patch(color='#888888', label='Exchange Rate'),
        ]
        if arbitrage_edges:
            legend_elements.extend([
                mpatches.Patch(color='#FF6B6B', label='Arbitrage Node'),
                mpatches.Patch(color='#FF4444', label='Arbitrage Path'),
            ])
        self.ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
        
        # Add arbitrage info text
        if arbitrage_info:
            info_text = "Detected Arbitrage Opportunities:\n"
            for i, result in enumerate(arbitrage_info[:3]):
                info_text += f"{i+1}. {' → '.join(result.cycle)}\n"
                info_text += f"   Profit: {result.profit_pct:.4f}% ({result.order}-way)\n"
            
            self.ax.text(
                0.02, 0.02, info_text,
                transform=self.ax.transAxes,
                fontsize=9,
                verticalalignment='bottom',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            )
        
        # Style
        self.ax.set_facecolor('#F5F5F5')
        self.ax.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Graph saved to: {save_path}")
        
        return self.fig
    
    def show(self):
        """Display the graph"""
        plt.show()


def demo_visualization():
    """Demo with sample data"""
    print("="*60)
    print("  Currency Exchange Graph Visualization Demo")
    print("="*60)
    
    # Major + emerging currencies
    currencies = ["USD", "EUR", "GBP", "JPY", "CHF", "INR", "CNY"]
    
    viz = GraphVisualizer(currencies)
    
    # Add exchange rates (some with arbitrage opportunity)
    rates = [
        # Major pairs
        ("USD", "EUR", 0.92),
        ("EUR", "USD", 1.087),
        ("USD", "GBP", 0.79),
        ("GBP", "USD", 1.27),
        ("USD", "JPY", 149.5),
        ("JPY", "USD", 0.00671),
        ("EUR", "GBP", 0.858),
        ("GBP", "EUR", 1.17),
        ("EUR", "JPY", 162.5),
        ("JPY", "EUR", 0.00618),
        ("USD", "CHF", 0.875),
        ("CHF", "USD", 1.145),
        
        # Create an arbitrage opportunity: USD -> EUR -> JPY -> USD
        # 0.92 * 163 * 0.00673 = 1.0096 (0.96% profit)
        ("EUR", "JPY", 163.0),
        ("JPY", "USD", 0.00673),
        
        # Emerging market rates
        ("USD", "INR", 83.5),
        ("INR", "USD", 0.012),
        ("USD", "CNY", 7.25),
        ("CNY", "USD", 0.138),
        ("EUR", "INR", 90.5),
        ("INR", "EUR", 0.0111),
    ]
    
    print("\nAdding exchange rates...")
    for from_curr, to_curr, rate in rates:
        viz.update_rate(from_curr, to_curr, rate)
        print(f"  {from_curr} -> {to_curr}: {rate}")
    
    print("\nGenerating visualization...")
    viz.visualize(
        title="Cross-Border Arbitrage - Currency Exchange Graph",
        highlight_cycles=True,
        save_path="graph_visualization.png"
    )
    
    print("\nDisplaying graph (close window to exit)...")
    viz.show()


def live_monitoring(currencies, interval=2.0):
    """
    Live monitoring mode - refreshes graph periodically
    
    Args:
        currencies: List of currencies to monitor
        interval: Refresh interval in seconds
    """
    print("="*60)
    print("  Live Graph Monitoring Mode")
    print(f"  Refresh interval: {interval}s")
    print("  Press Ctrl+C to stop")
    print("="*60)
    
    viz = GraphVisualizer(currencies)
    plt.ion()  # Interactive mode
    
    try:
        iteration = 0
        while True:
            iteration += 1
            
            # In live mode, you would fetch rates from the graph_core
            # For demo, we'll just show the current state
            viz.visualize(
                title=f"Live Currency Graph (Update #{iteration})",
                highlight_cycles=True
            )
            
            plt.pause(interval)
            plt.clf()
            
    except KeyboardInterrupt:
        print("\nStopping live monitoring...")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Currency Graph Visualization Tool")
    parser.add_argument('--demo', action='store_true', help='Run demo visualization')
    parser.add_argument('--live', action='store_true', help='Live monitoring mode')
    parser.add_argument('--save', type=str, help='Save graph to file')
    parser.add_argument('--interval', type=float, default=2.0, help='Refresh interval for live mode')
    
    args = parser.parse_args()
    
    if args.demo or (not args.live):
        demo_visualization()
    elif args.live:
        currencies = ["USD", "EUR", "GBP", "JPY", "CHF", "AUD", "CAD", "INR", "CNY"]
        live_monitoring(currencies, args.interval)


if __name__ == "__main__":
    main()
