import networkx as nx
import matplotlib.pyplot as plt
import logging

class GraphVisualizer:
    def __init__(self, graph: nx.DiGraph):
        self.graph = graph
        self.logger = logging.getLogger(__name__)

    def generate_visualization(self, filename: str = "agent_task_graph.png"):
        """
        Generates and saves a visualization of the agent-task graph.

        Args:
            filename (str): The name of the file to save the visualization.
        """
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(self.graph)
        node_colors = ['lightblue' if data.get('type') == 'agent' else 'lightgreen' for _, data in self.graph.nodes(data=True)]
        
        nx.draw_networkx_nodes(self.graph, pos, node_color=node_colors, node_size=1500)
        nx.draw_networkx_edges(self.graph, pos, arrowstyle='->', arrowsize=20)
        nx.draw_networkx_labels(self.graph, pos, {node: node for node in self.graph.nodes()}, font_size=10, font_weight='bold')
        
        plt.title("Agent and Task Graph Visualization")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        self.logger.info(f"Graph visualization saved as {filename}")
