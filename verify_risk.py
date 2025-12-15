
import networkx as nx
import pandas as pd

def get_social_risk_scores(G, node_labels, iterations=10):
    # Start with known labels (0 or 1), others are 0.5 (unknown)
    # Note: Using G.nodes[node].get('label', 0.5) relies on labels being set on graph attributes
    # The snippet we are testing takes G, but in the loop uses 'node_labels' dictionary for clamping.
    # So we must pass node_labels to this function or scope it correctly.
    # In the original code, 'node_labels' is a global/outer scope variable.
    # Here we accept it as an argument for testing.
    
    scores = {}
    for node in G.nodes():
        if node in node_labels and not pd.isna(node_labels[node]):
             scores[node] = node_labels[node]
        else:
             scores[node] = 0.5

    print(f"Initial scores: {scores}")

    for i in range(iterations):
        new_scores = {}
        for node in G.nodes():
            neighbors = list(G.neighbors(node))
            if not neighbors:
                new_scores[node] = scores[node]
                continue
            
            # Every node becomes the average of its neighbors
            neighbor_avg = sum(scores[n] for n in neighbors) / len(neighbors)
            
            # "Clamping": Keep the original 1s as 1 and 0s as 0
            if node in node_labels and not pd.isna(node_labels[node]):
                new_scores[node] = node_labels[node]
            else:
                new_scores[node] = neighbor_avg
        scores = new_scores
        print(f"Iteration {i+1}: {scores}")
    return scores

def run_test():
    # Setup simple graph: A(1) -- B(?) -- C(?) -- D(0)
    G = nx.Graph()
    G.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'D')])
    
    # Known labels
    node_labels = {'A': 1.0, 'D': 0.0}
    # B and C are unknown
    
    # Run scoring
    final_scores = get_social_risk_scores(G, node_labels, iterations=5)
    
    # Verification
    print("\n--- Verification Results ---")
    
    # 1. Clamping Check
    assert final_scores['A'] == 1.0, f"Error: A should be 1.0, got {final_scores['A']}"
    assert final_scores['D'] == 0.0, f"Error: D should be 0.0, got {final_scores['D']}"
    print("SUCCESS: Known labels clamped correctly.")
    
    # 2. Propagation Check
    # B should be influenced by A(1), so B > 0.5 initially and stays relative high
    # C should be influenced by D(0), so C < 0.5 initially and stays relative low
    print(f"Final B: {final_scores['B']}")
    print(f"Final C: {final_scores['C']}")
    
    # In a perfect line A(1)-B-C-D(0), B should be 2/3 and C should be 1/3 at convergence? 
    # Or at least B > C.
    assert final_scores['B'] > final_scores['C'], "Error: B should have higher risk than C due to being closer to A(1)"
    print("SUCCESS: Risk propagated correctly.")

if __name__ == "__main__":
    run_test()
