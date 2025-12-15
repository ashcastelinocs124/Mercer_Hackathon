
import pandas as pd
import networkx as nx
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import mean_squared_error

# 1. Load Data
train_df = pd.read_csv('train.csv')
edges = pd.read_csv('social_graph.csv')

# 2. Risk Score Calculation (Feature Engineering)
# Add all users to graph
G = nx.from_pandas_edgelist(edges, 'user_a', 'user_b')
G.add_nodes_from(train_df['user_hash'])

# 3. Split Data FIRST to avoid Data Leakage
# If we calculate risk scores using ALL labels, then the validation set's 'risk_factor' 
# will be exactly its label (due to clamping), leading to 0 MSE.
# We must calculate risk scores using ONLY the training split's labels.

Target = 'is_cheating'
id_column = 'user_hash'

# Filter to valid rows
train_df = train_df.dropna(subset=[Target])
train_df[Target] = train_df[Target].astype(int)

# Split into Train and Validation sets
train_subset, val_subset = train_test_split(
    train_df, 
    test_size=0.2, 
    random_state=42, 
    stratify=train_df[Target]
)

print(f"Train subset shape: {train_subset.shape}")
print(f"Val subset shape: {val_subset.shape}")

# 4. Calculate Risk Scores using ONLY Train Labels
# Prepare labels dictionary from TRAIN subset only
node_labels = train_subset.set_index('user_hash')['is_cheating'].to_dict()

print("Calculating risk scores using only training labels...")
iterations = 10
scores = {}

# Initialize
# Note: G still contains all nodes (from edges and full train_df), which is fine.
# But we only use labels from train_subset.
for node in G.nodes():
    # If node is in train_subset, use its label.
    # If node is in val_subset (or edges only), treat as unknown (0.5).
    if node in node_labels:
        val = node_labels[node]
    else:
        # Check if we have a label in the graph attribute from before? 
        # Ideally we shouldn't use G's attributes because they might have full data.
        # Let's rely strictly on node_labels dict which is derived from train_subset.
        val = 0.5
    scores[node] = val

for i in range(iterations):
    new_scores = {}
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        if not neighbors:
            new_scores[node] = scores[node]
            continue
        
        neighbor_avg = sum(scores[n] for n in neighbors) / len(neighbors)
        
        # Clamp ONLY if in training set
        if node in node_labels:
            new_scores[node] = node_labels[node]
        else:
            new_scores[node] = neighbor_avg
    scores = new_scores

print("Risk scores calculated.")

# 5. Map Scores back to Dataframes
# Now val_subset users will have risk_factors derived from neighbors, not self-clamped.
train_subset = train_subset.copy()
val_subset = val_subset.copy()

train_subset['risk_factor'] = train_subset['user_hash'].map(scores).fillna(0.5)
val_subset['risk_factor'] = val_subset['user_hash'].map(scores).fillna(0.5)

# Fill numeric NaNs
numeric_cols = train_df.select_dtypes(include=['number']).columns
cols_to_fill = [c for c in numeric_cols if c != 'is_cheating']
train_medians = train_subset[cols_to_fill].median() # Calculate medians from train only

train_subset[cols_to_fill] = train_subset[cols_to_fill].fillna(train_medians)
val_subset[cols_to_fill] = val_subset[cols_to_fill].fillna(train_medians)

# Prepare Model Inputs
X_train = train_subset.drop(columns=[Target, id_column])
y_train = train_subset[Target]

X_val = val_subset.drop(columns=[Target, id_column])
y_val = val_subset[Target]

ratio_neg_to_pos = (len(y_train) - y_train.sum()) / y_train.sum()

# 4. Hyperparameter Tuning
learning_rates = [0.01, 0.05, 0.1]
max_depths = [5, 9, 12]
n_estimators_list = [100, 300, 500]
subsamples = [0.7, 0.8]
colsamples = [0.7, 0.8]

best_mse = float('inf')
best_params = {}

print(f"\nStarting Grid Search...")
print(f"X_train shape: {X_train.shape}")

# Total combinations: 3*3*3*2*2 = 108. Might take a while.
# Let's prune slightly for speed if needed, or run full if user wants rigorous search.
# Given the user asked, we'll run it.

for lr in learning_rates:
    for depth in max_depths:
        for n_est in n_estimators_list:
            for subsample in subsamples:
                for colsample in colsamples:
                    model = XGBClassifier(
                        objective='binary:logistic',
                        n_estimators=n_est,
                        learning_rate=lr,
                        max_depth=depth,
                        subsample=subsample,
                        colsample_bytree=colsample,
                        scale_pos_weight=ratio_neg_to_pos,
                        use_label_encoder=False,
                        eval_metric='logloss',
                        random_state=42,
                        tree_method='hist'
                    )
                    
                    model.fit(X_train, y_train, verbose=False)
                    
                    # Predict probabilities
                    preds_proba = model.predict_proba(X_val)[:, 1]
                    
                    # Calculate MSE
                    mse = mean_squared_error(y_val, preds_proba)
                    
                    print(f"LR:{lr} Dep:{depth} Nest:{n_est} Sub:{subsample} Col:{colsample} => MSE:{mse:.6f}")
                    
                    if mse < best_mse:
                        best_mse = mse
                        best_params = {
                            'learning_rate': lr, 
                            'max_depth': depth,
                            'n_estimators': n_est,
                            'subsample': subsample,
                            'colsample_bytree': colsample
                        }

print(f"\nBest Parameters: {best_params}")
print(f"Lowest MSE: {best_mse:.6f}")
