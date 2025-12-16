import networkx as nx
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, mean_squared_error

# ... (skip to model function)

def model(final_df):
    Target = 'is_cheating'
    id_column = 'user_hash'

    # Drop target and id column from features
    X = final_df.drop(columns=[Target, id_column])
    y = final_df[Target]

    valid_rows = y.dropna().index
    X = X.loc[valid_rows]
    y = y.loc[valid_rows]

    X_train, X_val, y_train, y_val  = train_test_split(
        X,y,
        test_size= 0.2,
        random_state= 42,
        stratify=y
    )
    ratio_neg_to_pos = (len(y_train) - y_train.sum()) / y_train.sum()

    lgb_model = LGBMClassifier(
        objective='binary',
        n_estimators=100,
        learning_rate=0.01,
        max_depth=12,
        subsample=0.8,
        colsample_bytree=0.7,
        scale_pos_weight=ratio_neg_to_pos,
        metric='binary_logloss',
        random_state=42,
        verbose=-1  # Suppress warnings
    )
    
    # LightGBM uses 'eval_set' and 'eval_metric' in fit, similar to XGBoost but sometimes 'eval_metric' is list
    lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    
    preds = lgb_model.predict(X_val)
    preds_proba = lgb_model.predict_proba(X_val)[:, 1]
    print("F1 Score:", f1_score(y_val, preds))
    print("ROC AUC:", roc_auc_score(y_val, preds))
    print("Confusion Matrix:\n", confusion_matrix(y_val, preds))
    print("MSE:", mean_squared_error(y_val, preds_proba))
    
    return lgb_model
df_train = pd.read_csv('train.csv')
edges = pd.read_csv('social_graph.csv')

# Update is_cheating based on high_conf_clean
# If high_conf_clean is 1 and is_cheating is NaN, set is_cheating to 0
mask = (df_train['high_conf_clean'] == 1) & (df_train['is_cheating'].isna())
df_train.loc[mask, 'is_cheating'] = 0
G = nx.from_pandas_edgelist(edges, 'user_a', 'user_b')
G.add_nodes_from(df_train['user_hash'])

# 2. Add your known labels to the graph
# labels_dict should have: {user_hash: 1 for cheaters, user_hash: 0 for clean}
# Leave the NaNs and Ghost Users as 'None' or empty
node_labels = df_train.set_index('user_hash')['is_cheating'].to_dict()
nx.set_node_attributes(G, node_labels, 'label')



def get_social_risk_scores(G, iterations=10):
    # Start with known labels (0 or 1), others are 0.5 (unknown)
    scores = {}
    for node in G.nodes():
        val = G.nodes[node].get('label', 0.5)
        if pd.isna(val):
            val = 0.5
        scores[node] = val

    for _ in range(iterations):
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
    return scores




risk_scores = get_social_risk_scores(G)

# Add risk_factor column to df_train
df_train['risk_factor'] = df_train['user_hash'].map(risk_scores)
print(df_train[['user_hash', 'risk_factor']].head(10))

trained_model = model(df_train)

print(trained_model)

# --- Process Test Data ---
print("\n--- Processing Test Data ---")
df_test = pd.read_csv('test.csv')

# 1. Fill missing values in test data (using training data medians to avoid data leakage)
numeric_cols = df_train.select_dtypes(include=['number']).columns
cols_to_fill = [c for c in numeric_cols if c != 'is_cheating']

# Only fill columns that actually exist in the test set
# 'high_conf_clean' and 'risk_factor' (which we add later) might be in cols_to_fill but not in df_test yet/ever
cols_to_fill_test = [c for c in cols_to_fill if c in df_test.columns]

train_medians = df_train[cols_to_fill_test].median()
df_test[cols_to_fill_test] = df_test[cols_to_fill_test].fillna(train_medians)

# 2. Add risk_factor to test data
# Note: Ensure all test users are in the graph. If not, they get 0.5 default from get_social_risk_scores 
# logic if we were to rerun it, BUT the risk_scores dictionary already contains scores for all nodes in G.
# We need to check if test users are in G. If not, they are essentially unknown.
# The previous risk_scores calculation covered all nodes in edges + all nodes in df_train.
# If a test user appears in edges, they have a score. If they are completely new (not in edges, not in train),
# they won't be in risk_scores. We should handle this.
# However, usually test users are part of the social graph provided.
# Test users: 48416
print(f"Test users: {len(df_test)}")
df_test['risk_factor'] = df_test['user_hash'].map(risk_scores).fillna(0.5)

# Add missing columns expected by the model (e.g., high_conf_clean not in test.csv)
train_features = trained_model.booster_.feature_name()
for col in train_features:
    if col not in df_test.columns:
        print(f"Adding missing column '{col}' with default 0")
        df_test[col] = 0

# 3. Predict using the trained model
X_test = df_test.drop(columns=['user_hash'])
# Ensure columns match training features
train_features = trained_model.booster_.feature_name()
X_test = X_test[train_features]

probabilities = trained_model.predict_proba(X_test)[:, 1]

# 4. Create submission file
submission = pd.DataFrame({
    'user_hash': df_test['user_hash'],
    'prediction': probabilities
})

print(submission.head())
submission.to_csv('submission_with_risk.csv', index=False)
print("Submission saved to submission_with_risk.csv")
