import networkx as nx
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, mean_squared_error
from sklearn.model_selection import StratifiedKFold

# 1. Load Data
edges = pd.read_csv('social_graph.csv')
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

# Preprocess 'is_cheating'
mask = (df_train['high_conf_clean'] == 1) & (df_train['is_cheating'].isna())
df_train.loc[mask, 'is_cheating'] = 0
df_train = df_train.dropna(subset=['is_cheating'])
df_train['is_cheating'] = df_train['is_cheating'].astype(int)

# Build Graph
G = nx.from_pandas_edgelist(edges, 'user_a', 'user_b')
G.add_nodes_from(df_train['user_hash'])
G.add_nodes_from(df_test['user_hash'])

def calculate_risk_scores(G, known_labels, iterations=10):
    scores = {}
    # Initialize
    for node in G.nodes():
        if node in known_labels:
            scores[node] = known_labels[node]
        else:
            scores[node] = 0.5
            
    # Propagate
    for _ in range(iterations):
        new_scores = {}
        for node in G.nodes():
            neighbors = list(G.neighbors(node))
            if not neighbors:
                new_scores[node] = scores[node]
                continue
            
            neighbor_avg = sum(scores[n] for n in neighbors) / len(neighbors)
            
            if node in known_labels:
                new_scores[node] = known_labels[node]
            else:
                new_scores[node] = neighbor_avg
        scores = new_scores
    return scores

# --- K-Fold Ensemble ---
k = 5
kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

# Features to fill (compute once for columns, valid per fold)
numeric_cols = df_train.select_dtypes(include=['number']).columns
cols_to_fill = [c for c in numeric_cols if c != 'is_cheating']
# intersect with test columns
cols_to_fill_test = [c for c in cols_to_fill if c in df_test.columns]

test_predictions_sum = np.zeros(len(df_test))
mses = []

print(f"Starting {k}-Fold Cross Validation Ensemble...")

X = df_train.drop(columns=['is_cheating'])
y = df_train['is_cheating']

fold = 0
for train_index, val_index in kf.split(X, y):
    fold += 1
    print(f"\n--- Fold {fold}/{k} ---")
    
    # 1. Split Data
    X_train_fold = X.iloc[train_index].copy()
    X_val_fold = X.iloc[val_index].copy()
    y_train_fold = y.iloc[train_index]
    y_val_fold = y.iloc[val_index]
    
    # 2. Calculate Risk Scores (Train Fold Knowledge Only)
    print("Calculating Risk Scores...")
    train_labels_map = dict(zip(X_train_fold['user_hash'], y_train_fold))
    risk_scores = calculate_risk_scores(G, train_labels_map)
    
    # 3. Add Features
    X_train_fold['risk_factor'] = X_train_fold['user_hash'].map(risk_scores).fillna(0.5)
    X_val_fold['risk_factor'] = X_val_fold['user_hash'].map(risk_scores).fillna(0.5)
    
    # Prepare Test Set for this fold
    df_test_fold = df_test.copy()
    df_test_fold['risk_factor'] = df_test_fold['user_hash'].map(risk_scores).fillna(0.5)
    
    # Fill Missing Numeric
    train_medians = X_train_fold[cols_to_fill].median()
    X_train_fold[cols_to_fill] = X_train_fold[cols_to_fill].fillna(train_medians)
    X_val_fold[cols_to_fill] = X_val_fold[cols_to_fill].fillna(train_medians)
    df_test_fold[cols_to_fill_test] = df_test_fold[cols_to_fill_test].fillna(train_medians)
    
    # Drop user_hash
    drop_cols = ['user_hash']
    X_train_final = X_train_fold.drop(columns=drop_cols)
    X_val_final = X_val_fold.drop(columns=drop_cols)
    X_test_final = df_test_fold.drop(columns=drop_cols)
    
    # Align Columns (add missing if any)
    # Typically not needed if schema is consistent, but good practice
    # XGBoost handles it, but let's be safe.
    
    # 4. Train Model
    ratio = (len(y_train_fold) - y_train_fold.sum()) / y_train_fold.sum()
    
    xgb = XGBClassifier(
        n_estimators=100,
        learning_rate=0.01,
        max_depth=12,
        subsample=0.8,
        colsample_bytree=0.7,
        scale_pos_weight=ratio,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        tree_method='hist'
    )
    
    xgb.fit(X_train_final, y_train_fold, eval_set=[(X_val_final, y_val_fold)], verbose=False)
    
    # Evaluate Validation
    val_probs = xgb.predict_proba(X_val_final)[:, 1]
    mse = mean_squared_error(y_val_fold, val_probs)
    auc = roc_auc_score(y_val_fold, val_probs)
    print(f"MSE: {mse:.6f} | AUC: {auc:.4f}")
    mses.append(mse)
    
    # Predict Test
    # Ensure Test columns align with Train
    train_cols = X_train_final.columns
    # Add any missing columns to test (initialized to 0)
    for c in train_cols:
        if c not in X_test_final.columns:
            X_test_final[c] = 0
    X_test_final = X_test_final[train_cols]
    
    test_probs = xgb.predict_proba(X_test_final)[:, 1]
    test_predictions_sum += test_probs

# Average Predictions
final_predictions = test_predictions_sum / k

print(f"\nAverage Validation MSE: {np.mean(mses):.6f}")

# Create Submission
submission = pd.DataFrame({
    'user_hash': df_test['user_hash'],
    'prediction': final_predictions
})

print(submission.head())
submission.to_csv('submission_with_risk.csv', index=False)
print("Submission saved to submission_with_risk.csv")
