import pandas as pd
import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, mean_squared_error
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def main():
    print("--- Model Comparison Pipeline ---")
    
    # 1. Load Data
    print("Loading data...")
    df_train = pd.read_csv('train.csv')
    edges = pd.read_csv('social_graph.csv')
    
    # 2. Preprocess 'is_cheating' (same logic as before)
    mask = (df_train['high_conf_clean'] == 1) & (df_train['is_cheating'].isna())
    df_train.loc[mask, 'is_cheating'] = 0
    
    # Drop rows where target is still NaN
    df_train = df_train.dropna(subset=['is_cheating'])
    df_train['is_cheating'] = df_train['is_cheating'].astype(int)
    
    # 3. SPLIT DATA FIRST (Critical Step)
    print("Splitting data into Train (80%) and Validation (20%) BEFORE risk scoring...")
    train_subset, val_subset = train_test_split(
        df_train, test_size=0.2, random_state=42, stratify=df_train['is_cheating']
    )
    
    print(f"Train size: {len(train_subset)}, Val size: {len(val_subset)}")
    
    # 4. Build Graph
    print("Building Social Graph...")
    G = nx.from_pandas_edgelist(edges, 'user_a', 'user_b')
    # Add all nodes in train/val to ensure they exist in graph
    G.add_nodes_from(df_train['user_hash'])
    
    # 5. Risk Score Propagation (Using ONLY Train Labels)
    print("Calculating Risk Scores using ONLY Train labels...")
    node_labels = train_subset.set_index('user_hash')['is_cheating'].to_dict()
    
    # Initialize scores
    scores = {}
    for node in G.nodes():
        if node in node_labels:
            scores[node] = node_labels[node]
        else:
            scores[node] = 0.5
            
    # Propagate
    iterations = 10 
    for _ in range(iterations):
        new_scores = {}
        for node in G.nodes():
            neighbors = list(G.neighbors(node))
            if not neighbors:
                new_scores[node] = scores[node]
                continue
            
            neighbor_avg = sum(scores[n] for n in neighbors) / len(neighbors)
            
            # CLAMP ONLY KNOWN TRAIN NODES
            if node in node_labels:
                new_scores[node] = node_labels[node]
            else:
                new_scores[node] = neighbor_avg
        scores = new_scores
    
    print("Risk scores calculated.")
    
    # 6. Map Risk Scores back to Train and Val
    # Note: Val nodes were treated as unknown (0.5 or neighbor avg) during propagation,
    # so their final score depends on their neighbors, not their own label.
    train_subset = train_subset.copy()
    val_subset = val_subset.copy()
    
    train_subset['risk_factor'] = train_subset['user_hash'].map(scores).fillna(0.5)
    val_subset['risk_factor'] = val_subset['user_hash'].map(scores).fillna(0.5)
    
    # 7. Fill Missing Numeric Features
    numeric_cols = df_train.select_dtypes(include=['number']).columns
    cols_to_fill = [c for c in numeric_cols if c != 'is_cheating']
    
    train_medians = train_subset[cols_to_fill].median()
    
    train_subset[cols_to_fill] = train_subset[cols_to_fill].fillna(train_medians)
    val_subset[cols_to_fill] = val_subset[cols_to_fill].fillna(train_medians)
    
    # 8. Prepare X and y
    target = 'is_cheating'
    drop_cols = ['user_hash', target]
    
    X_train = train_subset.drop(columns=drop_cols)
    y_train = train_subset[target]
    
    X_val = val_subset.drop(columns=drop_cols)
    y_val = val_subset[target]
    
    ratio = (len(y_train) - y_train.sum()) / y_train.sum()
    
    results = []
    
    # --- MODEL 1: XGBoost ---
    print("\nTraining XGBoost...")
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
    xgb.fit(X_train, y_train)
    xgb_probs = xgb.predict_proba(X_val)[:, 1]
    xgb_preds = (xgb_probs > 0.5).astype(int)
    
    results.append({
        'Model': 'XGBoost',
        'MSE': mean_squared_error(y_val, xgb_probs),
        'F1': f1_score(y_val, xgb_preds),
        'AUC': roc_auc_score(y_val, xgb_probs)
    })
    
    # --- MODEL 2: LightGBM ---
    print("Training LightGBM...")
    lgbm = LGBMClassifier(
        n_estimators=100,
        learning_rate=0.01,
        max_depth=12,
        subsample=0.8,
        colsample_bytree=0.7,
        scale_pos_weight=ratio,
        metric='binary_logloss',
        random_state=42,
        verbose=-1
    )
    lgbm.fit(X_train, y_train)
    lgb_probs = lgbm.predict_proba(X_val)[:, 1]
    lgb_preds = (lgb_probs > 0.5).astype(int)
    
    results.append({
        'Model': 'LightGBM',
        'MSE': mean_squared_error(y_val, lgb_probs),
        'F1': f1_score(y_val, lgb_preds),
        'AUC': roc_auc_score(y_val, lgb_probs)
    })
    
    # --- MODEL 3: Random Forest (Baseline) ---
    print("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_probs = rf.predict_proba(X_val)[:, 1]
    rf_preds = (rf_probs > 0.5).astype(int)
    
    results.append({
        'Model': 'Random Forest',
        'MSE': mean_squared_error(y_val, rf_probs),
        'F1': f1_score(y_val, rf_preds),
        'AUC': roc_auc_score(y_val, rf_probs)
    })
    
    # --- MODEL 4: Logistic Regression (Risk Factor Only) ---
    print("Training Logistic Regression (Risk Factor Only)...")
    # Reshape for single feature
    X_train_lr = X_train[['risk_factor']]
    X_val_lr = X_val[['risk_factor']]
    
    lr = LogisticRegression(class_weight='balanced', random_state=42)
    lr.fit(X_train_lr, y_train)
    lr_probs = lr.predict_proba(X_val_lr)[:, 1]
    lr_preds = (lr_probs > 0.5).astype(int)
    
    results.append({
        'Model': 'Logistic Regression (Risk Only)',
        'MSE': mean_squared_error(y_val, lr_probs),
        'F1': f1_score(y_val, lr_preds),
        'AUC': roc_auc_score(y_val, lr_probs)
    })
    
    # 9. Results Table
    results_df = pd.DataFrame(results).sort_values(by='MSE')
    print("\n--- RESULTS ---")
    print(results_df)

    # Feature Importance (from XGBoost as representative)
    print("\nXGBoost Feature Importance:")
    feat_imp = pd.Series(xgb.feature_importances_, index=X_train.columns).sort_values(ascending=False).head(10)
    print(feat_imp)

if __name__ == "__main__":
    main()
