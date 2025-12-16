import pandas as pd
import numpy as np
import networkx as nx
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error, f1_score, roc_auc_score
from xgboost import XGBClassifier

def calculate_risk_scores(G, train_labels_dict, iterations=10):
    scores = {}
    # Initialize
    for node in G.nodes():
        if node in train_labels_dict:
            scores[node] = train_labels_dict[node]
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
            
            # NOTE: Only clamp nodes that are in the TRAINING set for this fold
            if node in train_labels_dict:
                new_scores[node] = train_labels_dict[node]
            else:
                new_scores[node] = neighbor_avg
        scores = new_scores
    return scores

def main():
    print("--- XGBoost K-Fold Cross Validation ---")
    
    # 1. Load Data
    print("Loading data...")
    df_train = pd.read_csv('train.csv')
    edges = pd.read_csv('social_graph.csv')
    
    # 2. Preprocess 'is_cheating'
    mask = (df_train['high_conf_clean'] == 1) & (df_train['is_cheating'].isna())
    df_train.loc[mask, 'is_cheating'] = 0
    df_train = df_train.dropna(subset=['is_cheating'])
    df_train['is_cheating'] = df_train['is_cheating'].astype(int)
    
    # 3. Build Graph (Structure is constant)
    print("Building Social Graph...")
    G = nx.from_pandas_edgelist(edges, 'user_a', 'user_b')
    G.add_nodes_from(df_train['user_hash'])
    
    # 4. K-Fold CV
    k = 5
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    
    mses = []
    aucs = []
    f1s = []
    
    # Prepare data for splitting
    X = df_train.drop(columns=['is_cheating']) # user_hash is still needed for mapping, dropped later
    y = df_train['is_cheating']
    
    print(f"Starting {k-1}-Fold CV...")
    
    fold = 0
    for train_index, val_index in kf.split(X, y):
        fold += 1
        print(f"\nFold {fold}/{k}")
        
        # Split Data
        X_train_fold = X.iloc[train_index].copy()
        X_val_fold = X.iloc[val_index].copy()
        y_train_fold = y.iloc[train_index]
        y_val_fold = y.iloc[val_index]
        
        # --- CRITICAL: Calculate Risk Scores using ONLY Train Fold Labels ---
        print("  Calculating Risk Scores (Train Fold Only)...")
        # Create dict: {user_hash: label}
        train_labels_map = dict(zip(X_train_fold['user_hash'], y_train_fold))
        
        risk_scores = calculate_risk_scores(G, train_labels_map)
        
        # Map Scores
        X_train_fold['risk_factor'] = X_train_fold['user_hash'].map(risk_scores).fillna(0.5)
        X_val_fold['risk_factor'] = X_val_fold['user_hash'].map(risk_scores).fillna(0.5)
        
        # Fill Medians (Calculated on Train Fold)
        numeric_cols = X_train_fold.select_dtypes(include=['number']).columns
        # Note: risk_factor is numeric, so it's included. But it shouldn't have NaNs anymore.
        # Other features might.
        train_medians = X_train_fold[numeric_cols].median()
        X_train_fold[numeric_cols] = X_train_fold[numeric_cols].fillna(train_medians)
        X_val_fold[numeric_cols] = X_val_fold[numeric_cols].fillna(train_medians)
        
        # Drop non-feature columns
        drop_cols = ['user_hash']
        X_train_final = X_train_fold.drop(columns=drop_cols)
        X_val_final = X_val_fold.drop(columns=drop_cols)
        
        # Train XGBoost
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
        xgb.fit(X_train_final, y_train_fold)
        
        # Evaluate
        probs = xgb.predict_proba(X_val_final)[:, 1]
        preds = (probs > 0.5).astype(int)
        
        mse = mean_squared_error(y_val_fold, probs)
        auc = roc_auc_score(y_val_fold, probs)
        f1 = f1_score(y_val_fold, preds)
        
        print(f"  MSE: {mse:.6f} | AUC: {auc:.4f} | F1: {f1:.4f}")
        
        mses.append(mse)
        aucs.append(auc)
        f1s.append(f1)

    print("\n--- K-Fold Results Summary ---")
    print(f"Average MSE: {np.mean(mses):.6f} (+/- {np.std(mses):.6f})")
    print(f"Average AUC: {np.mean(aucs):.4f}")
    print(f"Average F1 : {np.mean(f1s):.4f}")

if __name__ == "__main__":
    main()
