# %pip install neo4j

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from neo4j import GraphDatabase
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, confusion_matrix, classification_report
)

"""CONFIGURATION"""

NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "fraud@2213"

BASE_DIR = os.path.dirname(os.getcwd())
OUTPUT_DIR = BASE_DIR

"""BUSINESS UNDERSTANDING

PROBLEM STATEMENT:
    Identify customers who pose fraud risk AND generate high transaction volume
    (high-value but risky). Enable targeted monitoring and risk-based pricing.

    PRIMARY OBJECTIVES:
    1. Detect network-based fraud indicators using graph algorithms
    2. Score customer risk using graph centrality + transaction patterns
    3. Segment customers into actionable risk/profitability quadrants
    4. Demonstrate graph features improve fraud detection vs. baseline

    SUCCESS METRICS:
    - Identify ≥5 distinct customer communities via Louvain
    - Calculate centrality scores (PageRank, Betweenness) for all 500 customers
    - Generate 64-dim embeddings via FastRP
    - Create 4-segment quadrant matrix
    - Show positive lift from graph features (vs. transaction-only baseline)
    - Export top 30 flagged customers for investigation

    DATASET:
    - Customers: 500 records
    - Transactions: 65,010 records (Jul 3 - Oct 1, 2021)
    - Network density: Medium (each customer ~130 transactions)

DATA UNDERSTANDING

CUSTOMER DATA (customers.csv):
    - Rows: 500
    - Columns: customerID, name, email, phoneNumber, address, account, sortCode
    - Key: account number links customers to transactions
    - Completeness: 100% (no missing values)

    TRANSACTION DATA (transactions.csv):
    - Rows: 65,010
    - Columns: fromAccount, toAccount, amount, currency, transactionID, dateTime
    - Time Range: Jul 3, 2021 - Oct 1, 2021 (91 days)
    - Avg Transaction: $12,220.58 USD
    - Median Transaction: $555.43 USD
    - Max Transaction: $99,997.83 USD
    - Currency: 100% USD

    GRAPH SCHEMA:
    Nodes:
      - Customer (customerID, name, account, profitability_score, etc.)
      - Transaction relationships: (Customer)-[:SENT {amount, dateTime}]->(Customer)

    NETWORK CHARACTERISTICS:
    - Avg Transactions per Customer: 130
    - Unique From Accounts: 32,345 (includes external accounts)
    - Unique To Accounts: 32,531 (includes external accounts)
    - Network Type: Directed, weighted by amount

PHASE 1: FEATURE EXTRACTION FROM NEO4J - This code should be executed locally any editor like VsCode and upload the generated file for further processing in this notebook.
"""

def extract_features_from_neo4j(uri, auth):
    print("PHASE 1: FEATURE EXTRACTION FROM NEO4J")

    driver = GraphDatabase.driver(uri, auth=auth)

    try:
        query = """
        MATCH (c:Customer)
        RETURN
          c.customerID AS customer_id,
          c.name AS name,
          c.account AS account,
          c.out_degree AS out_degree,
          c.in_degree AS in_degree,
          c.pagerank_score AS pagerank_score,
          c.betweenness_score AS betweenness_score,
          c.community_id AS community_id,
          c.reciprocal_connections AS reciprocal_connections,
          coalesce(c.circular_flow_flag, 0) AS circular_flow_flag,
          c.num_sent_transactions AS num_sent_transactions,
          c.total_sent_amount AS total_sent_amount,
          c.avg_sent_amount AS avg_sent_amount,
          c.num_received_transactions AS num_received_transactions,
          c.total_received_amount AS total_received_amount,
          c.profitability_score AS profitability_score,
          c.embedding AS embedding
        """

        with driver.session() as session:
            print("Session started")
            result = session.run(query)
            features_df = pd.DataFrame([dict(record) for record in result])

        print(f"Extracted {len(features_df)} customers")

        #parse embeddings
        def parse_embedding(emb):
            if isinstance(emb, (list, np.ndarray)):
                return np.array(emb, dtype=float)
            return np.zeros(64)

        print("Parsing embeddings")
        features_df['embedding_array'] = features_df['embedding'].apply(parse_embedding)
        features_df['embedding_norm'] = features_df['embedding_array'].apply(np.linalg.norm)
        features_df['embedding_mean'] = features_df['embedding_array'].apply(np.mean)
        features_df['embedding_std'] = features_df['embedding_array'].apply(np.std)

        #drop original embedding column
        features_df = features_df.drop('embedding', axis=1)

        #fill NaN with 0
        features_df = features_df.fillna(0)

        #save
        csv_path = os.path.join(OUTPUT_DIR, 'customer_features_with_embeddings.csv')
        features_df.to_csv(csv_path, index=False)

        return features_df

    except Exception as e:
        print(f"Error connecting to Neo4j: {e}")
        raise

    finally:
        driver.close()

"""PHASE 2: EXPLORATORY DATA ANALYSIS"""

def exploratory_data_analysis(features_df):
    print("PHASE 2: EXPLORATORY DATA ANALYSIS")

    print("\nFeature Summary Statistics:")
    print(features_df[[
        'out_degree', 'in_degree', 'pagerank_score', 'total_sent_amount',
        'profitability_score', 'reciprocal_connections'
    ]].describe())

    #visualizations
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    #out-degree
    axes[0, 0].hist(features_df['out_degree'], bins=30, edgecolor='black', color='steelblue')
    axes[0, 0].set_title('Out-Degree Distribution')
    axes[0, 0].set_xlabel('Number of Recipients')
    axes[0, 0].set_ylabel('Count')

    #in-degree
    axes[0, 1].hist(features_df['in_degree'], bins=30, edgecolor='black', color='coral')
    axes[0, 1].set_title('In-Degree Distribution')
    axes[0, 1].set_xlabel('Number of Senders')

    #total sent amount
    axes[0, 2].hist(features_df['total_sent_amount'], bins=30, edgecolor='black', color='lightgreen')
    axes[0, 2].set_title('Total Sent Amount Distribution')
    axes[0, 2].set_xlabel('Amount (USD)')
    axes[0, 2].set_yscale('log')

    #profitability
    axes[1, 0].hist(features_df['profitability_score'], bins=20, edgecolor='black', color='gold')
    axes[1, 0].set_title('Profitability Score Distribution')
    axes[1, 0].set_xlabel('Score [0–100]')

    #pageRank
    axes[1, 1].hist(features_df['pagerank_score'].fillna(0), bins=20, edgecolor='black', color='purple')
    axes[1, 1].set_title('PageRank Distribution')
    axes[1, 1].set_xlabel('PageRank Score')

    #scatter: Profitability vs Volume
    axes[1, 2].scatter(features_df['total_sent_amount'], features_df['profitability_score'],
                       alpha=0.6, s=30, c=features_df['out_degree'], cmap='viridis')
    axes[1, 2].set_title('Profitability vs Transaction Volume')
    axes[1, 2].set_xlabel('Total Sent Amount (USD)')
    axes[1, 2].set_ylabel('Profitability Score')

    plt.tight_layout()
    eda_path = os.path.join(OUTPUT_DIR, 'eda_distributions.png')
    plt.savefig(eda_path, dpi=300, bbox_inches='tight')
    print(f"EDA plot saved: {eda_path}")
    plt.close()

"""PHASE 3: MACHINE LEARNING MODELLING"""

def prepare_and_train_models(features_df):
    print("PHASE 3: MACHINE LEARNING MODELLING")

    risk_base = (
        features_df['out_degree'].fillna(0) +
        (features_df['total_sent_amount'].fillna(0) / features_df['total_sent_amount'].max()) +
        (features_df['pagerank_score'].fillna(0) / features_df['pagerank_score'].max()) * 0.5
    )

    features_df['risk_rank'] = risk_base.rank(method='first', ascending=False)
    target_rate = 0.20
    threshold_rank = len(features_df) * target_rate
    features_df['fraud_label'] = (features_df['risk_rank'] <= threshold_rank).astype(int)

    print(f"\nFraud Label Distribution:")
    print(f"  Fraud rate: {features_df['fraud_label'].mean():.2%}")
    print(f"  Frauds: {features_df['fraud_label'].sum()}")
    print(f"  Non-frauds: {(1 - features_df['fraud_label']).sum()}")

    graph_features = [
        'out_degree', 'in_degree', 'pagerank_score', 'betweenness_score',
        'reciprocal_connections', 'circular_flow_flag',
        'embedding_norm', 'embedding_mean', 'embedding_std'
    ]

    transaction_features = [
        'num_sent_transactions', 'total_sent_amount', 'avg_sent_amount',
        'num_received_transactions', 'total_received_amount'
    ]

    all_features = graph_features + transaction_features

    X = features_df[all_features].fillna(0)
    y = features_df['fraud_label']

    #normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=all_features)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTrain/Test Split:")
    print(f"  Train: {len(X_train)} samples, fraud rate {y_train.mean():.2%}")
    print(f"  Test: {len(X_test)} samples, fraud rate {y_test.mean():.2%}")

    #GRAPH-ENHANCED MODEL
    print(f"\nTraining Graph-Enhanced Model")
    model_graph = LogisticRegression(max_iter=1000, random_state=42)
    model_graph.fit(X_train, y_train)

    y_pred_graph = model_graph.predict(X_test)
    y_pred_proba_graph = model_graph.predict_proba(X_test)[:, 1]
    auc_graph = roc_auc_score(y_test, y_pred_proba_graph)

    print("GRAPH-ENHANCED MODEL")
    print(f"Precision: {precision_score(y_test, y_pred_graph):.3f}")
    print(f"Recall: {recall_score(y_test, y_pred_graph):.3f}")
    print(f"F1-Score: {f1_score(y_test, y_pred_graph):.3f}")
    print(f"ROC-AUC: {auc_graph:.3f}")

    #BASELINE MODEL
    print(f"\nTraining Baseline Model (Transaction Features Only)")
    X_train_baseline = X_train[transaction_features]
    X_test_baseline = X_test[transaction_features]

    model_baseline = LogisticRegression(max_iter=1000, random_state=42)
    model_baseline.fit(X_train_baseline, y_train)

    y_pred_baseline = model_baseline.predict(X_test_baseline)
    y_pred_proba_baseline = model_baseline.predict_proba(X_test_baseline)[:, 1]
    auc_baseline = roc_auc_score(y_test, y_pred_proba_baseline)

    print("BASELINE MODEL (Transaction Features Only)")
    print(f"Precision: {precision_score(y_test, y_pred_baseline):.3f}")
    print(f"Recall: {recall_score(y_test, y_pred_baseline):.3f}")
    print(f"F1-Score: {f1_score(y_test, y_pred_baseline):.3f}")
    print(f"ROC-AUC: {auc_baseline:.3f}")

    #LIFT
    lift = (auc_graph - auc_baseline) / auc_baseline * 100 if auc_baseline > 0 else 0

    print("MODEL COMPARISON")
    print(f"Baseline ROC-AUC: {auc_baseline:.3f}")
    print(f"Graph-Enhanced ROC-AUC: {auc_graph:.3f}")
    print(f"LIFT FROM GRAPH FEATURES: +{lift:.1f}%")

    #FEATURE IMPORTANCE
    print(f"\nCalculating Feature Importance...")
    feature_importance = pd.DataFrame({
        'feature': all_features,
        'coefficient': model_graph.coef_[0]
    })
    feature_importance['abs_coeff'] = abs(feature_importance['coefficient'])
    feature_importance = feature_importance.sort_values('abs_coeff', ascending=False)

    print("\nTop 15 Most Important Features:")
    print(feature_importance.head(15)[['feature', 'coefficient', 'abs_coeff']].to_string(index=False))

    #plot feature importance
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(feature_importance.head(15)['feature'],
            feature_importance.head(15)['abs_coeff'], color='steelblue')
    ax.set_xlabel('Absolute Coefficient Value')
    ax.set_title('Top 15 Feature Importance (Graph-Enhanced Model)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    fi_path = os.path.join(OUTPUT_DIR, 'feature_importance.png')
    plt.savefig(fi_path, dpi=300, bbox_inches='tight')
    print(f"Feature importance plot saved: {fi_path}")
    plt.close()

    #ROC CURVES
    fpr_graph, tpr_graph, _ = roc_curve(y_test, y_pred_proba_graph)
    fpr_baseline, tpr_baseline, _ = roc_curve(y_test, y_pred_proba_baseline)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(fpr_graph, tpr_graph, label=f'Graph-Enhanced (AUC={auc_graph:.3f})', linewidth=2, color='green')
    ax.plot(fpr_baseline, tpr_baseline, label=f'Baseline (AUC={auc_baseline:.3f})', linewidth=2, color='orange')
    ax.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve: Graph-Enhanced vs. Baseline', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    roc_path = os.path.join(OUTPUT_DIR, 'roc_curves.png')
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    print(f"ROC curves saved: {roc_path}")
    plt.close()

    return features_df, model_graph, scaler, X, all_features

"""PHASE 4: RISK-PROFITABILITY SEGMENTATION"""

def segment_customers(features_df, model_graph, scaler, X, all_features):
    print("PHASE 4: RISK-PROFITABILITY SEGMENTATION")

    #generate risk scores for all customers
    X_all = pd.DataFrame(scaler.transform(X), columns=all_features)
    features_df['risk_score_ml'] = model_graph.predict_proba(X_all)[:, 1]

    #quadrant thresholds
    risk_threshold = features_df['risk_score_ml'].median()
    profit_threshold = features_df['profitability_score'].median()

    print(f"\nQuadrant Thresholds:")
    print(f"  Risk threshold (median): {risk_threshold:.3f}")
    print(f"  Profit threshold (median): {profit_threshold:.1f}")

    #classify quadrants
    def classify_quadrant(row):
        risk_high = row['risk_score_ml'] > risk_threshold
        profit_high = row['profitability_score'] > profit_threshold

        if risk_high and profit_high:
            return 'High Risk / High Profit'
        elif risk_high and not profit_high:
            return 'High Risk / Low Profit'
        elif not risk_high and profit_high:
            return 'Low Risk / High Profit'
        else:
            return 'Low Risk / Low Profit'

    features_df['quadrant'] = features_df.apply(classify_quadrant, axis=1)

    print("QUADRANT DISTRIBUTION")
    quad_counts = features_df['quadrant'].value_counts()
    quad_pcts = features_df['quadrant'].value_counts(normalize=True) * 100

    for quadrant in ['Low Risk / High Profit', 'Low Risk / Low Profit',
                     'High Risk / High Profit', 'High Risk / Low Profit']:
        count = quad_counts.get(quadrant, 0)
        pct = quad_pcts.get(quadrant, 0)
        print(f"{quadrant:.<40} {count:>3} ({pct:>5.1f}%)")

    #visualize quadrants
    fig, ax = plt.subplots(figsize=(12, 8))

    colors = {
        'High Risk / High Profit': 'red',
        'High Risk / Low Profit': 'orange',
        'Low Risk / High Profit': 'green',
        'Low Risk / Low Profit': 'blue'
    }

    for quadrant, color in colors.items():
        mask = features_df['quadrant'] == quadrant
        ax.scatter(
            features_df[mask]['risk_score_ml'],
            features_df[mask]['profitability_score'],
            label=f"{quadrant} (n={mask.sum()})",
            alpha=0.6, s=80, c=color
        )

    #add threshold lines
    ax.axvline(risk_threshold, linestyle='--', alpha=0.5, color='gray', linewidth=2)
    ax.axhline(profit_threshold, linestyle='--', alpha=0.5, color='gray', linewidth=2)

    ax.set_xlabel('Risk Score (ML Model)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Profitability Score', fontsize=12, fontweight='bold')
    ax.set_title('Customer Risk vs. Profitability Matrix', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    matrix_path = os.path.join(OUTPUT_DIR, 'risk_profitability_matrix.png')
    plt.savefig(matrix_path, dpi=300, bbox_inches='tight')
    print(f"\nRisk-Profitability matrix saved: {matrix_path}")
    plt.close()

    #segment analysis
    print("SEGMENT STATISTICS")

    for quadrant in ['Low Risk / High Profit', 'High Risk / High Profit',
                     'High Risk / Low Profit', 'Low Risk / Low Profit']:
        seg = features_df[features_df['quadrant'] == quadrant]
        if len(seg) > 0:
            print(f"\n{quadrant}")
            print(f"  Count: {len(seg)}")
            print(f"  Avg Risk Score: {seg['risk_score_ml'].mean():.3f}")
            print(f"  Avg Profitability: {seg['profitability_score'].mean():.1f}")
            print(f"  Avg Transaction Volume: ${seg['total_sent_amount'].mean():,.0f}")
            print(f"  Avg PageRank: {seg['pagerank_score'].mean():.3f}")

    #export results
    seg_path = os.path.join(OUTPUT_DIR, 'customer_segmentation_results.csv')
    features_df.to_csv(seg_path, index=False)
    print(f"\nSegmentation results saved: {seg_path}")

    return features_df

"""PHASE 5: TOP RISKY CUSTOMERS FOR INVESTIGATION"""

def export_top_risky_customers(features_df):
    print("PHASE 9: TOP RISKY CUSTOMERS FOR INVESTIGATION")

    top_risky = features_df[
        features_df['risk_score_ml'] > features_df['risk_score_ml'].quantile(0.80)
    ].nlargest(30, 'risk_score_ml')[[
        'customer_id', 'name', 'account', 'risk_score_ml', 'profitability_score',
        'out_degree', 'in_degree', 'pagerank_score', 'betweenness_score',
        'total_sent_amount', 'circular_flow_flag', 'quadrant'
    ]]

    print(f"\nTop 30 Customers Flagged for Investigation:")
    print(top_risky.to_string())

    risky_path = os.path.join(OUTPUT_DIR, 'top_risky_customers_for_review.csv')
    top_risky.to_csv(risky_path, index=False)
    print(f"\nTop risky customers saved: {risky_path}")

"""PHASE 6: MAIN EXECUTION"""

def main():
    print("TASK 1: FRAUD RISK & PROFITABILITY ANALYSIS VIA GRAPH EMBEDDINGS")

    try:
        features_df = pd.read_csv('customer_features_with_embeddings.csv')


        #Phase 2: EDA
        exploratory_data_analysis(features_df)

        #Phase 3: ML Modelling
        features_df, model_graph, scaler, X, all_features = prepare_and_train_models(features_df)

        #Phase 4: Segmentation
        features_df = segment_customers(features_df, model_graph, scaler, X, all_features)

        #Phase 5: Top Risky
        export_top_risky_customers(features_df)

        print("PIPELINE COMPLETE")
        print(f"\nOutputs generated in: {OUTPUT_DIR}")

    except Exception as e:
        print(f"\n Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()