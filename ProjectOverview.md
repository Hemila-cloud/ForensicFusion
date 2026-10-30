## An Integrated Forensic Intelligence Framework Using Benford’s Law and a Graph Neural Network for Explainable and Graph-Based Financial Fraud Detection
## 1. Project Overview

ForensicFusion is a financial fraud detection framework designed to identify suspicious transactions by combining statistical anomaly analysis, graph-based learning and explainable artificial intelligence (XAI).

Traditional fraud detection methods often examine transactions individually. However, financial fraud may involve interconnected accounts, unusual transaction amounts, repeated transfers and coordinated activity between multiple entities. ForensicFusion addresses these challenges by analyzing both transaction-level characteristics and relationships between financial accounts.

The project integrates three complementary components:

1. **Benford’s Law-based analysis** to identify unusual numerical patterns in transaction amounts.
2. **Graph Neural Networks (GNNs)** using GraphSAGE to learn relationships between accounts and classify suspicious transactions.
3. **Explainable AI and network analysis** using SHAP and NetworkX to interpret selected predictions and visualize suspicious account connections.

The framework also explores combining statistical and graph-based evidence into a hybrid fraud risk assessment system.

## 2. Problem Statement

The increasing use of digital banking and online payment systems has made financial transactions more frequent and complex. Fraudulent activities may be difficult to identify because suspicious transactions can resemble legitimate transactions when examined individually.

Existing approaches may face several limitations:

- Rule-based systems primarily detect predefined fraud patterns.
- Conventional machine learning models may not adequately represent relationships between accounts.
- Statistical anomaly detection alone cannot establish whether a transaction is fraudulent.
- Graph-based predictions can be difficult to interpret without additional explanation methods.
- Severe class imbalance makes it difficult to detect fraudulent transactions without generating excessive false alarms.

Therefore, this project investigates a framework that combines numerical anomaly features, account relationships, fraud classification, and explanatory analysis to support financial forensic investigation.

## 3. Project Objectives

The main objectives are:

- Analyze large-scale financial transaction data.
- Engineer transaction and account-behavior features.
- Incorporate Benford’s Law-based scores as numerical anomaly indicators.
- Represent financial accounts as graph nodes and transactions as graph edges.
- Apply GraphSAGE to learn account representations and predict transaction fraud.
- Evaluate fraud predictions using metrics suitable for imbalanced datasets.
- Analyze suspicious account connections using NetworkX.
- Generate SHAP explanations for selected model outputs.
- Investigate whether combining statistical and graph-based evidence improves fraud detection.

## 4. Dataset Description

The project uses a large financial transaction dataset containing **1,048,576 transactions and 10 columns** in the final feature-engineered dataset.

The data includes transaction details, account identifiers, account balances, fraud labels, and engineered statistical and behavioral features.


The final dataset also contains engineered features for balance changes, balance inconsistencies, transaction behavior, Benford’s Law scores and normalized forensic scores.

### Class imbalance

The dataset contains:

- Genuine transactions: **6,354,407**
- Fraudulent transactions: **8,213**
- Fraud proportion: approximately **0.1291%**

Because genuine transactions greatly outnumber fraudulent ones, accuracy alone is insufficient for evaluating the system. Precision, recall, F1-score, ROC-AUC, PR-AUC, and the confusion matrix are important evaluation measures.

## 5. System Architecture

The overall workflow is organized into the following stages:

1. Transaction data loading and preprocessing.
2. Transaction-level feature engineering.
3. Benford’s Law-based anomaly scoring.
4. Account-to-node mapping and graph construction.
5. Account-level feature preparation.
6. GraphSAGE-based transaction fraud prediction.
7. Suspicious account-network analysis.
8. SHAP-based explanation of selected outputs.
9. Evaluation and investigation-oriented visualization.

The statistical and graph-learning components provide complementary information. Benford’s Law identifies numerical irregularities, while GraphSAGE learns from account relationships. Network analysis and SHAP provide additional ways to inspect the resulting evidence.

## 6. Data Preprocessing and Feature Engineering

Feature engineering is a major part of the ForensicFusion pipeline. It transforms the original transaction data into useful statistical, behavioral, and graph-related features.

### 6.1 Balance-related features

Balance changes and inconsistencies are calculated to identify differences between the recorded transaction amount and changes in account balances.

Examples include:

- `origBalanceChange`
- `destBalanceChange`
- `origBalanceError`
- `destBalanceError`

These features help characterize transaction behavior and identify inconsistencies that may deserve further investigation.

### 6.2 Behavioral features

Additional features summarize the activity of originating and receiving accounts.

Examples include:

- `sender_transaction_count`
- `receiver_transaction_count`
- `sender_avg_amount`
- `receiver_avg_amount`
- `sender_amount_deviation`
- `receiver_amount_deviation`
- `sender_window_velocity`
- `receiver_window_velocity`

These features capture aspects of account activity, transaction magnitude, and activity within a time window.

### 6.3 Normalization

Selected numerical features are normalized before graph-model processing. Normalization helps place features with different numerical scales into a more comparable range.

The resulting feature arrays are used during graph preparation and model development.

## 7. Benford’s Law-Based Forensic Analysis

Benford’s Law describes the expected distribution of leading digits in many naturally occurring numerical datasets. Under the expected distribution, smaller leading digits occur more frequently than larger ones.

The probability of a non-zero leading digit (d) is:

P(d)=log10(1+(1/d))

where (d) ranges from 1 to 9.

In financial forensic analysis, deviations from an expected digit distribution can be used as an anomaly indicator. However, Benford’s Law does not apply equally well to every dataset, and deviations do not independently prove fraud.

### Implementation in ForensicFusion

The feature-engineering pipeline includes the following Benford-related features:

- `BenfordTypeScore`
- `BenfordTimeScore_25`
- `BenfordTypeScore_Norm`
- `BenfordTimeScore_Norm`
- `BenfordForensicScore`

These features represent statistical indicators derived from the project’s Benford-based analysis and are included in the engineered dataset.

The resulting forensic score provides additional statistical information for downstream analysis. It should be interpreted as an anomaly indicator rather than a definitive fraud probability.

## 8. Graph Construction

Financial transactions naturally form a network of relationships between accounts.

For the graph representation:

- **Nodes** represent financial accounts.
- **Edges** represent transactions between accounts.
- **Node features** describe account-level activity and behavior.
- **Edge labels** indicate whether the corresponding transaction is genuine or fraudulent.

The project maps account identifiers to numerical node indices so they can be represented efficiently in graph structures.

The prepared graph data contains approximately **9,073,900 unique account nodes**. Node features are stored in a normalized NumPy array, and transaction relationships are represented using edge-index arrays.

PyTorch Geometric is used to construct and process graph data for graph-learning experiments.

### Temporal data separation

The project uses a time-based split rather than randomly mixing transactions from different time periods.


Training - 4,463,587 transactions - 3,643 fraudulent transactions - 1–323 time steps
Validation - 980,416 transactions - 564  fraudulent transactions- 324–378 time steps
Testing - 918,617 transactions- 4,006 fraudulent transactions- 379–743 time steps 

The fraud proportion changes across these periods. Consequently, evaluation must consider temporal distribution changes and the possibility of different fraud patterns in the later test period.

A production-quality implementation should also ensure that each prediction uses only information available at the prediction time and that target labels are not inadvertently exposed through graph construction.

## 9. GraphSAGE-Based Fraud Detection

GraphSAGE is a Graph Neural Network architecture that learns node representations by aggregating information from neighboring nodes.

Unlike methods that treat every transaction independently, GraphSAGE can use the account network to learn representations that reflect local graph structure and neighboring account features.

### Model design

The implemented GraphSAGE model contains:

- A first GraphSAGE convolution layer that transforms the input node features into hidden representations.
- A second GraphSAGE convolution layer that further learns neighborhood information.
- A source-and-destination representation stage that combines the embeddings of the two accounts participating in a transaction.
- A classification network that produces a fraud logit for each transaction.

The documented model configuration uses 12 input node features and 64 hidden channels in the GraphSAGE layers.

The resulting transaction-level output is converted into a fraud probability for classification and evaluation.

### Implementation status

The project notebook includes graph preparation, a GraphSAGE model definition, neighbor-sampling experiments, and a successful forward-pass test. It also contains downstream test-prediction outputs and evaluation visualizations.

The model's evaluation results must be interpreted alongside the exact training run, prediction threshold, test labels, and graph-sampling configuration used to generate them.

## 10. Suspicious Account-Network Analysis

NetworkX is used to analyze and visualize relationships between accounts associated with suspicious transactions.

The network-analysis stage uses transaction predictions to identify high-risk connections and construct a smaller graph for investigation.

Such visualizations can help investigators examine:

- Which accounts are connected through suspicious transactions.
- Whether suspicious activity involves multiple linked accounts.
- Which relationships deserve further review.
- How potentially coordinated activity appears in the transaction network.

A suspicious network is an investigative lead, not proof that every account in the network is involved in fraud. The visualization should be interpreted alongside transaction evidence and model uncertainty.

## 11. Explainable AI Using SHAP

Explainable AI helps users understand which input features contribute to a model output.

The project uses **SHAP (SHapley Additive exPlanations)**, including `shap.KernelExplainer`, to explain selected outputs and inspect feature contributions.

SHAP estimates feature contributions by evaluating how model outputs change when different combinations of input features are considered.

### Purpose in ForensicFusion

SHAP is intended to help answer questions such as:

- Which input features influenced a selected prediction?
- Which features contribute most strongly across the analyzed examples?
- What evidence should an investigator inspect when reviewing a suspicious transaction?

## 12. Hybrid Forensic Risk Assessment

The motivation behind ForensicFusion is to combine two complementary forms of evidence:

1. **Statistical evidence:** Benford’s Law-based anomaly features.
2. **Relational evidence:** Graph-based account representations and transaction fraud predictions.


The results of these experiments should be evaluated carefully before the combined approach is presented as superior. In the reported ablation output, the full ForensicFusion configuration performs worse than the GNN-only configuration on both ROC-AUC and PR-AUC.

Therefore, the current evidence does not justify claiming that the hybrid model improves detection performance. Further investigation is needed to verify score scaling, feature alignment, model training, threshold selection, and fusion strategy.

## 13. Evaluation Metrics

Because the dataset is highly imbalanced, the project considers multiple evaluation measures.

### Precision

Precision measures the proportion of transactions predicted as fraudulent that are actually fraudulent.

Precision=TP/TP+FP

### Recall

Recall measures the proportion of actual fraudulent transactions correctly identified.

Recall=TP/TP+FN

### F1-score

The F1-score is the harmonic mean of precision and recall.

F1= (Precision*Recall)/(Precision+Recall)

### ROC-AUC

ROC-AUC measures the ability of a scoring model to distinguish positive and negative classes across classification thresholds.

### PR-AUC

PR-AUC summarizes the precision-recall trade-off and is especially informative when the positive class is rare.

### Confusion matrix

A confusion matrix reports true positives, true negatives, false positives, and false negatives. It helps reveal the types of classification errors that occur.

### Results

The following are the GraphSAGE results:


| Configuration | ROC-AUC | PR-AUC |
|---|---:|---:|
| GNN Only | 0.8351 | 0.5740 |
| GNN + Benford | 0.5226 | 0.2678 |
| GNN + Behavioral | 0.5696 | 0.2450 |
| Full ForensicFusion | 0.907182 | 0.558865 |


## 14. Technology Stack

The project uses the following technologies:

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Tabular data processing and feature engineering |
| NumPy | Numerical operations and array handling |
| PyTorch | Neural network implementation |
| PyTorch Geometric | Graph representation, GraphSAGE layers, and graph sampling |
| NetworkX | Account-network construction and visualization |
| SHAP | Feature-contribution explanations |
| Matplotlib | Evaluation plots and visualizations |
| Seaborn | Statistical visualization where used |
| Scikit-learn | Evaluation metrics and analysis utilities |
| Google Colab | Interactive development and experimentation |

## 15. Project Workflow

The end-to-end workflow can be summarized as follows:

```text
Financial Transaction Dataset
              |
              v
     Data Preprocessing
              |
              v
 Transaction Feature Engineering
              |
              v
 Benford's Law-Based Features
              |
              v
 Account-to-Node Mapping
              |
              v
   Graph Construction
              |
              v
 GraphSAGE Fraud Classification
              |
              v
 Transaction-Level Predictions
              |
       +------+------+
       |             |
       v             v
 NetworkX Analysis   SHAP Explanations
       |             |
       +------+------+
              |
              v
 Fraud Evaluation and Forensic Review
```

The hybrid-scoring component is an experimental stage and requires further validation before the complete system can be described as a validated integrated solution.

## 16. Future Enhancements

Potential future improvements include:

- Investigating why the combined feature configurations underperform the GNN-only baseline.
- Evaluating alternative classification thresholds to balance fraud detection and false alarms.
- Applying leakage-safe temporal graph evaluation.
- Comparing GraphSAGE with appropriate non-graph and graph-based baselines.
- Improving explanations for both transaction-level predictions and graph relationships.
- Adding a forensic dashboard for reviewing suspicious transactions and network connections.
- Improving reproducibility through configuration files, saved model checkpoints, and documented evaluation scripts.
- Testing the framework on additional financial datasets and different temporal periods.

These are proposed extensions rather than claims of completed implementation.

## 17. Conclusion

ForensicFusion investigates a hybrid approach to financial fraud detection by combining Benford’s Law-based numerical anomaly features, GraphSAGE-based graph learning, NetworkX-based account-network analysis, and SHAP explanations.

The project has developed a large feature-engineered transaction dataset, prepared account-level graph representations, implemented a GraphSAGE architecture, and produced prediction and evaluation outputs. It also explores how statistical and behavioral features can be combined with graph-based learning.

The current experiments highlight an important challenge: integrating complementary techniques does not automatically improve fraud detection. 
Overall, ForensicFusion provides a foundation for exploring explainable, graph-based financial forensic analysis.


