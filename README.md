Anomaly Detection in Computer Networks

Overview
This project detects anomalies in computer network traffic using Machine Learning.
Normal traffic simulates web browsing, file transfers, and regular connections.
Anomalous traffic simulates attacks like DDoS, port scans, and data exfiltration.
Both supervised (Random Forest) and unsupervised (Isolation Forest) approaches are compared.

Workflow
Synthetic Dataset Creation
10,000 normal records (TCP, UDP, ICMP).
10,000 anomalies (high packet rates, large packet sizes).
Added noise → some normal data looks anomalous and vice versa.
Preprocessing & Feature Engineering

Features used:
Packets_per_sec

Packet_Size

Connection_Duration
Normalized using StandardScaler.

Model Training

Isolation Forest (unsupervised): Detects anomalies without labels.
Random Forest (supervised): Learns from labeled data.

Evaluation Metrics
Accuracy, Precision, Recall, F1-Score.
Confusion Matrix.
ROC-AUC curves.
Feature Importance (Random Forest).

📊 Results

Isolation Forest:
Good recall (detects many anomalies).
Useful for zero-day attacks with no labels.

Random Forest:
Higher precision and overall accuracy.
Works well when labeled data is available.

Key Insight:

Packets_per_sec and Packet_Size are the strongest indicators of anomalous traffic.

Project Structure
📁 anomaly-detection-cn
│── network_traffic_data.csv              # Full dataset
│── network_traffic_train.csv             # Training data
│── network_traffic_test.csv              # Test data
│── network_traffic_with_all_predictions.csv
│── network_data_visualization.png        # Exploratory plots
│── model_confusion_matrices.png          # Confusion matrices
│── feature_importance.png                # Random Forest importance
│── roc_curves.png                        # ROC-AUC curves
│── anomaly_detection.py                  # Main script
│── README.md                             # Project documentation

Tech Stack

Language: Python
Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn
Models: Isolation Forest, Random Forest

✅ Conclusion
Random Forest → better when you have labeled datasets.
Isolation Forest → better for unknown/zero-day attacks without labels.
A hybrid approach can be used for real-world Intrusion Detection Systems (IDS).
