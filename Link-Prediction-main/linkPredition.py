import networkx as nx
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from PyQt5 import QtWidgets, QtGui
import sys

# Load Facebook dataset
G = nx.read_edgelist("facebook_combined.txt", nodetype=int)
print("Total nodes:", G.number_of_nodes())
print("Total edges:", G.number_of_edges())

# Function to generate dataset for ML
def generate_features(graph):
    data = []
    labels = []
    nodes = list(graph.nodes())

    for _ in range(5000):
        u, v = np.random.choice(nodes, 2, replace=False)
        label = 1 if graph.has_edge(u, v) else 0

        # Topological features
        common_neighbors = len(list(nx.common_neighbors(graph, u, v)))
        jaccard_coeff = list(nx.jaccard_coefficient(graph, [(u, v)]))[0][2]
        preferential_attachment = list(nx.preferential_attachment(graph, [(u, v)]))[0][2]

        # Non-topological (random example feature)
        deg_u = graph.degree(u)
        deg_v = graph.degree(v)

        features = [common_neighbors, jaccard_coeff, preferential_attachment, deg_u, deg_v]
        data.append(features)
        labels.append(label)

    return np.array(data), np.array(labels)

# Prepare dataset
X, y = generate_features(G)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ML Models
svm = SVC(probability=True)
knn = KNeighborsClassifier(n_neighbors=5)
nn = MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=500)

# Train models
svm.fit(X_train, y_train)
knn.fit(X_train, y_train)
nn.fit(X_train, y_train)

# Predictions
y_pred_svm = svm.predict(X_test)
y_pred_knn = knn.predict(X_test)
y_pred_nn = nn.predict(X_test)

# AUC
auc_svm = roc_auc_score(y_test, svm.predict_proba(X_test)[:, 1])
auc_knn = roc_auc_score(y_test, knn.predict_proba(X_test)[:, 1])
auc_nn = roc_auc_score(y_test, nn.predict_proba(X_test)[:, 1])

# Results Dictionary
results = {
    'SVM Accuracy': accuracy_score(y_test, y_pred_svm),
    'SVM AUC': auc_svm,
    'k-NN Accuracy': accuracy_score(y_test, y_pred_knn),
    'k-NN AUC': auc_knn,
    'Neural Net Accuracy': accuracy_score(y_test, y_pred_nn),
    'Neural Net AUC': auc_nn
}

print(results)

# PyQt5 GUI App
class ResultWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Link Prediction Results")
        self.setGeometry(200, 200, 500, 300)

        layout = QtWidgets.QVBoxLayout()

        title = QtWidgets.QLabel("📊 Link Prediction Model Performance")
        title.setFont(QtGui.QFont("Arial", 16))
        layout.addWidget(title)

        for k, v in results.items():
            label = QtWidgets.QLabel(f"{k}: {v:.4f}")
            label.setFont(QtGui.QFont("Arial", 12))
            layout.addWidget(label)

        graph_button = QtWidgets.QPushButton("Show Network Graph")
        graph_button.clicked.connect(self.show_graph)
        layout.addWidget(graph_button)

        self.setLayout(layout)

    def show_graph(self):
        plt.figure(figsize=(8, 6))
        nx.draw_spring(G, node_size=30, edge_color='gray')
        plt.title("Facebook Friendship Network")
        plt.show()

app = QtWidgets.QApplication(sys.argv)
window = ResultWindow()
window.show()
sys.exit(app.exec_())
