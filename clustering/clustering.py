from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score


def eval_model(y_true, y_pred):
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")

    print("Métricas de evaluación:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

    return f1


def show_dendrogram(Z, cutting_threshold):
    # Plotting the dendrogram
    plt.figure(figsize=(20, 7))
    dendrogram(Z, no_labels=True, orientation='top', distance_sort='descending', show_leaf_counts=True,
               color_threshold=cutting_threshold)
    #plt.title('Hierarchical Clustering Dendrogram')
    plt.ylabel('Distance')
    plt.axhline(y=cutting_threshold, color='red', linestyle='--', linewidth=2, label='cutting threshold')
    plt.legend(loc='upper left')
    plt.show()


def obtain_clusters(Z, cutting_threshold):
    clusters = fcluster(Z, t=cutting_threshold, criterion='distance')
    num_clusters = len(np.unique(clusters))

    print("Cutting threshold:", cutting_threshold, "\nNúmero de Clusters:", num_clusters)
    # print("Asignación de Cluster para cada observación:")
    # print(clusters)
    return clusters


def get_clustter_distance(X, distance, method='single', metric='mahalanobis'):
    print("Metric:", metric, "\nMethod:", method)
    # Perform hierarchical clustering
    Z_test = linkage(X, method=method, metric=metric)

    clusters = obtain_clusters(Z_test, distance)

    show_dendrogram(Z_test, distance)

    return clusters


def show_clusters(X, labels):
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(X)
    print(reduced_features.shape)

    unique_labels = np.unique(labels)
    for label in unique_labels:
        cluster_points = reduced_features[labels == label]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {label}')
    #plt.title('Clustering with PCA')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.show()


def interpret_cluster(X, y, test_size=0.3, seed=8):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    tree_classifier = DecisionTreeClassifier(random_state=seed, max_depth=6)

    tree_classifier.fit(X_train, y_train)

    y_pred = tree_classifier.predict(X_test)

    eval_model(y_test, y_pred)

    plt.figure(figsize=(40, 15))
    features_names = list(X.columns)
    clases_names = list(np.unique(y).astype(str))
    plot_tree(tree_classifier, filled=True, feature_names=features_names, class_names=clases_names, rounded=True)
    plt.show()
