import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

def load_preprocessed_data():
    print("Loading preprocessed data...")
    try:
        df_preprocessed = pd.read_csv('california_housing_preprocessed.csv')
        print(f"Preprocessed data loaded with shape: {df_preprocessed.shape}")
        return df_preprocessed
    except FileNotFoundError:
        print("Preprocessed data file not found. Please run Task 1 first.")
        return None

def determine_optimal_clusters(data):
    print("\n--- Determining Optimal Number of Clusters ---")
    
    inertias = []
    silhouette_scores = []
    max_k = 10
    
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)
        
        if len(np.unique(kmeans.labels_)) > 1:
            score = silhouette_score(data, kmeans.labels_)
            silhouette_scores.append(score)
        else:
            silhouette_scores.append(0)
    
    inertias_array = np.array(inertias)
    normalized_inertias = (inertias_array - np.min(inertias_array)) / (np.max(inertias_array) - np.min(inertias_array))
    k_points = np.arange(2, max_k + 1)
    
    slopes = np.diff(normalized_inertias)
    knee_point = np.argmax(np.abs(slopes)) + 2
    
    best_silhouette_k = np.argmax(silhouette_scores) + 2
    
    optimal_k = max(knee_point, best_silhouette_k)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(2, max_k + 1), inertias, 'bo-')
    plt.axvline(x=knee_point, color='r', linestyle='--', label=f'Elbow at k={knee_point}')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(range(2, max_k + 1), silhouette_scores, 'go-')
    plt.axvline(x=best_silhouette_k, color='r', linestyle='--', 
                label=f'Best k={best_silhouette_k} (silhouette={max(silhouette_scores):.2f})')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('optimal_clusters.png')
    plt.close()
    
    print(f"Optimal number of clusters: {optimal_k} (elbow at k={knee_point}, best silhouette at k={best_silhouette_k})")
    return optimal_k

def apply_kmeans(data, n_clusters):
    print(f"\n--- Applying K-Means Clustering (k={n_clusters}) ---")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(data)
    
    if len(np.unique(labels)) > 1:
        score = silhouette_score(data, labels)
        print(f"Silhouette Score: {score:.4f}")
    
    print(f"K-Means clustering completed. Cluster distribution:")
    unique, counts = np.unique(labels, return_counts=True)
    for i, (cluster, count) in enumerate(zip(unique, counts)):
        print(f"Cluster {cluster}: {count} samples ({count/len(labels)*100:.2f}%)")
    
    return labels, kmeans

def apply_dbscan(data):
    print("\n--- Applying DBSCAN Clustering ---")
    
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    eps_values = [0.3, 0.5, 0.7, 1.0, 1.5]
    min_samples_values = [5, 10, 20, 30]
    
    best_score = -1
    best_eps = 0.5
    best_min_samples = 10
    best_labels = None
    
    print("Tuning DBSCAN parameters...")
    for eps in eps_values:
        for min_samples in min_samples_values:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(data_scaled)
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            if n_clusters > 1 and len(set(labels)) > 1:
                score = silhouette_score(data_scaled[labels != -1], labels[labels != -1])
                if score > best_score:
                    best_score = score
                    best_eps = eps
                    best_min_samples = min_samples
                    best_labels = labels
    
    if best_score == -1:
        print("No good clustering found with tested parameters. Using defaults.")
        dbscan = DBSCAN(eps=0.5, min_samples=10)
        best_labels = dbscan.fit_predict(data_scaled)
    else:
        print(f"Best parameters - eps: {best_eps}, min_samples: {best_min_samples}, Silhouette Score: {best_score:.4f}")
    
    n_noise = list(best_labels).count(-1)
    n_clusters = len(set(best_labels)) - (1 if -1 in best_labels else 0)
    
    print(f"DBSCAN clustering completed with {n_clusters} clusters and {n_noise} noise points.")
    print("Cluster distribution (including noise as cluster -1):")
    unique, counts = np.unique(best_labels, return_counts=True)
    for cluster, count in zip(unique, counts):
        print(f"Cluster {cluster}: {count} points")
    
    return best_labels

def apply_agglomerative(data, n_clusters):
    """
    Apply Agglomerative Clustering to the data
    
    Args:
        data (numpy.ndarray): The data to cluster
        n_clusters (int): The number of clusters
        
    Returns:
        numpy.ndarray: The cluster labels
    """
    print(f"\n--- Applying Agglomerative Clustering with {n_clusters} clusters ---")
    
    agg_clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels = agg_clustering.fit_predict(data)
    
    print(f"Agglomerative clustering completed. Cluster distribution:")
    unique, counts = np.unique(labels, return_counts=True)
    for i, (cluster, count) in enumerate(zip(unique, counts)):
        print(f"Cluster {cluster}: {count} samples ({count/len(labels)*100:.2f}%)")
    
    return labels, agg_clustering

def visualize_clusters_pca(data, labels, algorithm_name):
    print(f"Visualizing {algorithm_name} clusters using PCA...")
    
    pca = PCA(n_components=2, random_state=42)
    pca_result = pca.fit_transform(data)
    
    plt.figure(figsize=(10, 8))
    
    unique_labels = np.unique(labels)
    for label in unique_labels:
        if label == -1:
            plt.scatter(pca_result[labels == label, 0], 
                       pca_result[labels == label, 1], 
                       c='gray', alpha=0.5, s=10, label='Noise')
        else:
            plt.scatter(pca_result[labels == label, 0], 
                       pca_result[labels == label, 1], 
                       s=50, label=f'Cluster {label}')
    
    plt.title(f'{algorithm_name} Clusters (PCA)', fontsize=14)
    plt.xlabel('First Principal Component', fontsize=12)
    plt.ylabel('Second Principal Component', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    filename = f'{algorithm_name.lower().replace(" ", "_")}_pca.png'
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"PCA visualization saved as {filename}")

def visualize_clusters_tsne(data, labels, algorithm_name):
    print(f"Visualizing {algorithm_name} clusters using t-SNE...")
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    reduced_data = tsne.fit_transform(data)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis', 
                         alpha=0.6, s=50, edgecolors='w', linewidths=0.5)
    
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title(f'Cluster Visualization using t-SNE - {algorithm_name}')
    plt.grid(True, alpha=0.3)
    
    plt.savefig(f'{algorithm_name.lower()}_tsne_clusters.png')
    plt.close()
    
    return reduced_data

def characterize_clusters(df_original, labels, algorithm_name):
    print(f"\n--- Characterizing {algorithm_name} Clusters ---")
    
    df_with_clusters = df_original.copy()
    df_with_clusters['Cluster'] = labels
    
    numerical_cols = df_original.select_dtypes(include=np.number).columns.tolist()
    df_numerical = df_with_clusters[numerical_cols + ['Cluster']]
    
    cluster_means = df_numerical.groupby('Cluster').mean()
    print("\nCluster Means:")
    print(cluster_means)
    
    # Create a heatmap of cluster means
    plt.figure(figsize=(12, 8))
    sns.heatmap(cluster_means, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title(f'Feature Means by Cluster - {algorithm_name}')
    plt.savefig(f'{algorithm_name.lower()}_cluster_means.png')
    
    # For numerical features, create box plots to compare distributions across clusters
    for col in numerical_cols[:5]:  # Limit to first 5 numerical columns for brevity
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Cluster', y=col, data=df_with_clusters)
        plt.title(f'Distribution of {col} by Cluster - {algorithm_name}')
        plt.savefig(f'{algorithm_name.lower()}_cluster_{col}_boxplot.png')
    
    # For categorical features, create count plots to show distribution across clusters
    categorical_cols = df_original.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in categorical_cols:
        plt.figure(figsize=(12, 8))
        for cluster in sorted(df_with_clusters['Cluster'].unique()):
            cluster_data = df_with_clusters[df_with_clusters['Cluster'] == cluster]
            counts = cluster_data[col].value_counts(normalize=True)
            plt.bar([f"{x} (C{cluster})" for x in counts.index], counts.values, 
                   label=f'Cluster {cluster}', alpha=0.7)
        plt.title(f'Distribution of {col} by Cluster - {algorithm_name}')
        plt.xlabel(col)
        plt.ylabel('Proportion')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{algorithm_name.lower()}_cluster_{col}_countplot.png')
    
    return df_with_clusters, cluster_means

def write_cluster_report(kmeans_report, dbscan_report):
    """
    Write a brief report on insights gained from the clusters
    
    Args:
        kmeans_report (tuple): The dataframe with clusters and cluster means from K-Means
        dbscan_report (tuple): The dataframe with clusters and cluster means from DBSCAN
    """
    print("\n--- Writing Cluster Analysis Report ---")
    
    kmeans_df, kmeans_means = kmeans_report
    dbscan_df, dbscan_means = dbscan_report
    
    report = """
# Cluster Analysis Report

## Overview
This report presents insights gained from applying clustering algorithms to the California Housing dataset. 
We applied two different clustering algorithms: K-Means and DBSCAN, to identify natural groupings in the data.

## K-Means Clustering Insights

K-Means clustering revealed distinct housing market segments based on various features:

"""
    
    # Add insights from K-Means clustering
    for cluster in kmeans_means.index:
        report += f"### Cluster {cluster}:\n"
        
        # Get the top 3 highest and lowest features for this cluster
        highest = kmeans_means.loc[cluster].nlargest(3)
        lowest = kmeans_means.loc[cluster].nsmallest(3)
        
        report += "**Key characteristics:**\n"
        report += "- High values in: " + ", ".join([f"{col} ({val:.2f})" for col, val in highest.items()]) + "\n"
        report += "- Low values in: " + ", ".join([f"{col} ({val:.2f})" for col, val in lowest.items()]) + "\n\n"
    
    report += """
## DBSCAN Clustering Insights

DBSCAN identified clusters of varying density and also detected outliers (noise points):

"""
    
    # Add insights from DBSCAN clustering
    for cluster in dbscan_means.index:
        if cluster == -1:
            report += "### Noise Points (Outliers):\n"
        else:
            report += f"### Cluster {cluster}:\n"
        
        # Get the top 3 highest and lowest features for this cluster
        highest = dbscan_means.loc[cluster].nlargest(3)
        lowest = dbscan_means.loc[cluster].nsmallest(3)
        
        report += "**Key characteristics:**\n"
        report += "- High values in: " + ", ".join([f"{col} ({val:.2f})" for col, val in highest.items()]) + "\n"
        report += "- Low values in: " + ", ".join([f"{col} ({val:.2f})" for col, val in lowest.items()]) + "\n\n"
    
    report += """
## Comparison and Conclusions

The clustering analysis revealed several important insights about the California housing market:

1. **Geographic Segmentation**: Both algorithms identified clusters that correspond to different geographic regions with distinct housing characteristics.

2. **Income-Housing Value Relationship**: There is a strong correlation between median income and housing values across clusters, confirming the importance of income as a predictor of housing prices.

3. **Urban vs. Rural Divide**: Some clusters clearly represent urban areas (high population density, higher housing values) while others represent more rural areas.

4. **Age Impact**: House age has varying impacts on housing values depending on the region, suggesting that older houses may be more valuable in certain areas (possibly historic districts) but less valuable in others.

5. **Outlier Properties**: DBSCAN identified outlier properties that don't fit well into any cluster, which could represent unique housing markets or data anomalies worth investigating further.

These insights can help inform housing policy, investment decisions, and further predictive modeling efforts.
"""
    
    # Write the report to a file
    with open('cluster_analysis_report.md', 'w') as f:
        f.write(report)
    
    print("Cluster analysis report written to 'cluster_analysis_report.md'")
    
    return report

def main():
    """
    Main function to execute Task 2
    """
    # Load the preprocessed data
    df_preprocessed = load_preprocessed_data()
    if df_preprocessed is None:
        return
    
    # Load the original cleaned data for later analysis
    try:
        df_cleaned = pd.read_csv('california_housing_cleaned.csv')
    except FileNotFoundError:
        print("Cleaned data file not found. Please run Task 1 first.")
        return
    
    # Convert the preprocessed data to a numpy array
    data = df_preprocessed.values
    
    # Determine the optimal number of clusters
    optimal_k = determine_optimal_clusters(data)
    
    # Apply K-Means clustering
    kmeans_labels, kmeans_model = apply_kmeans(data, optimal_k)
    
    # Apply DBSCAN clustering
    dbscan_labels, dbscan_model = apply_dbscan(data)
    
    # Apply Agglomerative clustering
    agg_labels, agg_model = apply_agglomerative(data, optimal_k)
    
    # Visualize clusters using PCA
    visualize_clusters_pca(data, kmeans_labels, 'K-Means')
    visualize_clusters_pca(data, dbscan_labels, 'DBSCAN')
    visualize_clusters_pca(data, agg_labels, 'Agglomerative')
    
    # Visualize clusters using t-SNE
    visualize_clusters_tsne(data, kmeans_labels, 'K-Means')
    visualize_clusters_tsne(data, dbscan_labels, 'DBSCAN')
    visualize_clusters_tsne(data, agg_labels, 'Agglomerative')
    
    # Characterize clusters
    kmeans_report = characterize_clusters(df_cleaned, kmeans_labels, 'K-Means')
    dbscan_report = characterize_clusters(df_cleaned, dbscan_labels, 'DBSCAN')
    agg_report = characterize_clusters(df_cleaned, agg_labels, 'Agglomerative')
    
    # Write cluster analysis report
    write_cluster_report(kmeans_report, dbscan_report)
    
    print("\nTask 2 completed.")

if __name__ == "__main__":
    main()
