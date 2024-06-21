# A tool to make classification easy.
"""
1. Read a dataset into (partition, id, label, image(path), metadata) format.
2. Compute embeddings via a model, and save them (partition, id, label, image(path), embedding, metadata).
3. Analyze the clusters and visualize them. (TSNE, PCA, UMAP)
4. Implement multiple methods to classify the embeddings.
    a. KNN
"""