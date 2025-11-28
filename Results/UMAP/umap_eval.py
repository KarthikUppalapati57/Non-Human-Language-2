import numpy as np
import pandas as pd
import umap
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
import matplotlib.pyplot as plt
import seaborn as sns


CHECKPOINT_FILE = "extracted_features_checkpoint.npz"
SAVE_METRICS_CSV = "umap_clustering_metrics.csv"
SAVE_FIG = "umap_map.png"


def load_checkpoint(path):
    print(f"Loading checkpoint from: {path}")
    data = np.load(path, allow_pickle=True)

    emb_raw = data["embeddings"]
    y_act = data["activities"]
    y_voc = data["vocalizations"]


    if emb_raw.ndim == 1:
        X = np.vstack(emb_raw)
    else:
        X = emb_raw

    print("Embeddings shape:", X.shape)
    print("Num activity labels:", len(y_act))
    print("Num vocalization labels:", len(y_voc))
    return X, np.array(y_act), np.array(y_voc)


def run_umap(X, n_neighbors=15, min_dist=0.1, random_state=42):
    print("Running UMAP...")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="euclidean",
        random_state=random_state,
    )
    X_emb = reducer.fit_transform(X)
    print("UMAP embedding shape:", X_emb.shape)
    return X_emb


def compute_metrics(X_emb, labels, label_name):
    unique = np.unique(labels)
    if len(unique) < 2:
        print(f"[{label_name}] Not enough classes for metrics.")
        return None

    sil = silhouette_score(X_emb, labels)
    dbi = davies_bouldin_score(X_emb, labels)
    chi = calinski_harabasz_score(X_emb, labels)

    print(f"\n[{label_name}] clustering metrics:")
    print(f"  Silhouette Score       : {sil:.4f}")
    print(f"  Davies–Bouldin Index   : {dbi:.4f}  (lower is better)")
    print(f"  Calinski–Harabasz Score: {chi:.4f}  (higher is better)")

    return {
        "LabelType": label_name,
        "Silhouette": sil,
        "DaviesBouldin": dbi,
        "CalinskiHarabasz": chi,
        "n_samples": len(labels),
        "n_classes": len(unique),
    }


def plot_umap(X_emb, y_act, y_voc, save_path):
    print("\nSaving UMAP plot...")
    df = pd.DataFrame(
        {
            "x": X_emb[:, 0],
            "y": X_emb[:, 1],
            "Activity": y_act,
            "Vocalization": y_voc,
        }
    )

    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        data=df,
        x="x",
        y="y",
        hue="Activity",
        style="Vocalization",
        s=40,
        alpha=0.7,
    )
    plt.title("UMAP Embedding – Activity (color) + Vocalization (marker)")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"UMAP figure saved to: {save_path}")


def main():
    X, y_act, y_voc = load_checkpoint(CHECKPOINT_FILE)
    X_emb = run_umap(X)

    metrics = []
    m1 = compute_metrics(X_emb, y_act, "Activity")
    if m1 is not None:
        metrics.append(m1)
    m2 = compute_metrics(X_emb, y_voc, "Vocalization")
    if m2 is not None:
        metrics.append(m2)

    if metrics:
        df_metrics = pd.DataFrame(metrics)
        df_metrics.to_csv(SAVE_METRICS_CSV, index=False)
        print(f"\nMetrics saved to: {SAVE_METRICS_CSV}")

    plot_umap(X_emb, y_act, y_voc, SAVE_FIG)

if __name__ == "__main__":
    main()
