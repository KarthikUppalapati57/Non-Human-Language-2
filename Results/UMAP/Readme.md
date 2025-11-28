This folder contains the UMAP-based clustering analysis of the HuBERT embeddings extracted from dolphin audio segments.
To evaluate how well the embedding space separates:

Activity labels (FFR, PLAY, ORD, NIGHT, unknown)

Vocalization labels (W, MW, FB, BPS, ECT, NOISE)

Input File:extracted_features_checkpoint.npz 

Results:
Activity-Level Structure (colors)
ORD (Red)
Forms a large, compact, well-defined region on the bottom right.
Indicates consistent, structured acoustic patterns during “Ordering” commands/training.

FFR (Orange)
Forms a dense, cohesive cluster beside ORD.
Good separability, reflecting the characteristic bursts of play-fight interactions.

NIGHT (Blue)
Occupies ~60% of the entire UMAP space, very spread out.
NIGHT recordings contain mixed events (whistles, clicks, noise), so they naturally scatter.

PLAY (Purple) & unknown (Green)
Small, partially overlapping blobs.
Low sample count and acoustically ambiguous behaviors lead to weaker structure.


Vocalization-Level Structure (markers)

Markers = specific vocalization types:
W (whistles) = X markers
MW (multi-loop whistles) = squares
FB (feeding buzzes) = + markers
ECT (echolocation clicks) = dots
BPS (burst pulses) = triangles
NOISE = circles

These markers do not form separate clusters.
All vocalization types appear scattered everywhere, even inside ORD, FFR, and NIGHT.
Frequent overlap of whistles, multi-loop whistles, and buzzes confirms acoustic similarity.

Numerical Analysis:
The numerical clustering metrics confirm the visual patterns observed in the UMAP projection. Activity labels show modest but meaningful structure in the HuBERT embedding space, with a Silhouette score of 0.21 indicating weak–moderate separability and a high Calinski–Harabasz score of 5542 reflecting substantial between-class dispersion driven mainly by the well-formed ORD and FFR regions. The Davies–Bouldin index of 3.14 suggests some overlap among activities, particularly in the PLAY and unknown categories, which aligns with their visual intermixing. In contrast, vocalization labels exhibit no cluster structure, with a near-zero Silhouette score (–0.007) and a very high Davies–Bouldin index (5.15) indicating severe overlap between whistle types, buzzes, burst-pulses, and echolocation clicks. The lower Calinski–Harabasz score (1806) further confirms weak global separability. Together, these metrics show that HuBERT embeddings capture behavioral distinctions far better than fine-grained vocalization types, reinforcing the need for the downstream LLM reasoning layer to interpret ambiguous acoustic predictions.



