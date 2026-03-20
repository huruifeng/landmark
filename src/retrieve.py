"""
Image retrieval: build a database index from training images and
retrieve similar images for validation queries.

Usage:
    python src/retrieve.py --data_dir data/gldv2_micro --checkpoint checkpoints/best_model.pth --top_k 5
"""

import os
import argparse
import numpy as np
import torch
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

from dataset import LandmarkDataset, get_val_transform
from model import LandmarkRetrievalModel


@torch.no_grad()
def extract_embeddings(model, dataset, batch_size, device):
    """Extract embeddings for all images in a dataset."""
    model.eval()
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    all_embeddings = []
    all_labels = []
    all_filenames = []

    for images, labels, filenames in tqdm(loader, desc="Extracting embeddings"):
        images = images.to(device)
        emb = model.extract_embedding(images)
        all_embeddings.append(emb.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_filenames.extend(filenames)

    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.array(all_labels)
    return embeddings, labels, all_filenames


def compute_map(query_labels, retrieved_labels):
    """Compute mean Average Precision (mAP) for retrieval results."""
    aps = []
    for i in range(len(query_labels)):
        q_label = query_labels[i]
        r_labels = retrieved_labels[i]

        # Binary relevance
        relevant = (r_labels == q_label).astype(float)
        if relevant.sum() == 0:
            aps.append(0.0)
            continue

        # Compute AP
        cumsum = np.cumsum(relevant)
        precision_at_k = cumsum / np.arange(1, len(relevant) + 1)
        ap = (precision_at_k * relevant).sum() / relevant.sum()
        aps.append(ap)

    return np.mean(aps)


def main():
    parser = argparse.ArgumentParser(description="Landmark image retrieval")
    parser.add_argument("--data_dir", type=str, default="data/gldv2_micro")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--image_size", type=int, default=224)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    num_classes = checkpoint["num_classes"]
    embedding_dim = checkpoint["embedding_dim"]
    print(f"Loaded model: embedding_dim={embedding_dim}, num_classes={num_classes}")

    # Build model and load weights
    model = LandmarkRetrievalModel(num_classes, embedding_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    image_dir = os.path.join(args.data_dir, "images")
    transform = get_val_transform(args.image_size)

    # Build database index from training images
    print("\n--- Building database index from training images ---")
    train_dataset = LandmarkDataset(
        os.path.join(args.data_dir, "train.csv"), image_dir, transform
    )
    db_embeddings, db_labels, db_filenames = extract_embeddings(
        model, train_dataset, args.batch_size, device
    )

    # Extract query embeddings from validation images
    print("\n--- Extracting query embeddings from validation images ---")
    val_dataset = LandmarkDataset(
        os.path.join(args.data_dir, "val.csv"), image_dir, transform
    )
    query_embeddings, query_labels, query_filenames = extract_embeddings(
        model, val_dataset, args.batch_size, device
    )

    # Nearest neighbor search
    print(f"\n--- Retrieving top-{args.top_k} results ---")
    nn_index = NearestNeighbors(n_neighbors=args.top_k, metric="cosine", algorithm="brute")
    nn_index.fit(db_embeddings)
    distances, indices = nn_index.kneighbors(query_embeddings)

    # Retrieve labels for the top-k results
    retrieved_labels = db_labels[indices]

    # Compute mAP
    mAP = compute_map(query_labels, retrieved_labels)
    print(f"\nmAP@{args.top_k}: {mAP:.4f}")

    # Per-query accuracy (at least one correct in top-k)
    top_k_hits = np.any(retrieved_labels == query_labels[:, None], axis=1)
    recall_at_k = top_k_hits.mean()
    print(f"Recall@{args.top_k}: {recall_at_k:.4f}")

    # Show a few example retrievals
    print(f"\n--- Example retrievals (top {args.top_k}) ---")
    for i in range(min(10, len(query_filenames))):
        q_file = query_filenames[i]
        q_label = query_labels[i]
        r_files = [db_filenames[j] for j in indices[i]]
        r_labels = retrieved_labels[i]
        match_str = ["Y" if rl == q_label else "N" for rl in r_labels]
        print(f"  Query: {q_file} (class {q_label})")
        for j, (rf, ms) in enumerate(zip(r_files, match_str)):
            dist = distances[i][j]
            print(f"    [{ms}] {rf} (dist={dist:.4f})")
        print()

    # Save results
    results_path = os.path.join(args.data_dir, "retrieval_results.csv")
    with open(results_path, "w") as f:
        f.write("query_filename,query_label,retrieved_filenames,retrieved_labels,distances\n")
        for i in range(len(query_filenames)):
            r_files = " ".join(db_filenames[j] for j in indices[i])
            r_labs = " ".join(str(l) for l in retrieved_labels[i])
            dists = " ".join(f"{d:.6f}" for d in distances[i])
            f.write(f"{query_filenames[i]},{query_labels[i]},{r_files},{r_labs},{dists}\n")
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
