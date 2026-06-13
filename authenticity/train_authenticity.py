"""
Train Authenticity Scorer

The authenticity scorer is a binary classifier that distinguishes between
real data samples and generated/synthetic samples. It's used as a reward
signal in RL training to ensure generated samples look realistic.

Training strategy:
- Positive samples: Real data from training set
- Negative samples: Corrupted/augmented versions of real data
  (feature shuffling, value perturbation, partial denoising artifacts)
"""

import os
import json
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import joblib


class AuthenticityMLP(nn.Module):
    """MLP classifier for authenticity scoring."""

    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout=0.15):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def predict_proba(self, x_tensor):
        """Predict probability of being real."""
        with torch.no_grad():
            logits = self.forward(x_tensor)
            prob_real = torch.sigmoid(logits)
            # Return [P(fake), P(real)] format
            return torch.cat([1 - prob_real, prob_real], dim=-1)


class AuthenticityScorer:
    """Wrapper for authenticity scoring with sklearn-compatible interface."""

    def __init__(self, model_path, scaler_path, feature_columns, device='cpu'):
        self.device = device
        self.feature_columns = feature_columns

        # Load model
        checkpoint = torch.load(model_path, map_location=device)
        config = checkpoint.get('model_config', {})
        hidden_dims = config.get('hidden_dims', [256, 128, 64])

        self.model = AuthenticityMLP(
            input_dim=len(feature_columns),
            hidden_dims=hidden_dims
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()

        # Load scaler
        self.scaler = joblib.load(scaler_path)

    def predict_proba(self, sample_dict):
        """
        Predict authenticity probability for a sample.

        Args:
            sample_dict: Dictionary containing feature values

        Returns:
            numpy array of shape (1, 2): [P(fake), P(real)]
        """
        features = []
        for col in self.feature_columns:
            val = sample_dict.get(col, 0)
            if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
                val = 0
            try:
                val = float(val)
            except (ValueError, TypeError):
                val = 0
            features.append(val)

        X = np.array([features], dtype=np.float32)
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        proba = self.model.predict_proba(X_tensor)
        return proba.cpu().numpy()


def build_corrupted_samples(real_samples, feature_cols, target_col, num_per_type=1000):
    """Build negative samples through various corruption strategies."""
    corrupted = []

    for _ in range(num_per_type):
        # Strategy 1: Shuffle feature values across samples
        sample = random.choice(real_samples).copy()
        keys_to_shuffle = random.sample(feature_cols, min(3, len(feature_cols)))
        donor = random.choice(real_samples)
        for key in keys_to_shuffle:
            if key in donor:
                sample[key] = donor[key]
        corrupted.append(sample)

    for _ in range(num_per_type):
        # Strategy 2: Perturb numeric values
        sample = random.choice(real_samples).copy()
        for key in feature_cols:
            if isinstance(sample.get(key), (int, float)):
                noise = random.gauss(0, abs(sample[key]) * 0.5 + 1)
                sample[key] = sample[key] + noise
        corrupted.append(sample)

    for _ in range(num_per_type):
        # Strategy 3: Random value replacement
        sample = random.choice(real_samples).copy()
        num_replace = random.randint(1, len(feature_cols) // 2)
        keys_to_replace = random.sample(feature_cols, num_replace)
        for key in keys_to_replace:
            if isinstance(sample.get(key), str):
                sample[key] = random.choice(["unknown", "null", "", "N/A"])
            else:
                sample[key] = 0
        corrupted.append(sample)

    return corrupted


def extract_features(samples, feature_cols):
    """Extract feature matrix from samples."""
    features = []
    for sample in samples:
        row = []
        for col in feature_cols:
            val = sample.get(col, 0)
            if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
                val = 0
            try:
                val = float(val)
            except (ValueError, TypeError):
                val = 0
            row.append(val)
        features.append(row)
    return np.array(features, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Train authenticity scorer")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory with train.jsonl")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--feature_cols", type=str, default=None, help="Comma-separated feature columns")
    parser.add_argument("--target_col", type=str, default="income", help="Target column name")
    parser.add_argument("--hidden_dims", type=str, default="256,128,64", help="Hidden layer dimensions")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load real samples
    real_samples = []
    train_file = os.path.join(args.data_dir, "train.jsonl")
    with open(train_file, "r") as f:
        for line in f:
            item = json.loads(line)
            text = item.get("text", "{}")
            sample = json.loads(text) if isinstance(text, str) else text
            real_samples.append(sample)

    print(f"Loaded {len(real_samples)} real samples")

    # Determine feature columns
    if args.feature_cols:
        feature_cols = args.feature_cols.split(",")
    else:
        feature_cols = [k for k in real_samples[0].keys() if k != args.target_col]

    print(f"Feature columns: {feature_cols}")

    # Build corrupted samples
    corrupted_samples = build_corrupted_samples(real_samples, feature_cols, args.target_col)

    # Extract features
    X_real = extract_features(real_samples, feature_cols)
    X_corrupted = extract_features(corrupted_samples, feature_cols)

    # Combine and create labels
    X = np.vstack([X_real, X_corrupted])
    y = np.array([1] * len(X_real) + [0] * len(X_corrupted))

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42, stratify=y_train)

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Create dataloaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Create model
    hidden_dims = [int(x) for x in args.hidden_dims.split(",")]
    model = AuthenticityMLP(input_dim=len(feature_cols), hidden_dims=hidden_dims).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)

    # Training loop
    best_val_auc = 0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x).squeeze(-1)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * batch_x.size(0)

        # Evaluate
        model.eval()
        with torch.no_grad():
            val_logits = model(torch.FloatTensor(X_val).to(device)).squeeze(-1)
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
            val_preds = (val_probs >= 0.5).astype(int)

            val_auc = roc_auc_score(y_val, val_probs)
            val_acc = accuracy_score(y_val, val_preds)
            val_f1 = f1_score(y_val, val_preds)

        print(f"Epoch {epoch + 1}/{args.epochs} | Loss: {total_loss / len(X_train):.4f} | "
              f"Val AUC: {val_auc:.4f} | Val F1: {val_f1:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_config': {'hidden_dims': hidden_dims, 'feature_cols': feature_cols},
                'best_val_auc': best_val_auc,
            }, os.path.join(args.output_dir, 'best_model.pt'))

    # Save scaler
    joblib.dump(scaler, os.path.join(args.output_dir, 'scaler.pkl'))

    # Save config
    config = {
        'feature_columns': feature_cols,
        'target_col': args.target_col,
        'hidden_dims': hidden_dims,
        'best_val_auc': best_val_auc,
    }
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\nTraining complete! Best Val AUC: {best_val_auc:.4f}")
    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
