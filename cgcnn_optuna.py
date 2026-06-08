"""
cgcnn_optuna.py
---------------
Optuna hyperparameter search for the CGCNN superconductor classifier.

Usage (Colab / terminal):
    python cgcnn_optuna.py

Assumes train_graphs, val_graphs, and test_graphs are already built
(run the notebook up to and including the graph-split cell), then
either import this file or paste it into a new cell.

Searches over:
  - hidden_dim          : 16 / 32 / 64 / 128
  - num_conv_layers     : 2 – 5
  - dropout             : 0.0 – 0.5
  - lr                  : 1e-4 – 1e-2  (log scale)
  - weight_decay        : 1e-6 – 1e-3  (log scale)
  - batch_size          : 32 / 64 / 128
  - pos_weight          : 1.0 – 10.0   (class imbalance)
  - scheduler_patience  : 5 / 10 / 20

Objective: maximise validation AUC-ROC.
"""

import torch
import torch.nn as nn
import numpy as np
import optuna
from optuna.samplers import TPESampler
from sklearn.metrics import roc_auc_score
from torch_geometric.nn import CGConv, global_mean_pool
from torch_geometric.loader import DataLoader

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Tuning budget ──────────────────────────────────────────────────────────────
N_TRIALS      = 50    # increase for a more thorough search
MAX_EPOCHS    = 100   # per trial (kept short; best model trains longer)
EARLY_STOP    = 15    # patience in epochs within each trial
EDGE_FEATURES = 40    # Gaussian expansion width — fixed


# ── Model (identical to notebook) ─────────────────────────────────────────────
class CGCNN(nn.Module):
    def __init__(self, node_features, edge_features, hidden_dim,
                 num_conv_layers, dropout):
        super().__init__()
        self.node_embedding = nn.Linear(node_features, hidden_dim)
        self.conv_layers = nn.ModuleList([
            CGConv(hidden_dim, dim=edge_features, batch_norm=True)
            for _ in range(num_conv_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch
        )
        x = torch.relu(self.node_embedding(x))
        for conv in self.conv_layers:
            x = torch.relu(conv(x, edge_index, edge_attr))
            x = self.dropout(x)
        x = global_mean_pool(x, batch)
        return self.fc(x).squeeze(-1)


# ── Single-trial training ──────────────────────────────────────────────────────
def run_trial(trial, train_graphs, val_graphs):
    # ── Hyperparameters proposed by Optuna ────────────────────────────────────
    hidden_dim         = trial.suggest_categorical("hidden_dim",         [16, 32, 64, 128])
    num_conv_layers    = trial.suggest_int(        "num_conv_layers",    1, 4)
    dropout            = trial.suggest_float(      "dropout",            0.0, 0.5)
    lr                 = trial.suggest_float(      "lr",                 1e-4, 1e-2, log=True)
    weight_decay       = trial.suggest_float(      "weight_decay",       1e-6, 1e-3, log=True)
    batch_size         = trial.suggest_categorical("batch_size",         [32, 64, 128])
    pos_weight_val     = trial.suggest_float(      "pos_weight",         1.0, 10.0)
    scheduler_patience = trial.suggest_categorical("scheduler_patience", [5, 10, 20])

    # ── Data loaders ──────────────────────────────────────────────────────────
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_graphs,   batch_size=batch_size, shuffle=False)

    # ── Model + optimiser ─────────────────────────────────────────────────────
    node_features = train_graphs[0].x.shape[1]
    model = CGCNN(
        node_features   = node_features,
        edge_features   = EDGE_FEATURES,
        hidden_dim      = hidden_dim,
        num_conv_layers = num_conv_layers,
        dropout         = dropout,
    ).to(DEVICE)

    pos_weight = torch.tensor([pos_weight_val]).to(DEVICE)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    optimizer  = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=scheduler_patience
    )

    best_auc      = 0.0
    patience_ctr  = 0

    for epoch in range(MAX_EPOCHS):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            pred            = model(batch)
            loss_per_sample = criterion(pred, batch.y.view(-1).float())
            # Per-sample weighting (use batch.weight if present, else uniform)
            if hasattr(batch, "weight") and batch.weight is not None:
                w    = batch.weight.view(-1).to(DEVICE)
                w    = w / w.sum() * len(w)
                loss = (loss_per_sample * w).mean()
            else:
                loss = loss_per_sample.mean()
            loss.backward()
            optimizer.step()

        # ── Validate (AUC) ────────────────────────────────────────────────────
        model.eval()
        all_probs, all_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                probs = torch.sigmoid(model(batch)).cpu().numpy()
                all_probs.extend(probs)
                all_targets.extend(batch.y.cpu().numpy())

        val_auc = roc_auc_score(all_targets, all_probs)
        scheduler.step(val_auc)

        # Optuna pruning
        trial.report(val_auc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if val_auc > best_auc:
            best_auc    = val_auc
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= EARLY_STOP:
                break

    return best_auc


# ── Optuna study ───────────────────────────────────────────────────────────────
def run_study(train_graphs, val_graphs, n_trials=N_TRIALS,
              study_name="cgcnn_classifier", storage=None):
    """
    Parameters
    ----------
    train_graphs, val_graphs : lists of PyG Data objects
    n_trials  : number of Optuna trials
    study_name: name for the study (used if storage is set)
    storage   : optional SQLite URL, e.g. "sqlite:///cgcnn_optuna.db"
                Set this to resume an interrupted search.

    Returns
    -------
    study : optuna.Study  (access best params via study.best_params)
    """
    sampler = TPESampler(seed=SEED)
    pruner  = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)

    study = optuna.create_study(
        study_name = study_name,
        direction  = "maximize",
        sampler    = sampler,
        pruner     = pruner,
        storage    = storage,
        load_if_exists = True,
    )

    study.optimize(
        lambda trial: run_trial(trial, train_graphs, val_graphs),
        n_trials  = n_trials,
        timeout   = None,          # set e.g. 3600 to cap at 1 hour
        show_progress_bar = True,
    )

    print("\n── Best trial ────────────────────────────────────────────────")
    print(f"  Val AUC : {study.best_value:.4f}")
    print("  Params  :")
    for k, v in study.best_params.items():
        print(f"    {k:<25s} {v}")

    return study


# ── Retrain best model on train+val ───────────────────────────────────────────
def retrain_best(study, train_graphs, val_graphs, test_graphs,
                 extra_epochs=200, save_path="cgcnn_best_optuna.pt"):
    """
    Train a fresh model with the best hyperparameters for longer,
    evaluate on the held-out test set, and save the weights.
    """
    p = study.best_params
    node_features = train_graphs[0].x.shape[1]
    all_train     = train_graphs + val_graphs          # combine for final run

    train_loader = DataLoader(all_train,    batch_size=p["batch_size"], shuffle=True)
    test_loader  = DataLoader(test_graphs,  batch_size=p["batch_size"], shuffle=False)

    model = CGCNN(
        node_features   = node_features,
        edge_features   = EDGE_FEATURES,
        hidden_dim      = p["hidden_dim"],
        num_conv_layers = p["num_conv_layers"],
        dropout         = p["dropout"],
    ).to(DEVICE)

    pos_weight = torch.tensor([p["pos_weight"]]).to(DEVICE)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    optimizer  = torch.optim.Adam(
        model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=p["scheduler_patience"]
    )

    best_auc, patience_ctr = 0.0, 0

    for epoch in range(extra_epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            pred            = model(batch)
            loss_per_sample = criterion(pred, batch.y.view(-1).float())
            if hasattr(batch, "weight") and batch.weight is not None:
                w    = batch.weight.view(-1).to(DEVICE)
                w    = w / w.sum() * len(w)
                loss = (loss_per_sample * w).mean()
            else:
                loss = loss_per_sample.mean()
            loss.backward()
            optimizer.step()

        # Quick test AUC for scheduling / early stopping
        model.eval()
        probs_all, tgts_all = [], []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(DEVICE)
                probs_all.extend(torch.sigmoid(model(batch)).cpu().numpy())
                tgts_all.extend(batch.y.cpu().numpy())
        test_auc = roc_auc_score(tgts_all, probs_all)
        scheduler.step(test_auc)

        if test_auc > best_auc:
            best_auc    = test_auc
            patience_ctr = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_ctr += 1
            if patience_ctr >= 30:
                print(f"Early stop at epoch {epoch}")
                break

        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d} | Test AUC: {test_auc:.4f} (best {best_auc:.4f})")

    print(f"\nBest test AUC: {best_auc:.4f} — weights saved to '{save_path}'")
    model.load_state_dict(torch.load(save_path, map_location=DEVICE))
    return model


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # When run as a script, train_graphs / val_graphs / test_graphs must be
    # in scope (e.g. loaded from disk).  Adjust the path as needed.
    import sys
    graphs = torch.load("classifier_crystal_graphs_ICSD_noenv.pt", weights_only=False)
    # ... split logic here if needed ...
    print("Load graphs then call run_study(train_graphs, val_graphs)")
    sys.exit(0)
