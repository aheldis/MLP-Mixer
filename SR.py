# pip install -U timm torch torchvision
import os
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms
import timm
from timm.data import resolve_model_data_config, create_transform

# -------------------------------------------------------
# CLI  — pass your checkpoint with:
#   python eval_sr_defense.py --checkpoint /path/to/your_model.pt
# -------------------------------------------------------
_parser = argparse.ArgumentParser(description="SR defense evaluation")
_parser.add_argument("--checkpoint", type=str, default=None,
                     help="Path to your .pt checkpoint (state_dict of base model). "
                          "Overrides CHECKPOINT_PATH in the config below.")
_args, _ = _parser.parse_known_args()

# -------------------------------------------------------
# Config
# -------------------------------------------------------
MODEL_NAME      = "mixer_b16_224.goog_in21k_ft_in1k"
NUM_CLASSES     = 10
BATCH_SIZE      = 16        # SR level 3 = 49x batch — keep this small
NUM_WORKERS     = 0
PIN_MEMORY      = False
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = _args.checkpoint if _args.checkpoint else "mlp_mixer_cifar10_head_only.pt"
USE_AMP         = (DEVICE == "cuda")

# Attack settings — matching the paper's CIFAR evaluation
# PGD: 20 steps, step size = eps/4  (standard white-box setting)
PGD_STEPS = 20
EPSILONS  = [2/255, 4/255, 8/255]   # three columns per attack type

# SR resonance levels to sweep (paper uses 0-3, where level = max shift in px)
SR_LEVELS = [0, 1, 2, 3]


# -------------------------------------------------------
# SR Defense Wrapper  (identical to training script)
# -------------------------------------------------------
class SRDefenseWrapper(nn.Module):
    def __init__(self, backbone, classifier, dx=1, dy=1):
        super().__init__()
        self.backbone     = backbone
        self.classifier   = classifier
        self.pool         = nn.AdaptiveAvgPool2d(1)
        self.dx           = dx
        self.dy           = dy
        self.patch_size   = 16
        self.patch_pixels = self.patch_size ** 2

    def _sr_forward(self, x):
        B = x.shape[0]
        aug_list  = []
        ind_to_ij = {}
        ind       = 0
        for i in range(-self.dy, self.dy + 1):
            for j in range(-self.dx, self.dx + 1):
                aug_list.append(
                    transforms.functional.affine(
                        x, translate=[i, j], angle=0, scale=1, shear=0
                    )
                )
                ind_to_ij[ind] = (i, j)
                ind += 1

        big_batch = torch.cat(aug_list, dim=0)
        all_feats = self.backbone.forward_features(big_batch)

        if all_feats.dim() == 3:
            TB, N, D = all_feats.shape
            H_ = W_ = int(N ** 0.5)
            all_feats = all_feats.permute(0, 2, 1).reshape(TB, D, H_, W_)

        feature_field_ensembled = all_feats[0:B, :, :, :].clone()

        for k in range(1, ind):
            i, j   = ind_to_ij[k]
            h_mag  = abs(i)
            w_mag  = abs(j)
            feat_k = all_feats[k * B:(k + 1) * B, :, :, :]

            feature_field_ensembled += feat_k * (
                (self.patch_size - h_mag) * (self.patch_size - w_mag) / self.patch_pixels
            )
            if i > 0:
                feature_field_ensembled[:, :, :-1, :] += feat_k[:, :, 1:, :] * (
                    h_mag * (self.patch_size - w_mag) / self.patch_pixels)
            elif i < 0:
                feature_field_ensembled[:, :, 1:, :] += feat_k[:, :, :-1, :] * (
                    h_mag * (self.patch_size - w_mag) / self.patch_pixels)
            if j > 0:
                feature_field_ensembled[:, :, :, :-1] += feat_k[:, :, :, 1:] * (
                    (self.patch_size - h_mag) * w_mag / self.patch_pixels)
            elif j < 0:
                feature_field_ensembled[:, :, :, 1:] += feat_k[:, :, :, :-1] * (
                    (self.patch_size - h_mag) * w_mag / self.patch_pixels)
            if i > 0 and j > 0:
                feature_field_ensembled[:, :, :-1, :-1] += feat_k[:, :, 1:, 1:] * (
                    h_mag * w_mag / self.patch_pixels)
            elif i > 0 and j < 0:
                feature_field_ensembled[:, :, :-1, 1:] += feat_k[:, :, 1:, :-1] * (
                    h_mag * w_mag / self.patch_pixels)
            elif i < 0 and j > 0:
                feature_field_ensembled[:, :, 1:, :-1] += feat_k[:, :, :-1, 1:] * (
                    h_mag * w_mag / self.patch_pixels)
            elif i < 0 and j < 0:
                feature_field_ensembled[:, :, 1:, 1:] += feat_k[:, :, :-1, :-1] * (
                    h_mag * w_mag / self.patch_pixels)

        pooled = self.pool(feature_field_ensembled).flatten(1)
        return self.classifier(pooled)

    def forward(self, x):
        if self.training:
            feat = self.backbone.forward_features(x)
            if feat.dim() == 3:
                feat = feat.mean(dim=1)
            elif feat.dim() == 4:
                feat = self.pool(feat).flatten(1)
            return self.classifier(feat)
        else:
            return self._sr_forward(x)


# -------------------------------------------------------
# Model builder
# -------------------------------------------------------
def load_state_dict_flexible(model, ckpt_path):
    """
    Load a state_dict robustly, handling three common save formats:
      1. Plain state_dict:          torch.save(model.state_dict(), path)
      2. Dict with 'model' key:     torch.save({'model': model.state_dict()}, path)
      3. Wrapper-prefixed keys:     keys start with '_base_model.' or 'backbone.'
                                    (happens if the SRDefenseWrapper was saved by mistake)
    """
    raw = torch.load(ckpt_path, map_location=DEVICE)

    # Unwrap dict checkpoints (timm-style or custom)
    if isinstance(raw, dict):
        if "model" in raw:
            raw = raw["model"]
        elif "state_dict" in raw:
            raw = raw["state_dict"]

    # Strip any wrapper prefix so keys match the bare timm model
    prefixes = ("_base_model.", "backbone.", "module.")
    cleaned = {}
    for k, v in raw.items():
        for prefix in prefixes:
            if k.startswith(prefix):
                k = k[len(prefix):]
                break
        cleaned[k] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"  [checkpoint] Missing keys  : {missing}")
    if unexpected:
        print(f"  [checkpoint] Unexpected keys: {unexpected}")


def build_model(level):
    """Load checkpoint and wrap with SR at given resonance level (dx=dy=level)."""
    assert os.path.exists(CHECKPOINT_PATH), (
        f"Checkpoint not found: {CHECKPOINT_PATH}\n"
        "Provide a path with --checkpoint /path/to/model.pt"
    )
    base = timm.create_model(MODEL_NAME, pretrained=False)
    base.reset_classifier(num_classes=NUM_CLASSES)
    load_state_dict_flexible(base, CHECKPOINT_PATH)
    wrapped = SRDefenseWrapper(
        backbone=base,
        classifier=base.get_classifier(),
        dx=level,
        dy=level,
    ).to(DEVICE)
    wrapped.eval()
    return wrapped


# -------------------------------------------------------
# Evaluators
# -------------------------------------------------------
@torch.no_grad()
def evaluate_clean(model, loader):
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        if USE_AMP:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(images)
        else:
            logits = model(images)
        correct += (logits.argmax(1) == labels).sum().item()
        total   += labels.size(0)
    return 100.0 * correct / total


def evaluate_fgsm(model, loader, eps):
    """Single-step FGSM L-inf attack."""
    criterion = nn.CrossEntropyLoss()
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        images.requires_grad_(True)

        if USE_AMP:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(images)
                loss   = criterion(logits, labels)
        else:
            logits = model(images)
            loss   = criterion(logits, labels)

        grad = torch.autograd.grad(loss, images)[0]
        adv  = (images + eps * grad.sign()).detach().clamp(0, 1)

        with torch.no_grad():
            adv_logits = model(adv)
        correct += (adv_logits.argmax(1) == labels).sum().item()
        total   += labels.size(0)
    return 100.0 * correct / total


def evaluate_pgd(model, loader, eps, steps):
    """PGD L-inf attack with random start, step size = eps/4."""
    criterion = nn.CrossEntropyLoss()
    step_size = eps / 4
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        delta = torch.empty_like(images).uniform_(-eps, eps)
        delta = torch.clamp(delta, 0 - images, 1 - images).detach()

        for _ in range(steps):
            delta.requires_grad_(True)
            if USE_AMP:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(images + delta)
                    loss   = criterion(logits, labels)
            else:
                logits = model(images + delta)
                loss   = criterion(logits, labels)

            grad  = torch.autograd.grad(loss, delta)[0]
            delta = (delta + step_size * grad.sign()).detach()
            delta = torch.clamp(delta, -eps, eps)
            delta = torch.clamp(delta, 0 - images, 1 - images)

        with torch.no_grad():
            adv_logits = model(images + delta)
        correct += (adv_logits.argmax(1) == labels).sum().item()
        total   += labels.size(0)
    return 100.0 * correct / total


# -------------------------------------------------------
# Data
# -------------------------------------------------------
_cfg = timm.create_model(MODEL_NAME, pretrained=False)
_cfg.reset_classifier(num_classes=NUM_CLASSES)
test_transform = create_transform(**resolve_model_data_config(_cfg), is_training=False)
del _cfg

test_set    = CIFAR10(root="./data", train=False, download=True, transform=test_transform)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

# -------------------------------------------------------
# Main sweep
# -------------------------------------------------------
results = {}

print(f"\nEvaluating {len(SR_LEVELS)} SR levels x "
      f"(1 clean + {len(EPSILONS)} FGSM + {len(EPSILONS)} PGD)")
print(f"Checkpoint : {CHECKPOINT_PATH}")
print(f"Device     : {DEVICE}  |  Batch: {BATCH_SIZE}")
print(f"PGD        : {PGD_STEPS} steps, step=eps/4, random start\n")

for level in SR_LEVELS:
    n_shifts = (2 * level + 1) ** 2
    print(f"SR level {level}  ({n_shifts} shifts/batch)")
    model = build_model(level)
    row   = {"fgsm": {}, "pgd": {}}

    t0 = time.time()
    row["clean"] = evaluate_clean(model, test_loader)
    print(f"  Clean              : {row['clean']:6.2f}%  ({time.time()-t0:.0f}s)")

    for eps in EPSILONS:
        t0  = time.time()
        acc = evaluate_fgsm(model, test_loader, eps)
        row["fgsm"][eps] = acc
        print(f"  FGSM  eps={eps*255:.0f}/255   : {acc:6.2f}%  ({time.time()-t0:.0f}s)")

    for eps in EPSILONS:
        t0  = time.time()
        acc = evaluate_pgd(model, test_loader, eps, PGD_STEPS)
        row["pgd"][eps] = acc
        print(f"  PGD   eps={eps*255:.0f}/255   : {acc:6.2f}%  ({time.time()-t0:.0f}s)")

    results[level] = row
    print()
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

# -------------------------------------------------------
# Paper-style results table
# -------------------------------------------------------
eps_labels = [f"{int(e*255)}/255" for e in EPSILONS]
cw = 10  # column width

# Build column headers
header = f"{'SR Level':>10}  {'Clean':>{cw}}"
for lbl in eps_labels:
    header += f"  {'FGSM '+lbl:>{cw}}"
for lbl in eps_labels:
    header += f"  {'PGD '+lbl:>{cw}}"
divider = "=" * len(header)

print(divider)
print("  Accuracy (%) — CIFAR-10, MLP-Mixer, white-box L-inf attacks")
print(divider)
print(header)
print("-" * len(header))

for level in SR_LEVELS:
    row  = results[level]
    line = f"{'Level '+str(level):>10}  {row['clean']:>{cw}.2f}"
    for eps in EPSILONS:
        line += f"  {row['fgsm'][eps]:>{cw}.2f}"
    for eps in EPSILONS:
        line += f"  {row['pgd'][eps]:>{cw}.2f}"
    print(line)

print(divider)
