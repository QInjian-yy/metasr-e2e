"""One complete WSI with an optional region micro-batch and one Adam step."""

import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

from augmentation import AugmentedPath
from wsi_data import load_images


def autocast(device, precision):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                          enabled=precision == "bf16" and device.type == "cuda")


def load_hr_crops(paths, box, size=8192):
    """Decode GT on CPU and transfer only RGB float crops, never an 8K GPU tensor."""
    y, x, h, w = box
    crops = []
    for path in paths:
        with Image.open(path) as image:
            if image.size != (size, size):
                raise ValueError(f"{path}: expected HR {(size, size)}, got {image.size}")
            if min(y, x) < 0 or min(h, w) < 1 or y+h > size or x+w > size:
                raise ValueError("Invalid HR crop")
            array = np.array(image.crop((x, y, x+w, y+h)).convert("RGB"), dtype=np.uint8)
        crops.append(torch.from_numpy(array).permute(2, 0, 1).float().div_(255))
    return torch.stack(crops)


def region_batch_size(config, n_regions):
    training = config["training"]
    if not training["use_region_microbatch"]:
        return n_regions
    return min(training["region_microbatch_size"], n_regions)


def wsi_loss(model, sample, device, config):
    """Retain a normal autograd graph; no gradient cache, BN replay or RNG restore."""
    n = sample["n_regions"]
    batch_size = region_batch_size(config, n)
    coefficient = config["lambda_sr"]
    if coefficient:
        for path in sample["lr_paths"] + sample["hr_paths"]:
            if isinstance(path, AugmentedPath):
                params = path.params
                if (params.horizontal_flip != params.vertical_flip
                        or (params.rotation_k + 2 * int(params.horizontal_flip)) % 4):
                    raise ValueError("SR supervision cannot use geometric augmentation: "
                                     "HR crops are not transformed to match augmented regions. "
                                     "Disable geometry or set lambda_sr=0.")
        crop = config["sr_train_crop"]
        # Sample once per region before batching, so batching does not change the crops.
        boxes = [(y, x, crop, crop)
                 for y, x in torch.randint(8192 - crop + 1, (n, 2)).tolist()]
    embeddings, sr_terms = [], []
    for start in range(0, n, batch_size):
        lr = load_images(sample["lr_paths"][start:start+batch_size], 256).to(device)
        with autocast(device, config["precision"]):
            feature, embedding = model.encode_regions(lr)
            embeddings.append(embedding.float())
            if coefficient:
                for index in range(lr.shape[0]):
                    box = boxes[start+index]
                    truth = load_hr_crops([sample["hr_paths"][start+index]], box).to(device)
                    prediction = model.sr.decode_crop(feature[index:index+1], box)
                    sr_terms.append(F.l1_loss(prediction.float(), truth) / n)
    with autocast(device, config["precision"]):
        logits = model.forward_embeddings(torch.cat(embeddings))
        target = torch.tensor([sample["label"]], dtype=torch.long, device=device)
        cls_loss = F.cross_entropy(logits, target)
    sr_loss = torch.stack(sr_terms).sum() if sr_terms else cls_loss.new_zeros(())
    return cls_loss + coefficient * sr_loss, cls_loss, sr_loss, logits


def memory_summary(device):
    if device.type != "cuda":
        return {"max_memory_allocated": None, "max_memory_reserved": None}
    torch.cuda.synchronize(device)
    return {"max_memory_allocated": torch.cuda.max_memory_allocated(device),
            "max_memory_reserved": torch.cuda.max_memory_reserved(device)}


def gradient_norm(module):
    values = [p.grad.detach().float().norm() for p in module.parameters() if p.grad is not None]
    return torch.stack(values).norm().item() if values else 0.0


def train_wsi(model, sample, optimizer, device, config, log_gradients=False):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    total, cls, sr, logits = wsi_loss(model, sample, device, config)
    if not torch.isfinite(total):
        raise FloatingPointError("Non-finite joint loss")
    total.backward()
    norms = {"rdn": gradient_norm(model.sr), "region_encoder": gradient_norm(model.region_encoder),
             "abmil": gradient_norm(model.mil_head), "classifier": gradient_norm(model.classifier),
             "p2w": gradient_norm(model.sr.P2W)}
    if not all(math.isfinite(value) for value in norms.values()):
        raise FloatingPointError(f"Non-finite gradients: {norms}")
    if log_gradients:
        print(f"[gradient] slide_id={sample['slide_id']} N={sample['n_regions']} {norms}", flush=True)
    optimizer.step()
    return {"train_loss_cls": cls.item(), "train_loss_sr": sr.item(),
            "train_loss_total": total.item(), **memory_summary(device)}


@torch.no_grad()
def evaluate(model, loader, device, config):
    # Matches E2E's offline classification-only train/val evaluation semantics.
    model.eval()
    labels, scores, decisions, predictions = [], [], [], []
    loss_sum = 0.0
    for sample in loader:
        chunks = []
        batch_size = region_batch_size(config, sample["n_regions"])
        for start in range(0, sample["n_regions"], batch_size):
            paths = sample["lr_paths"][start:start+batch_size]
            with autocast(device, config["precision"]):
                _, embedding = model.encode_regions(load_images(paths, 256).to(device))
            chunks.append(embedding.float())
        with autocast(device, config["precision"]):
            logits = model.forward_embeddings(torch.cat(chunks))
            target = torch.tensor([sample["label"]], dtype=torch.long, device=device)
            loss_sum += F.cross_entropy(logits, target).item()
        score = torch.softmax(logits.float(), dim=1)[0, 1].item()
        decision = logits.argmax(dim=1).item()
        labels.append(sample["label"])
        scores.append(score)
        decisions.append(decision)
        predictions.append({"slide_id": sample["slide_id"], "label": sample["label"],
                            "p_tumor": score, "prediction": decision, "n_regions": sample["n_regions"]})
    if not labels:
        raise ValueError("Cannot evaluate an empty split")
    return {"loss": loss_sum / len(labels),
            "auc": roc_auc_score(labels, scores) if len(set(labels)) == 2 else float("nan"),
            "acc": accuracy_score(labels, decisions),
            "bacc": balanced_accuracy_score(labels, decisions), "predictions": predictions}