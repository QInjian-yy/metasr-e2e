"""Audit the two real normal_001 pairs, crop wiring, and decoder coordinates."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import engine
from models.baseline import MetaSRABMIL
from scripts.probe_cuda import load_real_sample


def mae(a, b):
    return float(np.abs(a.astype(np.float64) - b.astype(np.float64)).mean())


def point_oracle(sr, feature, points):
    """Explicit channel/y/x neighborhood indexing; no unfold or decoder tiles."""
    vectors, positions = [], []
    for y, x in points:
        ly, lx = y // 32, x // 32
        neighborhood = feature.new_zeros((64, 3, 3))
        for ky in range(3):
            for kx in range(3):
                fy, fx = ly + ky - 1, lx + kx - 1
                if 0 <= fy < feature.shape[2] and 0 <= fx < feature.shape[3]:
                    neighborhood[:, ky, kx] = feature[0, :, fy, fx]
        vectors.append(neighborhood.flatten())
        positions.append((1 / 32, (y % 32) / 32, (x % 32) / 32))
    kernels = sr.P2W(feature.new_tensor(positions)).reshape(-1, 576, 3)
    rgb = torch.einsum("pk,pkc->pc", torch.stack(vectors), kernels)
    return sr.add_mean(rgb.T[None, :, :, None])[0, :, :, 0].T


def main():
    data_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"D:\ddata")
    output = ROOT / "reports/real_pair_alignment_20260930"
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    sample = load_real_sample(data_root, "normal_001", 2)
    report = {"data_root": str(data_root), "slide_id": "normal_001", "pairs": [],
              "replays": [], "decoder_checks": [], "original_wsi_available": False}
    pairs = []
    for lr_path, hr_path in zip(sample["lr_paths"], sample["hr_paths"]):
        with Image.open(lr_path) as image:
            lr = np.array(image.convert("RGB"))
        with Image.open(hr_path) as image:
            hr = np.array(image.convert("RGB"))
            down = np.array(image.resize((256, 256), Image.Resampling.LANCZOS))
        shifts = []
        for dy in range(-4, 5):
            for dx in range(-4, 5):
                shifts.append((mae(lr[4:252, 4:252], down[4+dy:252+dy, 4+dx:252+dx]), dy, dx))
        shifts.sort()
        item = {"filename": Path(lr_path).name, "lr_shape": list(lr.shape), "hr_shape": list(hr.shape),
                "downsample_method": "PIL LANCZOS", "mae_0_255": mae(lr, down),
                "correlation": float(np.corrcoef(lr.ravel(), down.ravel())[0, 1]),
                "best_translation_dy_dx": list(shifts[0][1:]),
                "best_translation_mae": shifts[0][0], "runner_up_translation_mae": shifts[1][0],
                "negative_controls_mae": {"horizontal_flip": mae(lr, down[:, ::-1]),
                                          "vertical_flip": mae(lr, down[::-1]),
                                          "transpose_xy": mae(lr, down.transpose(1, 0, 2)),
                                          "rotate90": mae(lr, np.rot90(down))}}
        assert shifts[0][1:] == (0, 0), item
        assert item["correlation"] > 0.99, item
        pairs.append((lr, hr, down))
        report["pairs"].append(item)
        print("PAIR", json.dumps(item), flush=True)
    for i in range(2):
        wrong = mae(pairs[i][0], pairs[1-i][2])
        report["pairs"][i]["negative_controls_mae"]["wrong_region_pair"] = wrong
        assert wrong > report["pairs"][i]["mae_0_255"] * 5

    # Replay the smoke-test seed and initialization, recording the actual engine calls.
    for n in (1, 2):
        torch.manual_seed(7)
        model = MetaSRABMIL(checkpoint_rdb=True).cuda().train()
        real_sample = load_real_sample(data_root, "normal_001", n)
        calls, features = [], []
        encode, load_gt, decode = model.encode_regions, engine.load_hr_crops, model.sr.decode_crop

        def record_encode(lr):
            result = encode(lr)
            features.append(result[0])
            return result

        def record_gt(paths, box):
            truth = load_gt(paths, box)
            index = real_sample["hr_paths"].index(paths[0])
            y, x, h, w = box
            expected = torch.from_numpy(pairs[index][1][y:y+h, x:x+w].copy()).permute(2, 0, 1).float() / 255
            error = float((truth[0] - expected).abs().max())
            assert error == 0
            calls.append({"region_index": index, "hr_path": paths[0], "box_y_x_h_w": list(box),
                          "gt_max_abs_error": error})
            return truth

        def record_decode(feature, box):
            call = calls[-1]
            index = call["region_index"]
            assert list(box) == call["box_y_x_h_w"]
            assert torch.equal(feature, features[0][index:index+1])
            call["matching_feature_and_box"] = True
            return decode(feature, box)

        config = {"training": {"use_region_microbatch": False, "region_microbatch_size": 1},
                  "lambda_sr": 0.1, "sr_train_crop": 256, "precision": "bf16"}
        print(f"REPLAY N={n}: real-image forward, seed=7", flush=True)
        with torch.no_grad(), patch.object(model, "encode_regions", side_effect=record_encode), \
                patch("engine.load_hr_crops", side_effect=record_gt), \
                patch.object(model.sr, "decode_crop", side_effect=record_decode):
            total, cls, sr, _ = engine.wsi_loss(model, real_sample, torch.device("cuda:0"), config)
        old = json.loads((ROOT / f"reports/real_normal_001_n{n}_20260930.json").read_text())
        losses = {"loss_total": total.item(), "loss_cls": cls.item(), "loss_sr": sr.item()}
        report["replays"].append({"n": n, "calls": calls, **losses,
                                  "max_loss_difference_from_smoke": max(abs(v - old[k]) for k, v in losses.items()),
                                  "scope": "forward only, no optimizer update"})
        print("REPLAY_RESULT", json.dumps(report["replays"][-1]), flush=True)
        if n == 2:
            real_features = features[0].cpu().double()
            decoder = model.sr.cpu().double()
            actual_boxes = [tuple(call["box_y_x_h_w"]) for call in calls]
        # Drop bound-method references before starting the next independent replay.
        del encode, load_gt, decode, model, features, total, cls, sr
        torch.cuda.empty_cache()

    # FP64 isolates coordinate/layout correctness from BF16 rounding.
    offsets = [(y, x) for y in (0, 1, 15, 31, 32, 127, 254, 255)
               for x in (0, 1, 15, 31, 32, 127, 254, 255)]
    with torch.no_grad():
        for i in range(2):
            feature = real_features[i:i+1]
            boxes = [actual_boxes[i], (0, 0, 256, 256), (137, 291, 256, 256),
                     (3968, 3968, 256, 256), (7936, 7936, 256, 256)]
            for box in boxes:
                y, x, h, w = box
                gt = engine.load_hr_crops([sample["hr_paths"][i]], box)[0]
                expected_gt = torch.from_numpy(pairs[i][1][y:y+h, x:x+w].copy()).permute(2, 0, 1).float() / 255
                gt_error = float((gt - expected_gt).abs().max())
                prediction = decoder.decode_crop(feature, box)
                points = [(y+dy, x+dx) for dy, dx in offsets]
                expected = point_oracle(decoder, feature, points)
                actual = torch.stack([prediction[0, :, dy, dx] for dy, dx in offsets])
                error = float((actual - expected).abs().max())
                assert gt_error == 0
                torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
                check = {"region_index": i, "box_y_x_h_w": list(box), "oracle_points": len(points),
                         "gt_max_abs_error": gt_error, "decoder_fp64_max_abs_error": error}
                if box == (137, 291, 256, 256):
                    swapped = point_oracle(decoder, feature, [(x, y) for y, x in points])
                    shifted = point_oracle(decoder, feature, [(y+32, x) for y, x in points])
                    check["negative_control_swapped_xy_error"] = float((actual - swapped).abs().max())
                    check["negative_control_shifted_one_lr_pixel_error"] = float((actual - shifted).abs().max())
                    assert check["negative_control_swapped_xy_error"] > 1e-5
                    assert check["negative_control_shifted_one_lr_pixel_error"] > 1e-5
                report["decoder_checks"].append(check)
                print("CROP", json.dumps(check), flush=True)

    fig, axes = plt.subplots(2, 4, figsize=(14, 7), constrained_layout=True)
    for i, (lr, hr, down) in enumerate(pairs):
        y, x, h, w = actual_boxes[i]
        panels = [lr, down, np.minimum(np.abs(lr.astype(float)-down)*5, 255).astype(np.uint8),
                  hr[y:y+h, x:x+w]]
        titles = ["Actual LR 256x256", "Actual HR resized to 256", "Absolute RGB difference x5",
                  f"Actual HR crop (y={y}, x={x})"]
        for j in range(4):
            axes[i, j].imshow(panels[j])
            axes[i, j].set_title(titles[j], fontsize=10)
            axes[i, j].axis("off")
            if j < 2:
                axes[i, j].add_patch(Rectangle((x/32, y/32), w/32, h/32, fill=False,
                                               edgecolor="#00c94f", linewidth=1.5))
        axes[i, 0].text(0, -0.07, report["pairs"][i]["filename"], transform=axes[i, 0].transAxes, fontsize=8)
    fig.suptitle("Real normal_001 pairs: green boxes mark the replayed N=2 SR crops", fontsize=13)
    fig.savefig(output / "alignment.png", dpi=150)
    plt.close(fig)
    report["status"] = "passed"
    report["limitations"] = ["Checks cover these two exported image pairs, not the original WSI file.",
                             "Image similarity and integer-LR shift search do not prove sub-HR-pixel registration.",
                             "The decoder oracle checks 64 points per crop in FP64 using features from real LR images."]
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("SAVED", str(output), flush=True)


if __name__ == "__main__":
    main()
