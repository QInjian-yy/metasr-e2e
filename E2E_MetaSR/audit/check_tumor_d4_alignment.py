"""Real tumor_111/D4 audit: image pairing, live training wiring, independent pixel oracle."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import engine
from audit.check_real_pair_alignment import mae, point_oracle
from models.baseline import MetaSRABMIL
from scripts.probe_cuda import load_real_sample
from train_e2e import load_config


def dense_oracle(sr, feature, box):
    """Every RGB pixel, explicit channel/3x3 indexing; no unfold, tiles or position_table."""
    y, x, h, w = box
    yy, xx = torch.meshgrid(torch.arange(y, y+h), torch.arange(x, x+w), indexing="ij")
    yy, xx = yy.flatten(), xx.flatten()
    positions = feature.new_tensor([(1/32, dy/32, dx/32) for dy in range(32) for dx in range(32)])
    kernels = sr.P2W(positions).reshape(1024, 576, 3)
    rows = []
    for start in range(0, len(yy), 1024):
        ys, xs = yy[start:start+1024], xx[start:start+1024]
        values = feature.new_zeros((len(ys), 64, 3, 3))
        for ky in range(3):
            for kx in range(3):
                fy, fx = ys//32 + ky-1, xs//32 + kx-1
                valid = (fy >= 0) & (fy < 256) & (fx >= 0) & (fx < 256)
                values[valid, :, ky, kx] = feature[0, :, fy[valid], fx[valid]].T
        selected = kernels[(ys % 32)*32 + xs % 32]
        rows.append(torch.bmm(values.reshape(-1, 1, 576), selected)[:, 0])
    rgb = torch.cat(rows).T.reshape(1, 3, h, w)
    return sr.add_mean(rgb)


def main():
    output = ROOT / "reports/tumor_d4_alignment_20260930"
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    config = load_config(ROOT / "configs/rdn_d4.yaml")
    sample = load_real_sample(Path(r"D:\ddata"), "tumor_111", 4)
    smoke = json.loads((ROOT / "reports/real_tumor_111_d4_n4_20260930.json").read_text())
    report = {"sample": sample, "config": config, "pairs": [], "training_calls": [], "decoder_checks": []}
    pairs = []
    for lr_path, hr_path in zip(sample["lr_paths"], sample["hr_paths"]):
        with Image.open(lr_path) as im:
            lr = np.array(im.convert("RGB"))
        with Image.open(hr_path) as im:
            hr = np.array(im.convert("RGB"))
            down = np.array(im.convert("RGB").resize((256, 256), Image.Resampling.LANCZOS))
        shifts = sorted((mae(lr[4:252, 4:252], down[4+dy:252+dy, 4+dx:252+dx]), dy, dx)
                        for dy in range(-4, 5) for dx in range(-4, 5))
        row = {"filename": Path(lr_path).name, "mae_0_255": mae(lr, down),
               "correlation": float(np.corrcoef(lr.ravel(), down.ravel())[0, 1]),
               "best_translation_dy_dx": list(shifts[0][1:]),
               "best_translation_mae": shifts[0][0], "runner_up_translation_mae": shifts[1][0],
               "orientation_negative_mae": {"flip_x": mae(lr, down[:, ::-1]),
                   "flip_y": mae(lr, down[::-1]), "transpose": mae(lr, down.transpose(1, 0, 2)),
                   "rotate90": mae(lr, np.rot90(down))}}
        pairs.append((lr, hr, down))
        report["pairs"].append(row)
        print("PAIR", json.dumps(row), flush=True)
    cross = [[mae(a[0], b[2]) for b in pairs] for a in pairs]
    report["pair_mae_matrix"] = cross
    (output / "image_pair_checks.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for i, row in enumerate(report["pairs"]):
        assert row["best_translation_dy_dx"] == [0, 0], row
        assert row["correlation"] > .99, row
        assert np.argmin(cross[i]) == i, cross[i]
        assert min(row["orientation_negative_mae"].values()) > row["mae_0_255"]

    torch.manual_seed(config["seed"])
    model = MetaSRABMIL(**config["metasr"]).cuda().train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    features, predictions, truths, bags = [], [], [], []
    encode, load_gt, decode = model.encode_regions, engine.load_hr_crops, model.sr.decode_crop

    def record_encode(lr):
        assert list(lr.shape) == [4, 3, 256, 256]
        result = encode(lr)
        features.append(result[0])
        assert result[0].requires_grad and result[0].grad_fn is not None
        return result

    def record_gt(paths, box):
        gt = load_gt(paths, box)
        i = sample["hr_paths"].index(paths[0])
        y, x, h, w = box
        expected = torch.from_numpy(pairs[i][1][y:y+h, x:x+w].copy()).permute(2, 0, 1).float()/255
        assert torch.equal(gt[0], expected)
        truths.append(gt.clone())
        report["training_calls"].append({"index": i, "hr_path": paths[0],
            "box_y_x_h_w": list(box), "gt_pixel_max_error": 0.0})
        return gt

    def record_decode(feature, box):
        row = report["training_calls"][-1]
        i = row["index"]
        assert list(box) == row["box_y_x_h_w"]
        assert torch.equal(feature, features[0][i:i+1])
        assert feature.requires_grad and feature.grad_fn is not None
        prediction = decode(feature, box)
        assert list(prediction.shape) == [1, 3, 256, 256]
        predictions.append(prediction.detach().float().cpu())
        row["correct_region_feature_and_absolute_box"] = True
        return prediction

    handle = model.mil_head.register_forward_pre_hook(lambda module, inputs: bags.append(list(inputs[0].shape)))
    print("CUDA: full N=4 BF16 training with live coordinate hooks", flush=True)
    with patch.object(model, "encode_regions", side_effect=record_encode), \
            patch("engine.load_hr_crops", side_effect=record_gt), \
            patch.object(model.sr, "decode_crop", side_effect=record_decode):
        loss, cls, sr_loss, logits = engine.wsi_loss(model, sample, torch.device("cuda:0"), config)
    handle.remove()
    assert len(features) == 1 and bags == [[4, 512]]
    assert len(report["training_calls"]) == 4
    for i, row in enumerate(report["training_calls"]):
        assert row["index"] == i and row["box_y_x_h_w"] == smoke["hr_crops"][i]["box_y_x_h_w"]
    independent_sr = torch.stack([(a-b).abs().mean() for a, b in zip(predictions, truths)]).mean()
    torch.testing.assert_close(sr_loss.detach().cpu(), independent_sr, atol=1e-7, rtol=1e-7)
    real_features = features[0].detach().cpu().double()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    report["gradient_norms"] = {"rdn": engine.gradient_norm(model.sr.SFENet1),
        "p2w": engine.gradient_norm(model.sr.P2W), "abmil": engine.gradient_norm(model.mil_head)}
    assert all(v > 0 for v in report["gradient_norms"].values())
    optimizer.step()
    report["cuda_training"] = {"forward": True, "backward": True, "adam_step": True,
        "encoder_input_shape": [4, 3, 256, 256], "mil_inputs": bags,
        "loss_sr": sr_loss.item(), "independent_mean_of_four_sr_losses": independent_sr.item(),
        "max_loss_difference_from_smoke": max(abs(value.item()-smoke[key])
            for value, key in [(loss, "loss_total"), (cls, "loss_cls"), (sr_loss, "loss_sr")])}
    # Fresh pre-update decoder for independent checks of the pre-update feature tensor.
    del encode, load_gt, decode, features, model, optimizer, loss, cls, sr_loss, logits
    torch.cuda.empty_cache()
    torch.manual_seed(config["seed"])
    decoder = MetaSRABMIL(**config["metasr"]).sr.double().eval()
    offsets = [(y, x) for y in (0, 1, 15, 31, 32, 127, 254, 255)
               for x in (0, 1, 15, 31, 32, 127, 254, 255)]
    with torch.no_grad():
        for i, row in enumerate(report["training_calls"]):
            feature = real_features[i:i+1]
            box = row["box_y_x_h_w"]
            predicted = decoder.decode_crop(feature, box)
            expected = dense_oracle(decoder, feature, box)
            torch.testing.assert_close(predicted, expected, atol=1e-10, rtol=1e-10)
            error = float((predicted-expected).abs().max())
            report["decoder_checks"].append({"region": i, "box": box,
                "pixels_checked": 256*256, "fp64_max_abs_error": error})
            print("ALL_PIXELS", i, error, flush=True)
            for boundary in ((0, 0, 256, 256), (7936, 7936, 256, 256), (137, 291, 256, 256)):
                y, x, h, w = boundary
                pred = decoder.decode_crop(feature, boundary)
                points = [(y+dy, x+dx) for dy, dx in offsets]
                ref = point_oracle(decoder, feature, points)
                actual = torch.stack([pred[0, :, dy, dx] for dy, dx in offsets])
                torch.testing.assert_close(actual, ref, atol=1e-10, rtol=1e-10)
                check = {"region": i, "box": list(boundary), "pixels_checked": len(points),
                         "fp64_max_abs_error": float((actual-ref).abs().max())}
                if boundary[0] == 137:
                    controls = {"swapped_xy": [(xx, yy) for yy, xx in points],
                        "shift_one_hr_pixel": [(yy+1, xx) for yy, xx in points],
                        "shift_one_lr_pixel": [(yy+32, xx) for yy, xx in points],
                        "reset_crop_origin": offsets}
                    check["negative_controls"] = {}
                    for key, wrong_points in controls.items():
                        wrong = point_oracle(decoder, feature, wrong_points)
                        assert not torch.allclose(actual, wrong, atol=1e-10, rtol=1e-10)
                        check["negative_controls"][key] = float((actual-wrong).abs().max())
                report["decoder_checks"].append(check)

    # Compare SR-only derivatives on selected real-feature pixels with the independent oracle.
    feature = real_features[:1].clone().requires_grad_()
    box = report["training_calls"][0]["box_y_x_h_w"]
    chosen = [(0, 0), (1, 31), (31, 32), (32, 31), (127, 127), (255, 255)]
    y, x, _, _ = box
    parameters = [feature, *decoder.P2W.parameters()]
    pred = decoder.decode_crop(feature, box)
    actual = torch.stack([pred[0, :, dy, dx] for dy, dx in chosen])
    ga = torch.autograd.grad(actual.square().mean(), parameters)
    expected = point_oracle(decoder, feature, [(y+dy, x+dx) for dy, dx in chosen])
    gb = torch.autograd.grad(expected.square().mean(), parameters)
    for a, b in zip(ga, gb):
        torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)
    report["sr_only_gradient_oracle"] = {"feature_and_p2w_checked": True,
        "max_abs_errors": [float((a-b).abs().max()) for a, b in zip(ga, gb)]}
    report["status"] = "passed"
    report["limitations"] = ["Exported JPEG pairs only; no original WSI or extraction transform was inspected.",
        "Downsample similarity and integer-LR shift tests do not prove exact sub-HR-pixel physical registration.",
        "Independent FP64 oracle validates coordinates/layout on real CUDA BF16 features, not BF16-vs-FP64 numerical equivalence.",
        "Randomly initialized SR predictions are not expected to visually match ground truth."]
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("PASSED", str(output / "report.json"), flush=True)


if __name__ == "__main__":
    main()
