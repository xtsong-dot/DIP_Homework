import argparse
import math
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from data_utils import ColmapDataset
from gaussian_model import GaussianModel
from gaussian_renderer import GaussianRenderer


def parse_indices(text):
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def psnr(mse):
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


def ssim_simple(a, b):
    """Channel-averaged SSIM in [0, 1] for float RGB images in [0, 1]."""
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    scores = []
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    for c in range(3):
        x = a[..., c]
        y = b[..., c]
        mu_x = cv2.GaussianBlur(x, (7, 7), 1.5)
        mu_y = cv2.GaussianBlur(y, (7, 7), 1.5)
        sigma_x = cv2.GaussianBlur(x * x, (7, 7), 1.5) - mu_x * mu_x
        sigma_y = cv2.GaussianBlur(y * y, (7, 7), 1.5) - mu_y * mu_y
        sigma_xy = cv2.GaussianBlur(x * y, (7, 7), 1.5) - mu_x * mu_y
        score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2))
        score /= ((mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2))
        scores.append(float(np.clip(score.mean(), 0.0, 1.0)))
    return float(np.mean(scores))


def draw_curve(rows, output_path):
    width, height = 900, 420
    margin_left, margin_right = 70, 30
    margin_top, margin_bottom = 40, 70
    image = np.full((height, width, 3), 255, np.uint8)

    epochs = np.array([row["epoch"] for row in rows], dtype=np.float32)
    l1 = np.array([row["l1"] for row in rows], dtype=np.float32)
    psnr_values = np.array([row["psnr"] for row in rows], dtype=np.float32)

    x0, y0 = margin_left, height - margin_bottom
    x1, y1 = width - margin_right, margin_top
    cv2.rectangle(image, (x0, y1), (x1, y0), (220, 220, 220), 1)

    def map_x(values):
        denom = max(float(epochs.max() - epochs.min()), 1.0)
        return x0 + ((values - epochs.min()) / denom * (x1 - x0)).astype(np.int32)

    def map_y(values, low, high):
        denom = max(float(high - low), 1e-6)
        return y0 - ((values - low) / denom * (y0 - y1)).astype(np.int32)

    l1_low, l1_high = float(l1.min()) * 0.95, float(l1.max()) * 1.05
    psnr_low, psnr_high = float(psnr_values.min()) * 0.95, float(psnr_values.max()) * 1.05
    xs = map_x(epochs)
    l1_ys = map_y(l1, l1_low, l1_high)
    psnr_ys = map_y(psnr_values, psnr_low, psnr_high)

    for pts, color in ((l1_ys, (40, 80, 220)), (psnr_ys, (40, 160, 60))):
        for i in range(1, len(xs)):
            cv2.line(image, (xs[i - 1], pts[i - 1]), (xs[i], pts[i]), color, 2)
        for x, y in zip(xs, pts):
            cv2.circle(image, (int(x), int(y)), 5, color, -1)

    cv2.putText(image, "Evaluation Curves", (margin_left, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (40, 40, 40), 2, cv2.LINE_AA)
    cv2.putText(image, "orange: L1 lower is better", (margin_left, height - 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 80, 220), 1, cv2.LINE_AA)
    cv2.putText(image, "green: PSNR higher is better", (margin_left + 270, height - 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 160, 60), 1, cv2.LINE_AA)
    cv2.putText(image, f"L1 range {l1_low:.3f}-{l1_high:.3f}", (width - 300, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (40, 80, 220), 1, cv2.LINE_AA)
    cv2.putText(image, f"PSNR range {psnr_low:.1f}-{psnr_high:.1f}", (width - 300, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (40, 160, 60), 1, cv2.LINE_AA)
    for row, x in zip(rows, xs):
        cv2.putText(image, str(row["epoch"]), (int(x) - 12, y0 + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1, cv2.LINE_AA)
    cv2.imwrite(str(output_path), image)


def save_comparison(dataset, model, renderer, indices, output_path, device):
    with torch.no_grad():
        params = model()
    gt_cells, rendered_cells = [], []
    for idx in indices:
        item = dataset[idx]
        with torch.no_grad():
            rendered = renderer(
                params["positions"], params["covariance"], params["colors"], params["opacities"],
                item["K"].to(device), item["R"].to(device), item["t"].reshape(3).to(device)
            )
        gt = (item["image"].numpy() * 255).clip(0, 255).astype(np.uint8)
        pred = (rendered.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        gt = cv2.cvtColor(gt, cv2.COLOR_RGB2BGR)
        pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
        cv2.putText(gt, Path(item["image_path"]).stem, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 1, cv2.LINE_AA)
        gt_cells.append(gt)
        rendered_cells.append(pred)
    gt_row = np.concatenate(gt_cells, axis=1)
    pred_row = np.concatenate(rendered_cells, axis=1)
    cv2.putText(gt_row, "GT", (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(pred_row, "Rendered", (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(output_path), np.concatenate([gt_row, pred_row], axis=0))


def main():
    parser = argparse.ArgumentParser(description="Evaluate simplified 3DGS checkpoints")
    parser.add_argument("--colmap_dir", required=True)
    parser.add_argument("--checkpoint_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--checkpoints", default=None,
                        help="Comma-separated checkpoint paths. Defaults to checkpoint_*.pt in checkpoint_dir.")
    parser.add_argument("--downsample_factor", type=int, default=4)
    parser.add_argument("--maximum_pts_num", type=int, default=3000)
    parser.add_argument("--max_views", type=int, default=30)
    parser.add_argument("--scale_multiplier", type=float, default=0.25)
    parser.add_argument("--eval_indices", default="0,8,16,24")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir or Path("results") / Path(args.colmap_dir).name)
    output_dir = Path(args.output_dir or checkpoint_dir / "analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.checkpoints:
        checkpoint_paths = [Path(p.strip()) for p in args.checkpoints.split(",") if p.strip()]
    else:
        checkpoint_paths = sorted(checkpoint_dir.glob("checkpoint_*.pt"))
    if not checkpoint_paths:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    indices = parse_indices(args.eval_indices)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    dataset = ColmapDataset(
        args.colmap_dir,
        downsample_factor=args.downsample_factor,
        maximum_pts_num=args.maximum_pts_num,
        max_views=args.max_views,
    )
    H, W = dataset[0]["image"].shape[:2]
    renderer = GaussianRenderer(H, W).to(device)

    summary_rows = []
    per_view_rows = []
    final_model = None
    final_epoch = -1
    for checkpoint_path in checkpoint_paths:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        epoch = int(checkpoint.get("epoch", -1))
        model = GaussianModel(
            dataset.points3D_xyz,
            dataset.points3D_rgb,
            scale_multiplier=args.scale_multiplier,
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        with torch.no_grad():
            params = model()

        metrics = []
        for idx in indices:
            item = dataset[idx]
            gt = item["image"].numpy().astype(np.float32)
            with torch.no_grad():
                pred_t = renderer(
                    params["positions"], params["covariance"], params["colors"], params["opacities"],
                    item["K"].to(device), item["R"].to(device), item["t"].reshape(3).to(device)
                )
            pred = pred_t.cpu().numpy().astype(np.float32)
            l1 = float(np.mean(np.abs(pred - gt)))
            mse = float(np.mean((pred - gt) ** 2))
            row = {
                "epoch": epoch,
                "view": idx,
                "l1": l1,
                "mse": mse,
                "psnr": psnr(mse),
                "ssim": ssim_simple(gt, pred),
            }
            metrics.append(row)
            per_view_rows.append(row)

        avg = {
            "epoch": epoch,
            "checkpoint": str(checkpoint_path),
            "l1": float(np.mean([m["l1"] for m in metrics])),
            "mse": float(np.mean([m["mse"] for m in metrics])),
            "psnr": float(np.mean([m["psnr"] for m in metrics])),
            "ssim": float(np.mean([m["ssim"] for m in metrics])),
        }
        summary_rows.append(avg)
        if epoch > final_epoch:
            final_epoch = epoch
            final_model = model

    summary_rows.sort(key=lambda r: r["epoch"])
    with open(output_dir / "metrics.csv", "w", encoding="utf-8") as f:
        f.write("epoch,checkpoint,l1,mse,psnr,ssim\n")
        for row in summary_rows:
            f.write(f"{row['epoch']},{row['checkpoint']},{row['l1']:.8f},{row['mse']:.8f},{row['psnr']:.4f},{row['ssim']:.4f}\n")
    with open(output_dir / "per_view_metrics.csv", "w", encoding="utf-8") as f:
        f.write("epoch,view,l1,mse,psnr,ssim\n")
        for row in per_view_rows:
            f.write(f"{row['epoch']},{row['view']},{row['l1']:.8f},{row['mse']:.8f},{row['psnr']:.4f},{row['ssim']:.4f}\n")

    draw_curve(summary_rows, output_dir / "metric_curves.png")
    if final_model is not None:
        save_comparison(dataset, final_model, renderer, indices, output_dir / "final_comparison.png", device)

    with open(output_dir / "metrics_summary.md", "w", encoding="utf-8") as f:
        f.write("# 3DGS Evaluation Summary\n\n")
        f.write("| Epoch | L1 lower better | PSNR higher better | SSIM higher better |\n")
        f.write("| --- | ---: | ---: | ---: |\n")
        for row in summary_rows:
            f.write(f"| {row['epoch']} | {row['l1']:.4f} | {row['psnr']:.2f} | {row['ssim']:.3f} |\n")
        best = summary_rows[-1]
        f.write("\n")
        f.write(
            f"Final checkpoint epoch {best['epoch']} reaches L1={best['l1']:.4f}, "
            f"PSNR={best['psnr']:.2f} dB, SSIM={best['ssim']:.3f} on views {indices}.\n"
        )

    print(f"Wrote evaluation artifacts to {output_dir}")


if __name__ == "__main__":
    main()
