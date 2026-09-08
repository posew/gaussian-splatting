#!/usr/bin/env python3
"""Validate the fixed Scene 1 experiment, then write summary.json and SUMMARY.md.

Usage: python summarize_run.py RUN_ROOT
       python summarize_run.py --self-test
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import tempfile


METHOD = "ours_30000"
METRICS = ("PSNR", "SSIM", "LPIPS")
VIEWS = {f"{i:05d}.png" for i in range(13)}
SIZE = (1600, 1175)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value, label):
    require(type(value) in (int, float) and math.isfinite(value),
            f"{label}: expected a finite number, got {value!r}")
    return value


def read_metrics(path):
    data = json.loads(path.read_text())
    require(isinstance(data, dict) and isinstance(data.get(METHOD), dict),
            f"{path}: missing {METHOD} object")
    return data[METHOD]


def check_images(directory):
    names = {p.name for p in directory.iterdir() if p.is_file()}
    require(names == VIEWS, f"{directory}: expected exactly 13 PNGs 00000–00012; "
            f"missing={sorted(VIEWS - names)}, extra={sorted(names - VIEWS)}")
    for name in sorted(names):
        path = directory / name
        with path.open("rb") as handle:
            header = handle.read(24)
        require(len(header) == 24 and header[:8] == b"\x89PNG\r\n\x1a\n"
                and header[12:16] == b"IHDR", f"{path}: invalid PNG header")
        require(struct.unpack(">II", header[16:24]) == SIZE,
                f"{path}: expected image dimensions {SIZE}")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def vertex_count(path):
    with path.open("rb") as handle:
        require(handle.readline(1024).strip() == b"ply", f"{path}: invalid PLY")
        count = None
        for _ in range(1024):
            line = handle.readline(4096).strip()
            if line.startswith(b"element vertex "):
                count = int(line.split()[2])
            if line == b"end_header":
                require(count is not None and count > 0,
                        f"{path}: missing or non-positive vertex count")
                return count
            require(bool(line), f"{path}: truncated PLY header")
    raise ValueError(f"{path}: PLY header exceeds bounded scan")


def summarize(root):
    runs = {}
    gt_hashes = {}
    for name in ("3dgs_original", "wm_v2"):
        model = root / name
        aggregate = read_metrics(model / "results.json")
        per_view = read_metrics(model / "per_view.json")
        runs[name] = {metric: finite(aggregate.get(metric), f"{name}/{metric}")
                      for metric in METRICS}
        for metric in METRICS:
            values = per_view.get(metric)
            require(isinstance(values, dict) and set(values) == VIEWS,
                    f"{model}/per_view.json: {metric} must contain exactly the 13 test views")
            for view, value in values.items():
                finite(value, f"{name}/{metric}/{view}")
        for kind in ("renders", "gt"):
            check_images(model / "test" / METHOD / kind)
        gt_hashes[name] = {view: sha256(model / "test" / METHOD / "gt" / view)
                           for view in sorted(VIEWS)}
        runs[name]["gaussians"] = vertex_count(
            model / "point_cloud" / "iteration_30000" / "point_cloud.ply")
    require(gt_hashes["3dgs_original"] == gt_hashes["wm_v2"],
            "Ground-truth SHA-256 mismatch between 3dgs_original and wm_v2")

    delta = {metric: runs["wm_v2"][metric] - runs["3dgs_original"][metric]
             for metric in (*METRICS, "gaussians")}
    report = {
        "run_root": str(root.resolve()),
        "protocol": {
            "scene": "datasetOnline Scene 1 high area1", "train_views": 86,
            "test_views": 13, "resolution": list(SIZE), "iterations": 30000,
            "baseline": "Same modified 3DGS codebase, with weight-map features disabled",
            "wm_v2": "COLMAP seed lookup includes image names ending in .jpg (fix applied)",
            "note": "Protocol declarations describe this experiment; artifact checks are listed separately.",
        },
        "validated": {"finite_aggregate_and_per_view_metrics": True,
                      "test_images_per_directory": 13, "png_dimensions": list(SIZE),
                      "gt_sha256_identical": True, "ply_vertex_headers": True},
        "results": runs, "delta_wm_v2_minus_baseline": delta,
        "gt_sha256": gt_hashes["3dgs_original"],
    }
    lines = [
        "# datasetOnline Scene 1 high area1 · 30k 对比", "",
        "实验设定：86 张训练图 / 13 张测试图，1600×1175，固定训练 30,000 步。",
        "基线：同一份改进版 3DGS 底座关闭权重相关功能（3dgs_original）；并非另行复现未经修改的官方仓库。",
        "wm_v2：已修复带 `.jpg` 图像名的 COLMAP 种子查找。以上为本次实验约定；脚本校验范围见下文。", "",
        "| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | 高斯数量 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, values in runs.items():
        lines.append(f"| {name} | {values['PSNR']:.6f} | {values['SSIM']:.6f} | "
                     f"{values['LPIPS']:.6f} | {values['gaussians']:,} |")
    lines.extend([
        f"| wm_v2 − baseline | {delta['PSNR']:+.6f} | {delta['SSIM']:+.6f} | "
        f"{delta['LPIPS']:+.6f} | {delta['gaussians']:+,} |", "",
        "校验通过：两组 ours_30000 的整体和逐帧指标均为有限数；逐帧指标各包含相同的 13 帧；",
        "两组 renders 和 gt 各有 13 张 1600×1175 PNG；两组 GT 逐文件 SHA-256 完全一致；",
        "高斯数量读取自 iteration_30000/point_cloud.ply 头部，未加载点云主体。", "",
        "此汇总不重算指标，也不以数值变化替代渲染图的视觉检查。", "",
    ])
    (root / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False,
                                                allow_nan=False) + "\n")
    (root / "SUMMARY.md").write_text("\n".join(lines))
    return report


def self_test():
    """Synthetic header-only fixtures live exclusively in a temporary directory."""
    with tempfile.TemporaryDirectory(prefix="synthetic-3dgs-summary-test-") as temp:
        root = Path(temp)
        for name in ("3dgs_original", "wm_v2"):
            model = root / name
            for kind in ("renders", "gt"):
                folder = model / "test" / METHOD / kind
                folder.mkdir(parents=True)
                for view in VIEWS:
                    (folder / view).write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\x0dIHDR"
                                               + struct.pack(">II", *SIZE))
            (model / "results.json").write_text(json.dumps({METHOD: dict.fromkeys(METRICS, 1.0)}))
            (model / "per_view.json").write_text(json.dumps({METHOD: {
                metric: dict.fromkeys(sorted(VIEWS), 1.0) for metric in METRICS}}))
            ply = model / "point_cloud" / "iteration_30000" / "point_cloud.ply"
            ply.parent.mkdir(parents=True)
            ply.write_bytes(b"ply\nformat binary_little_endian 1.0\nelement vertex 2\nend_header\n")
        assert summarize(root)["validated"]["gt_sha256_identical"]
        assert (root / "summary.json").is_file() and (root / "SUMMARY.md").is_file()
        (root / "wm_v2" / "test" / METHOD / "gt" / "00000.png").unlink()
        try:
            summarize(root)
        except ValueError as error:
            assert "missing=['00000.png']" in str(error), str(error)
        else:
            raise AssertionError("Missing test image was accepted")
    print("PASS: synthetic complete run accepted; missing test image rejected; temporary fixtures removed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path, nargs="?")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif args.run_root is None:
        parser.error("run_root is required unless --self-test is used")
    else:
        try:
            summarize(args.run_root)
        except (OSError, ValueError, TypeError) as error:
            print(f"Validation failed: {error}", file=sys.stderr)
            sys.exit(1)
        print(f"Validated: {args.run_root / 'summary.json'}; {args.run_root / 'SUMMARY.md'}")
