import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import nms

from groundingdino.util.inference import load_model, load_image, predict


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

WORKER_COLOR_MAP = {
    "red": "worker_0",
    "green": "worker_1",
    "light_blue": "worker_2",
    "yellow": "worker_3",
    "dark_blue": "worker_4",
    "unknown": "unknown",
}


CSV_HEADER = [
    "image_path",
    "relative_path",
    "has_worker",
    "num_workers",
    "worker_index",
    "worker_id",
    "color",
    "det_score",
    "color_confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "phrase",
]


def scan_images(input_dir: Path):
    images = []
    for p in input_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(p)
    return sorted(images)


def get_relative_path(path: Path, input_dir: Path):
    try:
        return str(path.relative_to(input_dir))
    except Exception:
        return str(path)


def is_file_ready(path: Path, min_file_age: float = 1.0):
    """
    判断图片是否已经写入完成。
    如果文件刚生成，可能还没写完，所以要求最后修改时间距离当前至少 min_file_age 秒。
    """
    try:
        stat = path.stat()
        if stat.st_size <= 0:
            return False
        age = time.time() - stat.st_mtime
        return age >= min_file_age
    except FileNotFoundError:
        return False


def load_processed(record_path: Path):
    if not record_path.exists():
        return set()

    with open(record_path, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())


def append_processed(record_path: Path, relative_path: str):
    with open(record_path, "a", encoding="utf-8") as f:
        f.write(relative_path + "\n")


def cxcywh_to_xyxy(boxes, image_w, image_h):
    """
    GroundingDINO 输出的是归一化 cx, cy, w, h。
    这里转换成像素坐标 x1, y1, x2, y2。
    """
    boxes = boxes.clone()
    boxes[:, 0] = boxes[:, 0] * image_w
    boxes[:, 1] = boxes[:, 1] * image_h
    boxes[:, 2] = boxes[:, 2] * image_w
    boxes[:, 3] = boxes[:, 3] * image_h

    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    xyxy = torch.stack([x1, y1, x2, y2], dim=1)

    xyxy[:, 0].clamp_(0, image_w - 1)
    xyxy[:, 1].clamp_(0, image_h - 1)
    xyxy[:, 2].clamp_(0, image_w - 1)
    xyxy[:, 3].clamp_(0, image_h - 1)

    return xyxy


def crop_upper_body(image_rgb, box_xyxy):
    """
    根据人体框裁剪上半身区域，用于判断衣服颜色。
    """
    h, w = image_rgb.shape[:2]
    x1, y1, x2, y2 = box_xyxy

    x1 = int(max(0, min(w - 1, x1)))
    y1 = int(max(0, min(h - 1, y1)))
    x2 = int(max(0, min(w - 1, x2)))
    y2 = int(max(0, min(h - 1, y2)))

    if x2 <= x1 or y2 <= y1:
        return None

    bw = x2 - x1
    bh = y2 - y1

    # 只取人体框中间偏上的区域，尽量避开头部、背景和腿部
    cx1 = int(x1 + 0.20 * bw)
    cx2 = int(x1 + 0.80 * bw)
    cy1 = int(y1 + 0.20 * bh)
    cy2 = int(y1 + 0.65 * bh)

    if cx2 <= cx1 or cy2 <= cy1:
        return None

    return image_rgb[cy1:cy2, cx1:cx2]


def classify_color_hsv(crop_rgb):
    """
    用 HSV 颜色规则判断衣服颜色。
    返回:
        color, confidence
    """
    if crop_rgb is None or crop_rgb.size == 0:
        return "unknown", 0.0

    hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    valid = (S > 45) & (V > 45) & (V < 250)
    valid_count = int(valid.sum())

    if valid_count < 5:
        return "unknown", 0.0

    masks = {
        "red": valid & ((H <= 10) | (H >= 170)),
        "yellow": valid & (H > 20) & (H <= 38),
        "green": valid & (H > 38) & (H <= 85),
        "light_blue": valid & (
            ((H > 85) & (H <= 115) & (V >= 135))
            | ((H > 85) & (H <= 125) & (V >= 150) & (S <= 210))
        ),
        "dark_blue": valid & (
            ((H > 105) & (H <= 135) & (V < 170))
            | ((H > 115) & (H <= 145))
        ),
    }

    ratios = {
        color: float(mask.sum()) / float(valid_count)
        for color, mask in masks.items()
    }

    best_color = max(ratios, key=ratios.get)
    best_conf = ratios[best_color]

    # 蓝色兜底：浅蓝/深蓝有时 HSV 会有偏移
    blue_mask = valid & (H > 85) & (H <= 145)
    blue_ratio = float(blue_mask.sum()) / float(valid_count)

    if blue_ratio > 0.20 and best_conf < 0.22:
        mean_v = float(V[blue_mask].mean()) if blue_mask.sum() > 0 else 0.0
        if mean_v >= 145:
            return "light_blue", blue_ratio
        else:
            return "dark_blue", blue_ratio

    if best_conf < 0.18:
        return "unknown", 0.0

    return best_color, best_conf


def draw_annotations(image_rgb, detections, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    img = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]

        label = (
            f'{det["worker_id"]} | {det["color"]} | '
            f'det={det["det_score"]:.2f} | '
            f'color={det["color_confidence"]:.2f}'
        )

        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

        if font is not None:
            try:
                bbox = draw.textbbox((x1, y1), label, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
            except Exception:
                text_w, text_h = 300, 14

            text_bg = [x1, max(0, y1 - text_h - 4), x1 + text_w + 4, y1]
            draw.rectangle(text_bg, fill="red")
            draw.text((x1 + 2, max(0, y1 - text_h - 2)), label, fill="white", font=font)

    img.save(output_path)


def write_csv_rows(csv_path: Path, rows):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    need_header = not csv_path.exists()

    with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        if need_header:
            writer.writerow(CSV_HEADER)
        writer.writerows(rows)


def append_jsonl(jsonl_path: Path, obj):
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def make_results_json(jsonl_path: Path, json_path: Path):
    if not jsonl_path.exists():
        return

    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def process_one_image(
    model,
    image_path: Path,
    input_dir: Path,
    output_dir: Path,
    args,
    device,
):
    relative_path = get_relative_path(image_path, input_dir)

    image_source, image = load_image(str(image_path))

    h, w = image_source.shape[:2]

    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=args.prompt,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        device=device,
    )

    detections = []

    if boxes is not None and len(boxes) > 0:
        boxes_xyxy = cxcywh_to_xyxy(boxes.cpu(), w, h)
        scores = logits.cpu()

        keep = nms(boxes_xyxy, scores, args.nms_threshold)
        boxes_xyxy = boxes_xyxy[keep]
        scores = scores[keep]
        kept_phrases = [phrases[i] for i in keep.tolist()]

        for j, (box, score, phrase) in enumerate(zip(boxes_xyxy, scores, kept_phrases)):
            box_list = [float(v) for v in box.tolist()]

            torso = crop_upper_body(image_source, box_list)
            color, color_conf = classify_color_hsv(torso)
            worker_id = WORKER_COLOR_MAP.get(color, "unknown")

            detections.append({
                "worker_index": j,
                "bbox": box_list,
                "det_score": float(score.item()),
                "phrase": phrase,
                "color": color,
                "color_confidence": float(color_conf),
                "worker_id": worker_id,
            })

    annotated_path = ""

    if args.save_annotated:
        annotated_output_path = output_dir / "annotated" / relative_path
        draw_annotations(image_source, detections, annotated_output_path)
        annotated_path = str(annotated_output_path)

    csv_rows = []

    if len(detections) == 0:
        csv_rows.append([
            str(image_path),
            relative_path,
            False,
            0,
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
        ])
    else:
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            csv_rows.append([
                str(image_path),
                relative_path,
                True,
                len(detections),
                det["worker_index"],
                det["worker_id"],
                det["color"],
                det["det_score"],
                det["color_confidence"],
                x1,
                y1,
                x2,
                y2,
                det["phrase"],
            ])

    json_obj = {
        "image_path": str(image_path),
        "relative_path": relative_path,
        "has_worker": len(detections) > 0,
        "num_workers": len(detections),
        "detections": detections,
        "annotated_path": annotated_path,
    }

    return relative_path, csv_rows, json_obj


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)

    parser.add_argument("--prompt", type=str, default="person . worker . human .")
    parser.add_argument("--box_threshold", type=float, default=0.25)
    parser.add_argument("--text_threshold", type=float, default=0.25)
    parser.add_argument("--nms_threshold", type=float, default=0.50)

    parser.add_argument("--poll_interval", type=float, default=5.0)
    parser.add_argument("--max_empty_rounds", type=int, default=12)
    parser.add_argument("--min_file_age", type=float, default=1.0)

    parser.add_argument("--save_annotated", action="store_true")

    parser.add_argument(
        "--ignore_existing",
        action="store_true",
        help="启动时忽略当前已经存在的图片，只处理启动后新出现的图片",
    )

    parser.add_argument(
        "--cpu",
        action="store_true",
        help="强制使用 CPU，一般不建议",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "results.csv"
    jsonl_path = output_dir / "results.jsonl"
    json_path = output_dir / "results.json"
    processed_record_path = output_dir / "processed_images.txt"
    failed_record_path = output_dir / "failed_images.txt"

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] device: {device}")
    print(f"[INFO] input_dir: {input_dir}")
    print(f"[INFO] output_dir: {output_dir}")
    print(f"[INFO] poll_interval: {args.poll_interval}s")
    print(f"[INFO] max_empty_rounds: {args.max_empty_rounds}")
    print(f"[INFO] no-new timeout: {args.poll_interval * args.max_empty_rounds:.1f}s")

    print("[INFO] loading model...")
    model = load_model(args.config, args.weights, device=device)
    print("[INFO] model loaded.")

    processed = load_processed(processed_record_path)

    if args.ignore_existing:
        existing_images = scan_images(input_dir)
        for p in existing_images:
            rel = get_relative_path(p, input_dir)
            if rel not in processed:
                processed.add(rel)
                append_processed(processed_record_path, rel)
        print(f"[INFO] ignored existing images: {len(existing_images)}")

    empty_rounds = 0
    round_idx = 0

    while True:
        round_idx += 1

        all_images = scan_images(input_dir)

        pending_images = []
        not_ready_count = 0

        for image_path in all_images:
            rel = get_relative_path(image_path, input_dir)

            if rel in processed:
                continue

            if is_file_ready(image_path, args.min_file_age):
                pending_images.append(image_path)
            else:
                not_ready_count += 1

        if len(pending_images) == 0:
            if not_ready_count > 0:
                print(
                    f"[ROUND {round_idx}] no ready image, "
                    f"but {not_ready_count} image(s) may still be writing. wait..."
                )
            else:
                empty_rounds += 1
                print(
                    f"[ROUND {round_idx}] no new image. "
                    f"empty_rounds={empty_rounds}/{args.max_empty_rounds}"
                )

                if empty_rounds >= args.max_empty_rounds:
                    print("[DONE] reached max empty rounds. stop watching.")
                    break

            time.sleep(args.poll_interval)
            continue

        empty_rounds = 0

        print(f"[ROUND {round_idx}] found {len(pending_images)} new ready image(s).")

        for idx, image_path in enumerate(pending_images, start=1):
            rel = get_relative_path(image_path, input_dir)

            try:
                print(f"[PROCESS] {idx}/{len(pending_images)} {rel}")

                relative_path, csv_rows, json_obj = process_one_image(
                    model=model,
                    image_path=image_path,
                    input_dir=input_dir,
                    output_dir=output_dir,
                    args=args,
                    device=device,
                )

                write_csv_rows(csv_path, csv_rows)
                append_jsonl(jsonl_path, json_obj)

                processed.add(relative_path)
                append_processed(processed_record_path, relative_path)

            except Exception as e:
                print(f"[ERROR] failed to process {rel}: {repr(e)}")

                with open(failed_record_path, "a", encoding="utf-8") as f:
                    f.write(f"{rel}\t{repr(e)}\n")

                # 防止坏图导致程序一直重复处理同一张图
                processed.add(rel)
                append_processed(processed_record_path, rel)

        # 每轮处理完成后，同步生成普通 results.json
        make_results_json(jsonl_path, json_path)

        print(f"[INFO] current results saved to:")
        print(f"       {csv_path}")
        print(f"       {json_path}")
        print(f"       {jsonl_path}")

        time.sleep(args.poll_interval)

    make_results_json(jsonl_path, json_path)

    print("[FINAL OUTPUT]")
    print(f"CSV:  {csv_path}")
    print(f"JSON: {json_path}")
    print(f"JSONL:{jsonl_path}")
    print(f"Processed record: {processed_record_path}")
    print(f"Failed record:    {failed_record_path}")


if __name__ == "__main__":
    main()
