"""MIDV-500 문서 1종(01_alb_id)을 YOLO 형식 소형 데이터셋으로 변환한다.

- 입력: 01_alb_id.zip (MIDV-500, ftp://smartengines.com/midv-500/dataset/)
- 출력: <out>/images/{train,val}/*.jpg, <out>/labels/{train,val}/*.txt, <out>/data.yaml
- 분할: 촬영 조건(클립) 단위. 프레임 누수 방지를 위해 val 클립은 train에 섞지 않는다.
- 이미지: 긴 변 640px JPEG(quality 90)로 축소. 라벨은 quad(4점)의 외접 사각형, 이미지 경계로 자름.
"""
import argparse
import json
import sys
import zipfile
from io import BytesIO
from pathlib import Path

from PIL import Image

VAL_CLIPS = ("KS", "TS")  # 2개 조건은 검증 전용
FRAME_STEP = 2  # 클립당 30프레임 중 짝수 번째만 → 15장
MAX_SIDE = 640


def quad_to_yolo(quad, w, h):
    xs = [min(max(p[0], 0), w) for p in quad]
    ys = [min(max(p[1], 0), h) for p in quad]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return ((x0 + x1) / 2 / w, (y0 + y1) / 2 / h, (x1 - x0) / w, (y1 - y0) / h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    counts = {"train": 0, "val": 0, "skipped": 0}
    with zipfile.ZipFile(args.zip) as zf:
        names = set(zf.namelist())
        tifs = sorted(n for n in names if n.endswith(".tif") and "/images/" in n and n.count("/") >= 3)
        for name in tifs:
            stem = Path(name).stem  # e.g. TS01_05
            clip, frame = stem[:2], int(stem.split("_")[1])
            if frame % FRAME_STEP:
                continue
            gt_name = name.replace("/images/", "/ground_truth/").replace(".tif", ".json")
            if gt_name not in names:
                counts["skipped"] += 1
                continue
            quad = json.loads(zf.read(gt_name))["quad"]
            img = Image.open(BytesIO(zf.read(name))).convert("RGB")
            w, h = img.size
            box = quad_to_yolo(quad, w, h)
            if box is None:
                counts["skipped"] += 1
                continue
            split = "val" if clip in VAL_CLIPS else "train"
            scale = MAX_SIDE / max(w, h)
            img = img.resize((round(w * scale), round(h * scale)), Image.BILINEAR)
            (out / "images" / split).mkdir(parents=True, exist_ok=True)
            (out / "labels" / split).mkdir(parents=True, exist_ok=True)
            img.save(out / "images" / split / f"{stem}.jpg", quality=90)
            (out / "labels" / split / f"{stem}.txt").write_text("0 " + " ".join(f"{v:.6f}" for v in box) + "\n")
            counts[split] += 1
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: id_card\n"
    )
    print(json.dumps(counts))
    if counts["train"] == 0 or counts["val"] == 0:
        sys.exit("empty split: zip 구조 확인 필요")


if __name__ == "__main__":
    main()
