"""원본 split 코드(practical-mlops/object-detection-project data_prep_pipeline.py:56-78) 재현:
images·labels를 따로 glob하고 정렬 없이 train_test_split(random_state=42)에 넣는다.

확인할 것
1. 이 파일시스템(APFS)에서 glob 순서가 이름 정렬 순서와 같은가
2. images 목록과 labels 목록이 같은 순서로 나오는가(어긋나면 이미지-라벨 쌍이 뒤섞인다)
3. 입력 순서만 바꿔도(같은 random_state) 분할 멤버가 달라지는가
"""
import argparse
import glob
import json
import os
import random
import tempfile
import zipfile
from pathlib import Path

from sklearn.model_selection import train_test_split


def book_split(images, labels, rs=42):
    x_tr, x_te, y_tr, y_te = train_test_split(images, labels, test_size=1 - 0.75, random_state=rs)
    x_va, x_te, y_va, y_te = train_test_split(x_te, y_te, test_size=0.10 / 0.25, random_state=rs)
    return (x_tr, x_va, x_te), (y_tr, y_va, y_te)


def stem(p):
    return Path(p).stem


def mismatched_pairs(xs, ys):
    return sum(stem(a) != stem(b) for a, b in zip(xs, ys))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, help="MIDV-500 zip (파일 이름만 사용)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    with zipfile.ZipFile(args.zip) as zf:
        stems = [Path(n).stem for n in zf.namelist() if n.endswith(".tif") and "/images/" in n and n.count("/") >= 3]
    res = {"n_files": len(stems)}
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(f"{d}/images"), os.makedirs(f"{d}/labels")
        shuffled = stems[:]
        random.Random(7).shuffle(shuffled)  # 생성 순서를 섞어 둔다(다운로드·압축해제 순서가 다를 수 있음)
        for s in shuffled:
            Path(f"{d}/images/{s}.tif").touch()
        for s in sorted(stems):
            Path(f"{d}/labels/{s}.txt").touch()
        images = list(glob.glob(os.path.join(d, "images", "**")))
        labels = list(glob.glob(os.path.join(d, "labels", "**")))
        res["glob_images_equals_sorted"] = [stem(p) for p in images] == sorted(stems)
        res["glob_labels_equals_sorted"] = [stem(p) for p in labels] == sorted(stems)
        res["glob_images_equals_creation_order"] = [stem(p) for p in images] == shuffled
        res["image_label_order_mismatch"] = mismatched_pairs(images, labels)
        (xi, _, _), (yi, _, _) = book_split(images, labels)
        res["book_split_train_pairs_mismatched"] = f"{mismatched_pairs(xi, yi)}/{len(xi)}"
        # 원본은 x(이미지)와 y(라벨)를 각각 split 디렉터리에 복사하고, YOLO는 파일 이름으로 라벨을 찾는다.
        (xs, ys) = book_split(images, labels)
        for nm, xx, yy in zip(("train", "val", "test"), xs, ys):
            res[f"book_{nm}_images_with_own_label_in_same_split"] = (
                f"{len({stem(p) for p in xx} & {stem(p) for p in yy})}/{len(xx)}")
        res["repeat_glob_same_order"] = images == list(glob.glob(os.path.join(d, "images", "**")))

    base = sorted(stems)
    (tr_a, va_a, te_a), _ = book_split(base, base)
    perm = base[:]
    random.Random(1).shuffle(perm)
    (tr_b, va_b, te_b), _ = book_split(perm, perm)
    res["same_order_same_split"] = book_split(base, base)[0] == (tr_a, va_a, te_a)
    res["perm_order_train_overlap"] = f"{len(set(tr_a) & set(tr_b))}/{len(tr_a)}"
    res["perm_order_val_overlap"] = f"{len(set(va_a) & set(va_b))}/{len(va_a)}"
    res["perm_order_test_overlap"] = f"{len(set(te_a) & set(te_b))}/{len(te_a)}"
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
