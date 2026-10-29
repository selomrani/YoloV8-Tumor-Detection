import os
import cv2
import shutil
from pathlib import Path


def list_images_and_labels(root):
    """List all image and label files under root."""
    root = Path(root)
    images = []
    labels = []
    for p in root.rglob('*'):
        if p.is_file():
            ext = p.suffix.lower()
            if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                images.append(p)
            elif ext == '.txt':
                labels.append(p)
    return images, labels


def is_valid_image(path):
    try:
        img = cv2.imread(str(path))
        if img is None:
            return False
        return True
    except Exception:
        return False


def validate_yolo_label(path, nc=4):
    """Validate a yolo label file. Return list of valid lines (as floats/ints)."""
    try:
        with open(path, 'r') as f:
            lines = [l.strip() for l in f if l.strip() != '']
    except Exception:
        return None

    if len(lines) == 0:
        # empty file means no tumor - valid as negative
        return []

    valid = []
    for l in lines:
        parts = l.split()
        if len(parts) != 5:
            return None
        try:
            cls = int(float(parts[0]))
            x = float(parts[1])
            y = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
        except Exception:
            return None
        if cls < 0 or cls >= nc:
            return None
        if x < 0 or x > 1 or y < 0 or y > 1 or w < 0 or w > 1 or h < 0 or h > 1:
            return None
        valid.append((cls, x, y, w, h))
    return valid


def pair_data(raw_root, nc=4):
    """Pair images with labels, validate. Return list of dicts."""
    images, labels = list_images_and_labels(raw_root)
    label_map = {l.stem: l for l in labels}
    pairs = []
    for img in images:
        lab = label_map.get(img.stem)
        if lab is None:
            continue  # no label file
        if not is_valid_image(img):
            continue
        v = validate_yolo_label(lab, nc=nc)
        if v is None:
            continue
        # determine class for stratify
        if len(v) == 0:
            cls_name = 'No Tumor'
        else:
            # take first class if multiple? usually single; just use it
            cls_name = None
            # map will be done by caller; store raw class
            cls_name = str(v[0][0])
        pairs.append({
            'image_path': str(img),
            'label_path': str(lab),
            'cls': cls_name,
            'n_boxes': len(v)
        })
    return pairs


def copy_split(pairs_df, out_root):
    out_root = Path(out_root)
    for _, row in pairs_df.iterrows():
        split = row['split']
        img_src = Path(row['image_path'])
        lbl_src = Path(row['label_path'])
        img_dst = out_root / 'images' / split / img_src.name
        lbl_dst = out_root / 'labels' / split / lbl_src.name
        img_dst.parent.mkdir(parents=True, exist_ok=True)
        lbl_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img_src, img_dst)
        shutil.copy2(lbl_src, lbl_dst)
