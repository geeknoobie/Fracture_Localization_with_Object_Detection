# prepare_fracatlas.py
import argparse, os, shutil, random, glob, pathlib

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def copy(src, dst):
    ensure_dir(os.path.dirname(dst))
    shutil.copy(src, dst)

def main(root, split):
    # 1) source paths
    pos_imgs = glob.glob(os.path.join(root, "images", "Fractured", "*"))
    neg_imgs = glob.glob(os.path.join(root, "images", "Non_fractured", "*"))
    yolo_src = os.path.join(root, "Annotations", "YOLO")  # .txt for positives

    # 2) shuffle & split
    all_imgs = pos_imgs + neg_imgs
    random.shuffle(all_imgs)
    names   = ["train", "val", "test"]
    counts  = [int(len(all_imgs)*p) for p in split]
    counts[-1] = len(all_imgs) - sum(counts[:-1])  # adjust for rounding

    # 3) output roots
    img_out = pathlib.Path(root) / "dataset" / "images"
    lbl_out = pathlib.Path(root) / "dataset" / "labels"

    idx = 0
    for name, n in zip(names, counts):
        for img_path in all_imgs[idx: idx + n]:
            fname = os.path.basename(img_path)
            stem  = os.path.splitext(fname)[0]

            # copy image
            copy(img_path, img_out / name / fname)

            # determine label source
            txt_src = os.path.join(yolo_src, stem + ".txt")
            txt_dst = lbl_out / name / (stem + ".txt")

            # case 1: fractured ⇒ copy existing label
            if os.path.exists(txt_src):
                copy(txt_src, txt_dst)
            # case 2: non‑fractured ⇒ create empty label
            else:
                ensure_dir(txt_dst.parent)
                open(txt_dst, "w").close()
        idx += n

    print(f"✅  Finished: YOLO‑ready dataset is in {root}/dataset/")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="FracAtlas",
                   help="Path to FracAtlas folder")
    p.add_argument("--split", nargs=3, type=float, default=[0.8, 0.1, 0.1],
                   metavar=("TRAIN", "VAL", "TEST"),
                   help="Fractions for train/val/test")
    main(**vars(p.parse_args()))
