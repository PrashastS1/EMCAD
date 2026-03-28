"""
prepare_drishti.py

One-time script to reorganize the raw DRISHTI-GS dataset into the
folder structure expected by EMCAD's existing dataloader_polyp.py.

Actual DRISHTI-GS structure (from directory scan):
    DRISHTI-GS/
        Training-20211018T055246Z-001/
            Training/
                Images/
                    GLAUCOMA/   <- .png images
                    NORMAL/     <- .png images
                GT/
                    drishtiGS_XXX/
                        SoftMap/
                            drishtiGS_XXX_ODsegSoftmap.png
        Test-20211018T060000Z-001/
            Test/
                Images/
                    glaucoma/   <- .png images
                    normal/     <- .png images
                Test_GT/
                    drishtiGS_XXX/
                        SoftMap/
                            drishtiGS_XXX_ODsegSoftmap.png

Output structure (EMCAD-compatible):
    data/drishti/
        train/
            images/
            masks/
        val/
            images/
            masks/
        test/
            images/
            masks/

Usage:
    python prepare_drishti.py \
        --drishti_root ./DRISHTI-GS \
        --output_root ./data/drishti \
        --val_split 0.2 \
        --seed 42
"""

import os
import random
import argparse
import numpy as np
from PIL import Image
from glob import glob


def find_dir(drishti_root, keyword):
    """Find a directory containing keyword anywhere in its path."""
    for root, dirs, _ in os.walk(drishti_root):
        for d in dirs:
            if d == keyword:
                return os.path.join(root, d)
    return None


def get_all_images_in_dir(images_dir):
    """
    Collect all .png/.jpg images recursively under images_dir.
    Handles subdirectories like GLAUCOMA/, NORMAL/, glaucoma/, normal/
    """
    files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        files.extend(glob(os.path.join(images_dir, '**', ext), recursive=True))
    return sorted(files)


def get_image_mask_pairs(images_dir, gt_dir):
    """
    Match images to their OD softmap masks.
    Image filename example: drishtiGS_002.png
    Mask path example: GT/drishtiGS_002/SoftMap/drishtiGS_002_ODsegSoftmap.png
    """
    image_files = get_all_images_in_dir(images_dir)

    pairs = []
    missing_masks = []

    for img_path in image_files:
        basename = os.path.splitext(os.path.basename(img_path))[0]  # drishtiGS_002

        mask_path = os.path.join(
            gt_dir, basename, 'SoftMap', f'{basename}_ODsegSoftmap.png'
        )

        if os.path.exists(mask_path):
            pairs.append((img_path, mask_path))
        else:
            missing_masks.append(basename)

    if missing_masks:
        print(f"  WARNING: No OD softmap found for: {missing_masks}")

    return pairs


def softmap_to_binary_mask(softmap_path, threshold=0.5):
    """
    Convert DRISHTI soft probability map to binary mask.
    Pixel values 0-255, threshold at 127.
    Returns PIL Image with values 0 or 255.
    """
    img = Image.open(softmap_path).convert('L')
    arr = np.array(img)
    binary = (arr >= int(threshold * 255)).astype(np.uint8) * 255
    return Image.fromarray(binary)


def copy_pairs_to_split(pairs, output_split_dir):
    """
    Write image/mask pairs into output_split_dir/images/ and masks/.
    Masks are binarized from softmaps.
    """
    images_out = os.path.join(output_split_dir, 'images')
    masks_out = os.path.join(output_split_dir, 'masks')
    os.makedirs(images_out, exist_ok=True)
    os.makedirs(masks_out, exist_ok=True)

    for img_path, mask_path in pairs:
        basename = os.path.splitext(os.path.basename(img_path))[0]

        # Save image as RGB PNG
        dest_img = os.path.join(images_out, f'{basename}.png')
        Image.open(img_path).convert('RGB').save(dest_img)

        # Save binarized mask
        dest_mask = os.path.join(masks_out, f'{basename}.png')
        softmap_to_binary_mask(mask_path, threshold=0.5).save(dest_mask)

    return len(pairs)


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"\n{'='*55}")
    print(f"DRISHTI-GS Dataset Preparation for EMCAD")
    print(f"{'='*55}")
    print(f"Input  : {args.drishti_root}")
    print(f"Output : {args.output_root}")
    print(f"Val    : {args.val_split*100:.0f}% of training set")
    print(f"Seed   : {args.seed}")
    print(f"{'='*55}\n")

    # ── Locate Training folders ─────────────────────────────────────────────
    print("Step 1: Locating Training and Test directories...")

    train_images_dir = find_dir(args.drishti_root, 'Images')
    if train_images_dir is None:
        raise FileNotFoundError("Could not find 'Images' folder in Training split.")

    # Training GT is sibling of Images
    train_gt_dir = os.path.join(os.path.dirname(train_images_dir), 'GT')
    if not os.path.exists(train_gt_dir):
        raise FileNotFoundError(f"Could not find Training GT dir at: {train_gt_dir}")

    print(f"  Training Images : {train_images_dir}")
    print(f"  Training GT     : {train_gt_dir}")

    # ── Locate Test folders ─────────────────────────────────────────────────
    # Test images are under Test/Images/, GT is under Test/Test_GT/
    test_base = find_dir(args.drishti_root, 'Test')
    if test_base is None:
        raise FileNotFoundError("Could not find 'Test' folder.")

    test_images_dir = os.path.join(test_base, 'Images')
    test_gt_dir = os.path.join(test_base, 'Test_GT')

    if not os.path.exists(test_images_dir):
        raise FileNotFoundError(f"Could not find Test Images dir at: {test_images_dir}")
    if not os.path.exists(test_gt_dir):
        raise FileNotFoundError(f"Could not find Test GT dir at: {test_gt_dir}")

    print(f"  Test Images     : {test_images_dir}")
    print(f"  Test GT         : {test_gt_dir}")

    # ── Collect pairs ───────────────────────────────────────────────────────
    print("\nStep 2: Collecting image-mask pairs...")

    train_val_pairs = get_image_mask_pairs(train_images_dir, train_gt_dir)
    test_pairs = get_image_mask_pairs(test_images_dir, test_gt_dir)

    print(f"  Train+Val images : {len(train_val_pairs)}")
    print(f"  Test images      : {len(test_pairs)}")

    if len(train_val_pairs) == 0:
        raise RuntimeError(
            "No training pairs found. Something is wrong with the directory structure."
        )

    # ── Train / Val split ───────────────────────────────────────────────────
    print(f"\nStep 3: Splitting train -> train/val "
          f"({(1-args.val_split)*100:.0f}/{args.val_split*100:.0f})...")

    random.shuffle(train_val_pairs)
    n_val = max(1, int(len(train_val_pairs) * args.val_split))
    train_pairs = train_val_pairs[:-n_val]
    val_pairs   = train_val_pairs[-n_val:]

    print(f"  Train : {len(train_pairs)} images")
    print(f"  Val   : {len(val_pairs)} images")
    print(f"  Test  : {len(test_pairs)} images")

    # ── Write output ────────────────────────────────────────────────────────
    print(f"\nStep 4: Writing to {args.output_root} ...")

    for split_name, pairs in [('train', train_pairs),
                               ('val',   val_pairs),
                               ('test',  test_pairs)]:
        out_dir = os.path.join(args.output_root, split_name)
        n = copy_pairs_to_split(pairs, out_dir)
        print(f"  [{split_name:5s}] {n} pairs written -> {out_dir}")

    # ── Verify ──────────────────────────────────────────────────────────────
    print(f"\nStep 5: Verifying output...")
    all_ok = True
    for split_name in ['train', 'val', 'test']:
        img_dir = os.path.join(args.output_root, split_name, 'images')
        msk_dir = os.path.join(args.output_root, split_name, 'masks')
        imgs = sorted(os.listdir(img_dir))
        msks = sorted(os.listdir(msk_dir))

        # Check counts match
        count_ok = len(imgs) == len(msks)
        # Check filenames match
        names_ok = [os.path.splitext(i)[0] for i in imgs] == \
                   [os.path.splitext(m)[0] for m in msks]

        status = "OK" if (count_ok and names_ok) else "MISMATCH"
        print(f"  [{split_name:5s}] images: {len(imgs)}, "
              f"masks: {len(msks)} [{status}]")
        if not (count_ok and names_ok):
            all_ok = False

    if all_ok:
        print(f"\n{'='*55}")
        print("SUCCESS: Dataset is ready.")
        print(f"\nRun training with:")
        print(f"  python train_drishti.py \\")
        print(f"    --encoder pvt_v2_b2 \\")
        print(f"    --train_path {args.output_root}/train \\")
        print(f"    --test_path  {args.output_root} \\")
        print(f"    --batchsize 4 --epoch 200")
        print(f"{'='*55}\n")
    else:
        print("\nERROR: Mismatch detected. Check warnings above.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare DRISHTI-GS for EMCAD')
    parser.add_argument('--drishti_root', type=str, required=True,
                        help='Path to raw DRISHTI-GS root folder')
    parser.add_argument('--output_root', type=str, default='./data/drishti',
                        help='Output directory for prepared dataset')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Fraction of train data to use as val (default: 0.2)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    args = parser.parse_args()
    main(args)