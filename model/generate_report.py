# generate_report.py

import os
import re
import shutil
import pandas as pd
import matplotlib.pyplot as plt

# === EDIT THESE IF YOUR RUN FOLDERS DIFFER ===
TRAIN_RUN  = 'runs/train/uap_experiment2'
DETECT_RUN = 'runs/detect/val2'
REPORT_DIR = 'report'
# ============================================

os.makedirs(REPORT_DIR, exist_ok=True)

# 1) Load the CSV
csv_path = os.path.join(TRAIN_RUN, 'results.csv')
if not os.path.isfile(csv_path):
    raise FileNotFoundError(f"Cannot find {csv_path}")
df = pd.read_csv(csv_path)

# 2) Show what columns you have
print("Columns in results.csv:", df.columns.tolist())

# 3) Identify epoch, loss, and mAP columns
cols = df.columns.tolist()
epoch_col = next((c for c in cols if 'epoch' in c.lower()), cols[0])
loss_cols  = [c for c in cols if 'loss' in c.lower()]
map_cols   = [c for c in cols if re.search(r'map', c, re.IGNORECASE)]

if not loss_cols:
    print("⚠️ No loss columns found!")
if not map_cols:
    print("⚠️ No mAP columns found!")

# 4) Plot training losses
if loss_cols:
    plt.figure()
    for c in loss_cols:
        plt.plot(df[epoch_col], df[c], label=c)
    plt.xlabel(epoch_col)
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, 'training_losses.png'))
    plt.close()

# 5) Plot validation mAP
if map_cols:
    plt.figure()
    for c in map_cols:
        plt.plot(df[epoch_col], df[c], label=c)
    plt.xlabel(epoch_col)
    plt.ylabel('mAP')
    plt.title('Validation mAP over Epochs')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, 'val_map.png'))
    plt.close()

# 6) Copy confusion matrix & PR curve
for fname in ('confusion_matrix.png', 'PR_curve.png'):
    src = os.path.join(DETECT_RUN, fname)
    dst = os.path.join(REPORT_DIR, fname)
    if os.path.isfile(src):
        shutil.copy(src, dst)

# 7) Copy a few sample detection images
sample_src = os.path.join(DETECT_RUN, 'images')
sample_dst = os.path.join(REPORT_DIR, 'samples')
if os.path.isdir(sample_src):
    os.makedirs(sample_dst, exist_ok=True)
    imgs = sorted(os.listdir(sample_src))
    for img in imgs[:5]:
        shutil.copy(os.path.join(sample_src, img),
                    os.path.join(sample_dst, img))

print(f"\n✅ Report assets saved to ./{REPORT_DIR}/")
