from PIL import Image
import numpy as np
mask = Image.open('./data/drishti/train/masks/drishtiGS_002.png')
arr = np.array(mask)
print(f"Unique values: {np.unique(arr)}")
print(f"Shape: {arr.shape}")
print(f"Foreground pixels: {(arr > 0).sum()}")