import os
import glob
import time
import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

# Code gotten from ReadMe in the repo, formatted to test against images uploaded to web demo

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = (
    torch.bfloat16
    if (device == "cuda" and torch.cuda.get_device_capability()[0] >= 8)
    else torch.float16
)


model = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()

folders = [
    "input_images_20260226_232517_916858/images",
    "input_images_20260227_050615_955653/images",
    "input_images_20260227_072349_688065/images",
]

for folder in folders:
    image_paths = sorted(glob.glob(os.path.join(folder, "*.png")))

    if len(image_paths) == 0:
        print(f"\n No PNG images found in {folder}")
        continue

    print(f"\n Folder: {folder}")
    print(f" Number of images: {len(image_paths)}")

    images = load_and_preprocess_images(image_paths).to(device)

    # Warmup
    with torch.no_grad():
        if device == "cuda":
            with torch.cuda.amp.autocast(dtype=dtype):
                _ = model(images)
            torch.cuda.synchronize()
        else:
            _ = model(images)

    start = time.time()

    with torch.no_grad():
        if device == "cuda":
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = model(images)
            torch.cuda.synchronize()
        else:
            predictions = model(images)

    end = time.time()

    print(f"Total inference time: {end - start:.4f} seconds")
