import os
import glob

from PIL import Image

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")


def is_image_file(filename: str) -> bool:
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def load_images_from_folder(folder, batch_size=1):
    image_paths = [img_path for img_path in glob.glob(os.path.join(folder, "*")) if is_image_file(img_path)]
    idxs = [i for i in range(len(image_paths))]
    
    for i in range(0, len(idxs), batch_size):
        idx_batch = idxs[i:i+batch_size]
        img_paths_batch = [image_paths[idx] for idx in idx_batch]
        imgs_names = [os.path.basename(img_path) for img_path in img_paths_batch]
        imgs_batch = [Image.open(img_path) for img_path in img_paths_batch]
        yield idx_batch, imgs_names, imgs_batch
    