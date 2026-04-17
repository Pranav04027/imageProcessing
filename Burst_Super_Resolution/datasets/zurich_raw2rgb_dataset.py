import torch
import os
import cv2
import random
import numpy as np


class ZurichRAW2RGB(torch.utils.data.Dataset):
    """Canon RGB images from the "Zurich RAW to RGB mapping" dataset. You can download the full
    dataset (22 GB) from http://people.ee.ethz.ch/~ihnatova/pynet.html#dataset. Alternatively, you can only download the
    Canon RGB images (5.5 GB) from https://data.vision.ee.ethz.ch/bhatg/zurich-raw-to-rgb.zip
    """

    def __init__(self, root, split="train"):
        super().__init__()

        # Get absolute path
        root = os.path.abspath(root)

        if split in ["train", "test"]:
            # Check if original_images path exists (your dataset structure)
            orig_path = os.path.join(root, "original_images", "canon")
            if os.path.exists(orig_path):
                self.img_pth = orig_path
            else:
                self.img_pth = os.path.join(root, split, "canon")
        else:
            raise Exception("Unknown split {}".format(split))

        self.image_list = self._get_image_list(split)
        self.split = split

    def _get_image_list(self, split):
        import glob

        image_dir = os.path.join(self.img_pth)
        if os.path.exists(image_dir):
            jpg_files = glob.glob(os.path.join(image_dir, "*.jpg"))
            image_list = [os.path.basename(f) for f in jpg_files]
            image_list = sorted(
                image_list,
                key=lambda x: int(x.split(".")[0]) if x.split(".")[0].isdigit() else 0,
            )
        else:
            image_list = []
        return image_list

    def _get_image(self, im_id):
        path = os.path.join(self.img_pth, self.image_list[im_id])
        # Use Pillow for more robust loading
        from PIL import Image
        img_pil = Image.open(path).convert('RGB')
        img = np.array(img_pil)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # Keep BGR for consistency with original code
        
        if random.randint(0, 1) == 1 and self.split == "train":
            flag_aug = random.randint(1, 7)
            img = self.data_augmentation(img, flag_aug)
        else:
            img = img
        return img

    def get_image(self, im_id):
        frame = self._get_image(im_id)

        return frame

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        frame = self._get_image(index)

        return frame

    def data_augmentation(self, image, mode):
        """
        Performs data augmentation of the input image
        Input:
            image: a cv2 (OpenCV) image
            mode: int. Choice of transformation to apply to the image
                    0 - no transformation
                    1 - flip up and down
                    2 - rotate counterwise 90 degree
                    3 - rotate 90 degree and flip up and down
                    4 - rotate 180 degree
                    5 - rotate 180 degree and flip
                    6 - rotate 270 degree
                    7 - rotate 270 degree and flip
        """
        if mode == 0:
            # original
            out = image
        elif mode == 1:
            # flip up and down
            out = np.flipud(image)
        elif mode == 2:
            # rotate counterwise 90 degree
            out = np.rot90(image)
        elif mode == 3:
            # rotate 90 degree and flip up and down
            out = np.rot90(image)
            out = np.flipud(out)
        elif mode == 4:
            # rotate 180 degree
            out = np.rot90(image, k=2)
        elif mode == 5:
            # rotate 180 degree and flip
            out = np.rot90(image, k=2)
            out = np.flipud(out)
        elif mode == 6:
            # rotate 270 degree
            out = np.rot90(image, k=3)
        elif mode == 7:
            # rotate 270 degree and flip
            out = np.rot90(image, k=3)
            out = np.flipud(out)
        else:
            raise Exception("Invalid choice of image transformation")
        return out.copy()
