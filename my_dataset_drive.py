import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import cv2


class DriveDataset(Dataset):
    def __init__(self, root: str, train: bool, transforms=None):
        super(DriveDataset, self).__init__()
        self.flag = "training" if train else "validate"

        if self.flag == 'training':
            data_root = os.path.join(root, "DRIVE", self.flag)
        else:
            data_root = os.path.join(root, "DRIVE", self.flag)

        assert os.path.exists(data_root), f"path '{data_root}' does not exists."
        self.transforms = transforms
        img_names = [i for i in os.listdir(os.path.join(data_root, "images")) if i.endswith(".tif")]
        manual_names = [i for i in os.listdir(os.path.join(data_root, "manual")) if i.endswith(".gif")]
        mask_names = [i for i in os.listdir(os.path.join(data_root, "mask")) if i.endswith(".gif")]

        self.img_list = [os.path.join(data_root, "images", i) for i in img_names]
        self.manual = [os.path.join(data_root, "manual", i) for i in manual_names]
        self.roi_mask = [os.path.join(data_root, "mask", i) for i in mask_names]


    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx]).convert('RGB')
        manual = Image.open(self.manual[idx]).convert('L')
        manual = np.array(manual) / 255
        roi_mask = Image.open(self.roi_mask[idx]).convert('L')
        roi_mask = 255 - np.array(roi_mask)
        mask = np.clip(manual + roi_mask, a_min=0, a_max=255)
        mask = Image.fromarray(mask)

        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        return img, mask

    def __len__(self):
        return len(self.img_list)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs

def Test_data(basic_path, idx):
    image_path = os.path.join(basic_path, "test/images/")
    image_name = [i for i in os.listdir(image_path) if i.endswith(".tif")]
    image_list = [os.path.join(image_path, i) for i in image_name]
    
    mask_path = os.path.join(basic_path, "test/mask/")
    mask_name = [i for i in os.listdir(mask_path) if i.endswith(".gif")]
    mask_list = [os.path.join(mask_path, i) for i in mask_name]
    
    manual_path = os.path.join(basic_path, "test/manual/")
    manual_name = [i for i in os.listdir(manual_path) if i.endswith(".gif")]
    manual_list = [os.path.join(manual_path, i) for i in manual_name]

    
    test_image = Image.open(image_list[idx]).convert('RGB')
    test_manual = Image.open(manual_list[idx]).convert('L')
    test_mask = Image.open(mask_list[idx]).convert('L')
    
    test_manual = np.array(test_manual) / 255
    test_manual = Image.fromarray(test_manual)
    
    name = image_list[idx]

    return test_image, test_manual, test_mask, name


if __name__ == '__main__':
    train_dataset = DriveDataset('./', train=True, transforms=None)
