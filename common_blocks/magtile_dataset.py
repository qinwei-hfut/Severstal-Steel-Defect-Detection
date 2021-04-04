from torchvision.datasets.vision import VisionDataset

from PIL import Image
import numpy

import os
import os.path
import sys

import torch
import torchvision.transforms as transforms

import pdb
import cv2

from albumentations import (HorizontalFlip, ShiftScaleRotate, Normalize, CropNonEmptyMaskIfExists, Resize, Compose,
                            RandomBrightnessContrast, VerticalFlip, RandomBrightness, RandomContrast)
from albumentations.pytorch import ToTensor






def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def is_image_file(filename):
    """Checks if a file is an allowed image extension.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)



def make_dataset(dir_image, class_to_idx, train, extensions=None, is_valid_file=None):
    # print('make')

    images = []
    img_list = []
    dir_image = os.path.expanduser(dir_image)
    if not ((extensions is None) ^ (is_valid_file is None)):
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)
    idx = 0

    # print(class_to_idx)
    for target in sorted(class_to_idx.keys()):

        
        idx = idx + 1

        d_image = os.path.join(dir_image, target)
        # print('-------------------------------------------------------------------------------')
        # print(target)
        if not os.path.isdir(d_image):
            continue
        for root, _, fnames in sorted(os.walk(d_image, followlinks=True)):
            fnames = sorted(fnames)
            if train:
                fnames = fnames[0:int(len(fnames)/2)]
            else:
                fnames = fnames[int(len(fnames)/2):]
            for fname in sorted(fnames):
                fname = fname.split('.')[0]

                # print(fname)
                image_path = os.path.join(d_image, fname)
                if image_path not in img_list:
                    item = (image_path, class_to_idx[target])
                    images.append(item)
                    img_list.append(image_path)

    return images


class MTDatasetFolder(VisionDataset):
    """A generic data loader where the samples are arranged in this way: ::
        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext
        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext
    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root_image, loader, extensions=None, transform=None,
                 target_transform=None, is_valid_file=None,train=True):
        super(MTDatasetFolder, self).__init__(root_image, transform=transform,
                                            target_transform=target_transform)
        self.root_image = root_image

        classes, class_to_idx = self._find_classes(self.root_image)
        samples = make_dataset(self.root_image, class_to_idx, train,extensions, is_valid_file)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
                                "Supported extensions are: " + ",".join(extensions)))

        self.loader = loader
        self.extensions = extensions
        self.train = train

        self.transforms_img_sal = self.get_transforms_img_sal()
        self.transforms_img = self.get_transforms_img()
        self.transforms_sal = self.get_transforms_sal()

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    def get_transforms_img_sal(self):
        list_transforms = []
        if self.train:
            list_transforms.extend(
                # [Resize(224,224),HorizontalFlip(p=0.5),VerticalFlip(p=0.5)]
                [HorizontalFlip(p=0.5),VerticalFlip(p=0.5)]
            )
        list_trfms = Compose(list_transforms)
        return list_trfms

    def get_transforms_img(self):
        list_transforms = []
        if self.train:
            list_transforms.extend([RandomBrightnessContrast(p=0.1,brightness_limit=0.1, contrast_limit=0.1)])

        list_transforms.extend([
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1),
                ToTensor()
            ])
        list_trfms = Compose(list_transforms)
        return list_trfms

    def get_transforms_sal(self):
        list_transforms = []
        list_transforms.extend([
                ToTensor()
            ])
        list_trfms = Compose(list_transforms)
        return list_trfms

    # def get_transforms(self):
    #     list_transforms = []
    #     if self.train:
    #         # if crop_image_size is not None:
    #         #     list_transforms.extend(
    #         #         [CropNonEmptyMaskIfExists(crop_image_size[0], crop_image_size[1], p=0.85),
    #         #         HorizontalFlip(p=0.5),
    #         #         VerticalFlip(p=0.5),
    #         #         RandomBrightnessContrast(p=0.1, brightness_limit=0.1, contrast_limit=0.1)
    #         #         ])
    #         # else:
    #         list_transforms.extend(
    #             [Resize(224,224),
    #             HorizontalFlip(p=0.5),
    #             VerticalFlip(p=0.5),
    #             RandomBrightnessContrast(p=0.1, brightness_limit=0.1, contrast_limit=0.1)
    #             ]
    #         )
    #     list_transforms.extend(
    #         [
    #             Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1),
    #             ToTensor()
    #         ]
    #     )
    #     list_trfms = Compose(list_transforms)
    #     return list_trfms

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.
        Args:
            dir (string): Root directory path.
        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        image_path, target = self.samples[index]
        input_image_path = image_path+'.jpg'
        mask_image_path = image_path+'.png'
        img = cv2.imread(input_image_path)

        mask_img = cv2.imread(mask_image_path,flags=cv2.IMREAD_GRAYSCALE)
        mask_img = mask_img / 255.0
        # mask_img_2 = cv2.imread(mask_image_path)

        # mask_img_pil = self.pil_L_loader(mask_image_path)
        # mask_img = numpy.array(mask_img_pil)
        # mask_img = numpy.expand_dims(mask_img,axis=2)

        pdb.set_trace()
        # augmented = self.transforms(image=img, mask=mask_img_pil)
        # augmented = self.transforms(image=img, mask=mask_img)
        
        augmented = self.transforms_img_sal(image=img,mask=mask_img)
        img = augmented['image']
        mask_img = augmented['mask']
        # pdb.set_trace()
        augmented = self.transforms_img(image=img)
        img = augmented['image']
        augmented = self.transforms_sal(mask=mask_img)
        mask_img = augmented['mask']
        pdb.set_trace()
        target_mask_img = mask.expand(3,mask.size(1),mask.size(2))  
        pdb.set_trace()
        mask = mask[0].permute(2, 0, 1)  
        mask_layer = mask[0]
        final_mask = torch.zeros((5,mask_layer.shape[0],mask_layer.shape[1]),device='cpu')
        if target != 4:
            if target != 5:
                final_mask[target] = mask_layer
            elif target == 5:
                final_mask[target-1] = mask_layer
            


        if self.target_transform is not None:
            target = self.target_transform(target)

    
        # pdb.set_trace()
        return img, final_mask,mask

    def __len__(self):
        return len(self.samples)

    def pil_1_loader(self,path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('1')

    def pil_L_loader(self,path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp','.xml')


# transform_val_img = transforms.Compose([
#             transforms.Resize(256),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             normalize,
#         ])

#     transform_val_sal = transforms.Compose([
#             transforms.Resize(256),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#         ])

# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                         std=[0.229, 0.224, 0.225])

#     transform_train_img = transforms.Compose([
#             transforms.RandomResizedCrop(224),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             normalize,
#         ])
        
#     transform_train_sal = transforms.Compose([
#             transforms.RandomResizedCrop(224),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#         ])


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def pil_L_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L')

def pil_1_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('1')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class MTImageFolder(MTDatasetFolder):
    """A generic data loader where the images are arranged in this way: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root_image, transform=None, target_transform=None,
                 loader=default_loader, is_valid_file=None, train=True):
        super(MTImageFolder, self).__init__(root_image, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file, train=True)
        self.imgs = self.samples