import os
import cv2
import random
import numpy as np
from PIL import Image
import nibabel as nib
import SimpleITK as sitk

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class HecktorDataset(Dataset):
    def __init__(self, site, base_path, transforms=None, split='train'):
        assert site in ['CHUM', 'CHUP', 'CHUS', 'CHUV', 'MDA', 'HGJ', 'HMR']

        self.split = split
        self.transforms = transforms

        sample_list = sorted(os.listdir(base_path))
        sample_list = [sample for sample in sample_list if site in sample]

        image_ct_list = [os.path.join(base_path, sample, sample + '_ct.nii.gz') for sample in sample_list]
        image_pt_list = [os.path.join(base_path, sample, sample + '_pt.nii.gz') for sample in sample_list]
        image_gt_list = [os.path.join(base_path, sample, sample + '_ct_gtvt.nii.gz') for sample in sample_list]

        train_len = int(len(sample_list) * 0.6)
        valid_len = int(len(sample_list) * 0.2)
        test_len = int(len(sample_list) * 0.2)

        if split == 'train':
            self.image_ct_list = image_ct_list[:train_len]
            self.image_pt_list = image_pt_list[:train_len]
            self.image_gt_list = image_gt_list[:train_len]
        elif split == 'valid':
            self.image_ct_list = image_ct_list[train_len:train_len + valid_len]
            self.image_pt_list = image_pt_list[train_len:train_len + valid_len]
            self.image_gt_list = image_gt_list[train_len:train_len + valid_len]
        else:
            self.image_ct_list = image_ct_list[-test_len:]
            self.image_pt_list = image_pt_list[-test_len:]
            self.image_gt_list = image_gt_list[-test_len:]

    def __len__(self):
        return len(self.image_ct_list)

    def __getitem__(self, idx):

        image_ct = self.image_ct_list[idx]
        image_pt = self.image_pt_list[idx]
        image_gt = self.image_gt_list[idx]

        image_ct = self.read_data(image_ct)
        image_pt = self.read_data(image_pt)
        image_gt = self.read_data(image_gt)

        sample = {
            'input': np.stack([image_ct, image_pt], axis=-1),
            'target': np.expand_dims(image_gt, axis=3)}

        if self.transforms:
            sample = self.transforms(sample)

        return sample['input'], sample['target']

    @staticmethod
    def read_data(path_to_nifti, return_numpy=True):
        """Read a NIfTI image. Return a numpy array (default) or `nibabel.nifti1.Nifti1Image` object"""
        if return_numpy:
            return nib.load(str(path_to_nifti)).get_fdata()
        return nib.load(str(path_to_nifti))


class Prostate(Dataset):
    def __init__(self, site, base_path, train_ratio=0.6, split='train', transform=None):
        channels = {'BIDMC': 3, 'HK': 3, 'I2CVB': 3, 'BMC': 3, 'RUNMC': 3, 'UCL': 3}
        assert site in list(channels.keys())
        self.split = split
        self.train_ratio = train_ratio

        images, labels = [], []
        sitedir = os.path.join(base_path, site)

        sample_list = sorted(os.listdir(sitedir))
        sample_list = [x for x in sample_list if 'segmentation.nii.gz' in x.lower()]

        for sample in sample_list:
            sampledir = os.path.join(sitedir, sample)
            if os.path.getsize(sampledir) < 1024 * 1024 and sampledir.endswith("nii.gz"):
                imgdir = os.path.join(sitedir, sample[:6] + ".nii.gz")
                label_v = sitk.ReadImage(sampledir)
                image_v = sitk.ReadImage(imgdir)
                label_v = sitk.GetArrayFromImage(label_v)
                label_v[label_v > 1] = 1
                image_v = sitk.GetArrayFromImage(image_v)
                image_v = convert_from_nii_to_png(image_v)

                for i in range(1, label_v.shape[0] - 1):
                    label = np.array(label_v[i, :, :])
                    if np.all(label == 0):
                        continue
                    image = np.array(image_v[i - 1:i + 2, :, :])
                    image = np.transpose(image, (1, 2, 0))

                    labels.append(label)
                    images.append(image)
        labels = np.array(labels).astype(int)
        images = np.array(images)

        index_path = "data/prostate/{}-index.npy".format(site)
        if not os.path.exists(index_path):
            index = np.random.permutation(len(images)).tolist()
            np.save(index_path, index)
        else:
            index = np.load("data/prostate/{}-index.npy".format(site)).tolist()

        labels = labels[index]
        images = images[index]

        train_labeled_len = int(0.2 * len(labels))
        train_unlabeled_len = int(0.6 * len(labels))
        train_len = train_labeled_len + train_unlabeled_len
        vallen = int(0.1 * len(labels))
        testlen = int(0.1 * len(labels))

        if split == 'train_labeled':
            self.images, self.labels = images[:train_labeled_len], labels[:train_labeled_len]
        elif split == 'train_unlabeled':
            self.images, self.labels = images[train_labeled_len:train_labeled_len + train_unlabeled_len], labels[train_labeled_len:train_labeled_len + train_unlabeled_len]
        elif split == 'valid':
            self.images, self.labels = images[train_len:train_len + vallen], labels[train_len:train_len + vallen]
        else:
            self.images, self.labels = images[-testlen:], labels[-testlen:]

        self.transform = transform
        self.channels = channels[site]
        self.labels = self.labels.astype(np.long).squeeze()

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform is not None:
            if self.split == 'train_labeled':
                R1 = RandomRotate90()
                image, label = R1(image, label)
                R2 = RandomFlip()
                image, label = R2(image, label)

                image = np.transpose(image, (2, 0, 1))
                image = torch.Tensor(image)
                label = self.transform(label)
                return image, label
            elif self.split == 'train_unlabeled':
                prob = random.random()
                R1 = RandomRotate90()
                R2 = RandomFlip()
                R3 = RandomNoise()
                if prob < 0.25:
                    image1, label1 = R1(image, label)
                    image1, label1 = R2(image1, label1)
                    image2, label2 = R2(image, label)
                    image2, label2 = R3(image2, label2)
                elif 0.25 <= prob <  0.5:
                    image1, label1 = R1(image, label)
                    image1, label1 = R2(image1, label1)
                    image2, label2 = R1(image, label)
                    image2, label2 = R3(image2, label2)
                elif 0.5 <= prob < 0.75:
                    image1, label1 = R1(image, label)
                    image1, label1 = R2(image1, label1)
                    image2, label2 = R1(image, label)
                    image2, label2 = R2(image2, label2)
                else:
                    image1, label1 = R1(image, label)
                    image1, label1 = R2(image1, label1)
                    image2, label2 = R1(image, label)
                    image2, label2 = R2(image2, label2)
                    image2, label2 = R3(image2, label2)

                image1 = np.transpose(image1, (2, 0, 1))
                image1 = torch.Tensor(image1)
                image2 = np.transpose(image2, (2, 0, 1))
                image2 = torch.Tensor(image2)

                return image1, image2
            else:
                image = np.transpose(image, (2, 0, 1))
                image = torch.Tensor(image)
                label = self.transform(label)
                return image, label


class Fundus(Dataset):
    def __init__(self, site, base_path, split='train', transform=None):
        assert split in ['train_labeled', 'train_unlabeled', 'valid', 'test']
        assert site in ['SiteA', 'SiteB', 'SiteC', 'SiteD']

        self.base_path = base_path
        self.base_path = os.path.join(self.base_path, site)

        images = []
        labels = []
        if split == 'train_labeled' or split == 'train_unlabeled':
            self.base_path = os.path.join(self.base_path, 'train')
        else:
            self.base_path = os.path.join(self.base_path, split)
        img_path = os.path.join(self.base_path, "image")
        lbl_path = os.path.join(self.base_path, "label")
        for i in os.listdir(img_path):
            img_dir = os.path.join(img_path, i)
            ibl_dir = os.path.join(lbl_path, i)
            images.append(img_dir)
            labels.append(ibl_dir)

        train_labeled_len = int(0.2 * len(labels))

        if split == 'train_labeled':
            self.images, self.labels = images[:train_labeled_len], labels[:train_labeled_len]
        elif split == 'train_unlabeled':
            self.images, self.labels = images[train_labeled_len:], labels[train_labeled_len:]
        elif split == 'valid':
            self.images, self.labels = images, labels
        else:
            self.images, self.labels = images, labels

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')

        label = cv2.imread(self.labels[idx], 0)
        label[label == 0] = 1   #black 1
        label[label == 128] = 2 #gray 2
        label[label == 255] = 0 #background 0
        label = Image.fromarray(label)

        if self.transform is not None:
            image = self.transform(image)
            TTensor = transforms.ToTensor()
            image = TTensor(image)

            label = self.transform(label)
            label = np.array(label)
            label = torch.Tensor(label)
            label = torch.unsqueeze(label, dim=0)

        return image, label

class Nuclei(Dataset):
    def __init__(self, site, base_path=None, split='train', transform=None):
        assert split in ['train', 'val', 'test']
        assert site in ['SiteA', 'SiteB', 'SiteC', 'SiteD', 'SiteE', 'SiteF']

        self.base_path = base_path
        self.base_path = os.path.join(self.base_path, site)

        images = []
        labels = []

        self.base_path = os.path.join(self.base_path, split)
        img_path = os.path.join(self.base_path, "images")
        lbl_path = os.path.join(self.base_path, "labels")
        for i in os.listdir(img_path):
            img_dir = os.path.join(img_path, i)
            ibl_dir = os.path.join(lbl_path, i.split('.')[0] + ".png")
            images.append(img_dir)
            labels.append(ibl_dir)

        self.images, self.labels = images, labels

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        label = cv2.imread(self.labels[idx], 0)

        label[label == 255] = 1
        label = Image.fromarray(label)

        if self.transform is not None:
            image = self.transform(image)
            label = self.transform(label)

            TTensor = transforms.ToTensor()
            image = TTensor(image)

            label = np.array(label)
            label = torch.Tensor(label)

            label = torch.unsqueeze(label, dim=0)

        return image, label


class DomainNetDataset(Dataset):
    def __init__(self, site, base_path, train_ratio=1, split='train', transform=None):
        self.split = split
        if split in ('train_labeled', 'train_unlabeled'):
            self.paths, self.text_labels = np.load('./data/domainnet/{}_train.pkl'.format(site.lower()),
                                                   allow_pickle=True)
            train_labeled_len = int(0.2 * len(self.paths))
            train_unlabeled_len = int(0.8 * len(self.paths))
        else:
            self.paths, self.text_labels = np.load('./data/domainnet/{}_test.pkl'.format(site.lower()),
                                                   allow_pickle=True)
            val_len = int(0.5 * len(self.paths))
            test_len = int(0.5 * len(self.paths))

        label_dict = {'bird': 0, 'feather': 1, 'headphones': 2, 'ice_cream': 3, 'teapot': 4,
                      'tiger': 5, 'whale': 6, 'windmill': 7, 'wine_glass': 8, 'zebra': 9}

        if split == 'train_labeled':
            self.paths = self.paths[:train_labeled_len]
            self.text_labels = self.text_labels[:train_labeled_len]
        elif split == 'train_unlabeled':
            self.paths = self.paths[-train_unlabeled_len:]
            self.text_labels = self.text_labels[-train_unlabeled_len:]
        elif split == 'val':
            self.paths = self.paths[:val_len]
            self.text_labels = self.text_labels[:val_len]
        elif split == 'test':
            self.paths = self.paths[-test_len:]
            self.text_labels = self.text_labels[-test_len:]

        self.labels = [label_dict[text] for text in self.text_labels]
        if split == 'train_unlabeled':
            self.transform_0 = transform[0]
            self.transform_1 = transform[1]
        else:
            self.transform = transform
        self.base_path = base_path

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.base_path, self.paths[idx].replace('DomainNet', 'domainnet'))
        label = self.labels[idx]
        image = Image.open(img_path)

        if len(image.split()) != 3:
            image = transforms.Grayscale(num_output_channels=3)(image)

        if self.split == 'train_unlabeled':
            image_0 = self.transform_0(image)
            image_1 = self.transform_1(image)
            return image_0, image_1
        else:
            if self.transform is not None:
                image = self.transform(image)
            return image, label


class DigitsDataset(Dataset):
    def __init__(self, site, base_path, split='train', transform=None):
        if site == 'USPS':
            self.channels = 1
        else:
            self.channels = 3

        sitedir = os.path.join(base_path, site)
        if split == 'train':
            self.image_list, self.label_list = [np.load(os.path.join(sitedir, 'partitions/train_part{}.pkl'.format(i)),
                                                        allow_pickle=True) for i in range(8)]
            self.images = np.concatenate(self.image_list, axis=0)
            self.labels = np.concatenate(self.label_list, axis=0)
        elif split == 'val':
            self.image_list, self.label_list = [np.load(os.path.join(sitedir, 'partitions/train_part{}.pkl'.format(i)),
                                                        allow_pickle=True) for i in range(8, 10, 1)]
            self.images = np.concatenate(self.image_list, axis=0)
            self.labels = np.concatenate(self.label_list, axis=0)
        else:
            self.images, self.labels = np.load(os.path.join(base_path, 'test.pkl'), allow_pickle=True)

        self.transform = transform
        self.labels = self.labels.astype(np.long).squeeze()

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.channels == 1:
            image = Image.fromarray(image, mode='L')
        elif self.channels == 3:
            image = Image.fromarray(image, mode='RGB')
        else:
            raise ValueError("{} channel is not allowed.".format(self.channels))

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class OfficeDataset(Dataset):
    def __init__(self, site, base_path, split='train', train_ratio=1, transform=None):
        self.split = split
        if split == 'train_labeled':
            self.paths, self.text_labels = np.load('{}/{}_train.pkl'.format(base_path, site),
                                                   allow_pickle=True)
            total = len(self.paths)
            self.paths = self.paths[:int(0.2 * total)]
            self.text_labels = self.text_labels[:int(0.2 * total)]
        elif split == 'train_unlabeled':
            self.paths, self.text_labels = np.load('{}/{}_train.pkl'.format(base_path, site),
                                                   allow_pickle=True)
            total = len(self.paths)
            self.paths = self.paths[int(0.2 * total):]
            self.text_labels = self.text_labels[int(0.2 * total):]
        elif split == 'val':
            self.paths, self.text_labels = np.load('{}/{}_test.pkl'.format(base_path, site),
                                                   allow_pickle=True)
            total = len(self.paths)
            self.paths = self.paths[:int(0.5 * total)]
            self.text_labels = self.text_labels[:int(0.5 * total)]
        else:
            self.paths, self.text_labels = np.load('{}/{}_test.pkl'.format(base_path, site),
                                                   allow_pickle=True)
            total = len(self.paths)
            self.paths = self.paths[int(0.5 * total):]
            self.text_labels = self.text_labels[int(0.5 * total):]

        label_dict = {'back_pack': 0, 'bike': 1, 'calculator': 2, 'headphones': 3, 'keyboard': 4, 'laptop_computer': 5,
                      'monitor': 6, 'mouse': 7, 'mug': 8, 'projector': 9}
        self.labels = [label_dict[text] for text in self.text_labels]
        if split == 'train_unlabeled':
            self.transform_0 = transform[0]
            self.transform_1 = transform[1]
        else:
            self.transform = transform
        self.base_path = os.path.join(base_path, site)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        self.splits = self.paths[idx].split('/')
        self.image_path = '/'.join(self.splits[2:])

        img_path = os.path.join(self.base_path, self.image_path)
        label = self.labels[idx]
        image = Image.open(img_path)

        if len(image.split()) != 3:
            image = transforms.Grayscale(num_output_channels=3)(image)

        if self.split == 'train_unlabeled':
            image_0 = self.transform_0(image)
            image_1 = self.transform_1(image)
            return image_0, image_1
        else:
            if self.transform is not None:
                image = self.transform(image)
            return image, label

def np_load(base_dir, filename):
    return np.load(os.path.join(base_dir, filename), allow_pickle=True).item()

class CIFAR_10_Dataset(Dataset):
    def __init__(self, site, base_path, split='train', transform=None):
        self.split = split
        self.base_path = base_path
        if split == 'train_labeled':
            task = np_load(self.base_path, f's_cifar_10_{site}.npy')
            self.images, self.labels = task['x'], task['y']
        elif split == 'train_unlabeled':
            task = np_load(self.base_path, f'u_cifar_10_{site}.npy')
            self.images, self.labels = task['x'], task['y']
        elif split == 'val':
            task = np_load(self.base_path, 'valid_cifar_10.npy')
            self.images, self.labels = task['x'], task['y']
        else:
            task = np_load(self.base_path, 'test_cifar_10.npy')
            self.images, self.labels = task['x'], task['y']

        if split == 'train_unlabeled':
            self.transform_0 = transform[0]
            self.transform_1 = transform[1]
        else:
            self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        image = Image.fromarray(np.uint8(image))
        label = self.labels[idx]
        label = np.argmax(label)

        if self.split == 'train_unlabeled':
            image_0 = self.transform_0(image)
            image_1 = self.transform_1(image)
            return image_0, image_1
        else:
            if self.transform is not None:
                image = self.transform(image)
            return image, label

class VLCSDataset(Dataset):
    def __init__(self, site, base_path, split='train', train_ratio=1, transform=None):
        if split == 'train':
            data_path = os.path.join(base_path, site, 'train')
        elif split == 'val' or split == 'valid':
            data_path = os.path.join(base_path, site, 'crossval')
        elif split == 'test':
            data_path = os.path.join(base_path, site, 'test')

        self.paths = []
        self.labels = []

        cats = ['0', '1', '2', '3', '4']
        for i, cat in enumerate(cats):
            image_list = os.listdir(os.path.join(data_path, cat))
            image_path = [os.path.join(data_path, cat, x) for x in image_list]

            self.paths.extend(image_path)
            self.labels.extend([i] * len(image_path))

        c = list(zip(self.paths, self.labels))
        random.shuffle(c)
        self.paths, self.labels = zip(*c)

        num = int(len(self.paths) * 0.2)

        self.paths, self.labels = self.paths[:num], self.labels[:num]

        self.transform = transform
        self.base_path = os.path.join(base_path, site)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path)

        if len(image.split()) != 3:
            image = transforms.Grayscale(num_output_channels=3)(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def convert_from_nii_to_png(img):
    high = np.quantile(img, 0.99)
    low = np.min(img)
    img = np.where(img > high, high, img)
    lungwin = np.array([low * 1., high * 1.])
    newimg = (img - lungwin[0]) / (lungwin[1] - lungwin[0])
    newimg = (newimg * 255).astype(np.uint8)
    return newimg


class RandomRotate90:
    def __init__(self, prob=1.0):
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            factor = random.randint(0, 4)
            img = np.rot90(img, factor)
            if mask is not None:
                mask = np.rot90(mask, factor)
        return img.copy(), mask.copy()


class RandomFlip:
    def __init__(self, prob=0.75):
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            d = random.randint(-1, 1)
            img = cv2.flip(img, d)
            if mask is not None:
                mask = cv2.flip(mask, d)

        return img, mask

class RandomNoise:
    def __init__(self, mu=0, sigma=0.1, prob=0.5):
        self.mu = mu
        self.sigma = sigma
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            noise = np.clip(self.sigma * np.random.randn(img.shape[0], img.shape[1], img.shape[2]), -2*self.sigma, 2*self.sigma)
            noise = noise + self.mu
            img = img + noise

        return img, mask


if __name__ == '__main__':
    # sites = ['BIDMC', 'HK', 'I2CVB', 'BMC', 'RUNMC', 'UCL']
    sites = ['BMC']
    base_path = '/scratch/223284650.tmpdir/ProstateMRI'

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    for site in sites:
        print(site)
        trainset = Prostate(site=site, base_path=base_path, split='train', transform=transform)
        valset = Prostate(site=site, base_path=base_path, split='val', transform=transform)
        testset = Prostate(site=site, base_path=base_path, split='test', transform=transform)
