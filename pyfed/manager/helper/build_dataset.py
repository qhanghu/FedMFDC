import random
import numpy as np

import torch
import torchvision.transforms as transforms
import pyfed.utils.hecktor_transforms as hecktor_transforms
from pyfed.dataset.dataset import Prostate, Fundus, Nuclei, DomainNetDataset, HecktorDataset, OfficeDataset, VLCSDataset, CIFAR_10_Dataset
from pyfed.dataset.dataset_cls import _get_cifar, define_val_dataset
from pyfed.utils.log import print_log
from pyfed.manager.helper.randAugment import RandAugment
from pyfed.manager.helper.fixmatchAugment import RandAugmentMC

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def build_dataset(config, site):
    #assert site in config.INNER_SITES + config.OUTER_SITES
    if config.DATASET == 'prostate':
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_labeled_set = Prostate(site=site, base_path=config.DIR_DATA, train_ratio=config.TRAIN_RATIO,
                             split='train_labeled', transform=transform)
        train_unlabeled_set = Prostate(site=site, base_path=config.DIR_DATA, train_ratio=config.TRAIN_RATIO,
                                   split='train_unlabeled', transform=transform)
        valid_set = Prostate(site=site, base_path=config.DIR_DATA, train_ratio=config.TRAIN_RATIO,
                             split='valid', transform=transform)
        test_set = Prostate(site=site, base_path=config.DIR_DATA,train_ratio=config.TRAIN_RATIO,
                            split='test', transform=transform)
    elif config.DATASET == 'fundus':
        train_transform = transforms.Compose([
            transforms.Resize([384, 384]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((-30, 30)),
        ])
        valid_transform = transforms.Compose([
            transforms.Resize([384, 384]),
        ])

        train_labeled_set = Fundus(site=site, base_path=config.DIR_DATA, split='train_labeled', transform=train_transform)
        train_unlabeled_set = Fundus(site=site, base_path=config.DIR_DATA, split='train_unlabeled', transform=train_transform)
        valid_set = Fundus(site=site, base_path=config.DIR_DATA, split='valid', transform=valid_transform)
        test_set = Fundus(site=site, base_path=config.DIR_DATA, split='test', transform=valid_transform)
    elif config.DATASET == 'hecktor':
        train_transform = hecktor_transforms.Compose([
            hecktor_transforms.RandomRotation(p=0.5, angle_range=[0, 30]),
            hecktor_transforms.Mirroring(p=0.5),
            hecktor_transforms.NormalizeIntensity(),
            hecktor_transforms.ToTensor()
        ])

        valid_transform = hecktor_transforms.Compose([
            hecktor_transforms.NormalizeIntensity(),
            hecktor_transforms.ToTensor()
        ])

        train_set = HecktorDataset(site=site, base_path=config.DIR_DATA, split='train', transforms=train_transform)
        valid_set = HecktorDataset(site=site, base_path=config.DIR_DATA, split='valid', transforms=valid_transform)
        test_set = HecktorDataset(site=site, base_path=config.DIR_DATA, split='test', transforms=valid_transform)
    elif config.DATASET == 'nuclei':
        transform = transforms.Compose([
            transforms.Resize([256, 256]),
        ])

        train_set = Nuclei(site=site, base_path=config.DIR_DATA, split='train', transform=transform)
        valid_set = Nuclei(site=site, base_path=config.DIR_DATA, split='val', transform=transform)
        test_set = Nuclei(site=site, base_path=config.DIR_DATA, split='test', transform=transform)
    elif config.DATASET == 'domainnet':
        if config.CLIENT == 'UdaClient':
            train_transform = transforms.Compose([
                transforms.Resize([256, 256]),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation((-30, 30)),
                transforms.ToTensor(),
            ])

            train_unsup_transform = transforms.Compose([
                transforms.Resize([256, 256]),
                RandAugment(2, 15),
                transforms.ToTensor(),
            ])

            test_transform = transforms.Compose([
                transforms.Resize([256, 256]),
                transforms.ToTensor(),
            ])

            train_labeled_set = DomainNetDataset(site=site, base_path=config.DIR_DATA, train_ratio=config.TRAIN_RATIO,
                                         split='train_labeled', transform=train_transform)
            train_unlabeled_set = DomainNetDataset(site=site, base_path=config.DIR_DATA, train_ratio=config.TRAIN_RATIO,
                                      split='train_unlabeled', transform=[train_transform, train_unsup_transform])
            valid_set = DomainNetDataset(site=site, base_path=config.DIR_DATA, train_ratio=config.TRAIN_RATIO,
                                         split='val', transform=test_transform)
            test_set = DomainNetDataset(site=site, base_path=config.DIR_DATA, train_ratio=config.TRAIN_RATIO,
                                        split='test', transform=test_transform)
        else:
            train_transform = transforms.Compose([
                transforms.Resize([256, 256]),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation((-30, 30)),
                transforms.ToTensor(),
            ])

            train_strong_transform = transforms.Compose([
                transforms.Resize([256, 256]),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation((-30, 30)),
                RandAugmentMC(n=2, m=10),
                transforms.ToTensor(),
            ])

            test_transform = transforms.Compose([
                transforms.Resize([256, 256]),
                transforms.ToTensor(),
            ])

            train_labeled_set = DomainNetDataset(site=site, base_path=config.DIR_DATA, train_ratio=config.TRAIN_RATIO,
                                              split='train_labeled', transform=train_transform)
            train_unlabeled_set = DomainNetDataset(site=site, base_path=config.DIR_DATA, train_ratio=config.TRAIN_RATIO,
                                                split='train_unlabeled',
                                                transform=[train_transform, train_strong_transform])
            valid_set = DomainNetDataset(site=site, base_path=config.DIR_DATA, train_ratio=config.TRAIN_RATIO,
                                      split='val', transform=test_transform)
            test_set = DomainNetDataset(site=site, base_path=config.DIR_DATA, train_ratio=config.TRAIN_RATIO,
                                     split='test', transform=test_transform)
    elif config.DATASET == 'office':
        if config.CLIENT == 'UdaClient':
            train_transform = transforms.Compose([
                transforms.Resize([256, 256]),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation((-30, 30)),
                transforms.ToTensor(),
            ])

            train_unsup_transform = transforms.Compose([
                transforms.Resize([256, 256]),
                RandAugment(2, 15),
                transforms.ToTensor(),
            ])

            test_transform = transforms.Compose([
                transforms.Resize([256, 256]),
                transforms.ToTensor(),
            ])

            train_labeled_set = OfficeDataset(site=site, base_path=config.DIR_DATA, train_ratio=config.TRAIN_RATIO,
                                         split='train_labeled', transform=train_transform)
            train_unlabeled_set = OfficeDataset(site=site, base_path=config.DIR_DATA, train_ratio=config.TRAIN_RATIO,
                                      split='train_unlabeled', transform=[train_transform, train_unsup_transform])
            valid_set = OfficeDataset(site=site, base_path=config.DIR_DATA, train_ratio=config.TRAIN_RATIO,
                                         split='val', transform=test_transform)
            test_set = OfficeDataset(site=site, base_path=config.DIR_DATA, train_ratio=config.TRAIN_RATIO,
                                        split='test', transform=test_transform)
        else:
            train_transform = transforms.Compose([
                transforms.Resize([256, 256]),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation((-30, 30)),
                transforms.ToTensor(),
            ])

            train_strong_transform = transforms.Compose([
                transforms.Resize([256, 256]),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation((-30, 30)),
                RandAugmentMC(n=2, m=10),
                transforms.ToTensor(),
            ])

            test_transform = transforms.Compose([
                transforms.Resize([256, 256]),
                transforms.ToTensor(),
            ])

            train_labeled_set = OfficeDataset(site=site, base_path=config.DIR_DATA, train_ratio=config.TRAIN_RATIO,
                                              split='train_labeled', transform=train_transform)
            train_unlabeled_set = OfficeDataset(site=site, base_path=config.DIR_DATA, train_ratio=config.TRAIN_RATIO,
                                                split='train_unlabeled',
                                                transform=[train_transform, train_strong_transform])
            valid_set = OfficeDataset(site=site, base_path=config.DIR_DATA, train_ratio=config.TRAIN_RATIO,
                                      split='val', transform=test_transform)
            test_set = OfficeDataset(site=site, base_path=config.DIR_DATA, train_ratio=config.TRAIN_RATIO,
                                     split='test', transform=test_transform)
    elif config.DATASET == 'cifar10':
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop((32, 32), 4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_unsup_transform = transforms.Compose([
            RandAugment(2, 15),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_labeled_set = CIFAR_10_Dataset(site=site, base_path=config.base_dir, split='train_labeled', transform=train_transform)
        train_unlabeled_set = CIFAR_10_Dataset(site=site, base_path=config.base_dir, split='train_unlabeled', transform=[train_transform, train_unsup_transform])
        valid_set = CIFAR_10_Dataset(site=site, base_path=config.base_dir, split='val', transform=test_transform)
        test_set = CIFAR_10_Dataset(site=site, base_path=config.base_dir, split='test', transform=test_transform)
    elif config.DATASET == 'vlcs':
        train_transform = transforms.Compose([
            transforms.Resize([256, 256]),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation((-30, 30)),
            transforms.ToTensor(),
        ])

        test_transform = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.ToTensor(),
        ])

        train_set = VLCSDataset(site=site, base_path=config.DIR_DATA, train_ratio=config.TRAIN_RATIO,
                                  split='train', transform=train_transform)
        valid_set = VLCSDataset(site=site, base_path=config.DIR_DATA, train_ratio=config.TRAIN_RATIO,
                                  split='val', transform=test_transform)
        test_set = VLCSDataset(site=site, base_path=config.DIR_DATA, train_ratio=config.TRAIN_RATIO,
                                 split='test', transform=test_transform)


    print_log('[Client {}] Train_labeled={}, Train_unlabeled={}, Val={}, Test={}'.format(site, len(train_labeled_set), len(train_unlabeled_set),len(valid_set), len(test_set)))

    train_labeled_loader = torch.utils.data.DataLoader(train_labeled_set, batch_size=config.TRAIN_BATCHSIZE,
                                               shuffle=True, drop_last=True, num_workers=4, worker_init_fn=seed_worker)
    train_unlabeled_loader = torch.utils.data.DataLoader(train_unlabeled_set, batch_size=config.TRAIN_BATCHSIZE_UN,
                                               shuffle=True, drop_last=True, num_workers=4, worker_init_fn=seed_worker)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=config.TRAIN_BATCHSIZE,
                                               shuffle=False, drop_last=False, num_workers=4, worker_init_fn=seed_worker)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=config.TRAIN_BATCHSIZE,
                                              shuffle=False, drop_last=False, num_workers=4, worker_init_fn=seed_worker)

    return train_labeled_loader, train_unlabeled_loader, valid_loader, test_loader

def build_cls_dataset(config):
    if config.DATASET == 'cifar10' or config.DATASET == 'cifar100':
        train_set = _get_cifar(config.DATASET, root=config.DIR_DATA, split='train')
        test_set = _get_cifar(config.DATASET, root=config.DIR_DATA, split='test')

        train_set, valid_set, test_set = define_val_dataset(
            config, train_set, test_set
        )

    print(
            "Data stat for original dataset: we have {} samples for train, {} samples for val, {} samples for test.".format(
                len(train_set),
                len(valid_set) if valid_set is not None else 0,
                len(test_set),
            )
        )
    return {"train": train_set, "val": valid_set, "test": test_set}


def build_central_dataset(config, sites):
    train_sets, valid_sets, test_sets = [], [], []
    train_loaders, valid_loaders, test_loaders = [], [], []
    if config.DATASET == 'prostate':
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        for site in sites:
            train_set = Prostate(site=site, base_path=config.DIR_DATA, split='train', transform=transform)
            valid_set = Prostate(site=site, base_path=config.DIR_DATA, split='valid', transform=transform)
            test_set = Prostate(site=site, base_path=config.DIR_DATA, split='test', transform=transform)

            print_log(f'[Client {site}] Train={len(train_set)}, Val={len(valid_set)}, Test={len(test_set)}')
            train_sets.append(train_set)
            valid_sets.append(valid_set)
            test_sets.append(test_set)

        train_set = torch.utils.data.ConcatDataset(train_sets)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=config.TRAIN_BATCHSIZE, shuffle=True)
        valid_loaders, test_loaders = [], []
        for valid_set, test_set in zip(valid_sets, test_sets):
            valid_loaders.append(torch.utils.data.DataLoader(valid_set, batch_size=config.TRAIN_BATCHSIZE, shuffle=False))
            test_loaders.append(torch.utils.data.DataLoader(test_set, batch_size=config.TRAIN_BATCHSIZE, shuffle=False))

    elif config.DATASET == 'hecktor':
        train_transform = hecktor_transforms.Compose([
            hecktor_transforms.RandomRotation(p=0.5, angle_range=[0, 30]),
            hecktor_transforms.Mirroring(p=0.5),
            hecktor_transforms.NormalizeIntensity(),
            hecktor_transforms.ToTensor()
        ])

        valid_transform = hecktor_transforms.Compose([
            hecktor_transforms.NormalizeIntensity(),
            hecktor_transforms.ToTensor()
        ])

        for site in sites:
            train_set = HecktorDataset(site=site, base_path=config.DIR_DATA, split='train', transforms=train_transform)
            valid_set = HecktorDataset(site=site, base_path=config.DIR_DATA, split='valid', transforms=valid_transform)
            test_set = HecktorDataset(site=site, base_path=config.DIR_DATA, split='test', transforms=valid_transform)

            print_log(f'[Client {site}] Train={len(train_set)}, Val={len(valid_set)}, Test={len(test_set)}')
            train_sets.append(train_set)
            valid_sets.append(valid_set)
            test_sets.append(test_set)

        train_set = torch.utils.data.ConcatDataset(train_sets)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=config.TRAIN_BATCHSIZE, shuffle=True)
        valid_loaders, test_loaders = [], []
        for valid_set, test_set in zip(valid_sets, test_sets):
            valid_loaders.append(
                torch.utils.data.DataLoader(valid_set, batch_size=config.TRAIN_BATCHSIZE, shuffle=False))
            test_loaders.append(torch.utils.data.DataLoader(test_set, batch_size=config.TRAIN_BATCHSIZE, shuffle=False))

    elif config.DATASET == 'nuclei':
        transform = transforms.Compose([
            transforms.Resize([256, 256]),
        ])

        for site in sites:
            train_set = Nuclei(site=site, base_path=config.DIR_DATA, split='train', transform=transform)
            valid_set = Nuclei(site=site, base_path=config.DIR_DATA, split='val', transform=transform)
            test_set = Nuclei(site=site, base_path=config.DIR_DATA, split='test', transform=transform)

            print_log(f'[Client {site}] Train={len(train_set)}, Val={len(valid_set)}, Test={len(test_set)}')
            train_sets.append(train_set)
            valid_sets.append(valid_set)
            test_sets.append(test_set)

        train_set = torch.utils.data.ConcatDataset(train_sets)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=config.TRAIN_BATCHSIZE, shuffle=True)
        valid_loaders, test_loaders = [], []
        for valid_set, test_set in zip(valid_sets, test_sets):
            valid_loaders.append(
                torch.utils.data.DataLoader(valid_set, batch_size=config.TRAIN_BATCHSIZE, shuffle=False))
            test_loaders.append(torch.utils.data.DataLoader(test_set, batch_size=config.TRAIN_BATCHSIZE, shuffle=False))

    return train_loader, valid_loaders, test_loaders