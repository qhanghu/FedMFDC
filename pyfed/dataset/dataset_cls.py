import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from pyfed.dataset.partition_data import DataPartitioner


def _get_cifar(name, root='./downloaded_data', split='train', transform=None, target_transform=None, download=True):
    is_train = split == "train"

    # decide normalize parameter.
    if name == "cifar10":
        dataset_loader = datasets.CIFAR10
        normalize = (
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        )
    elif name == "cifar100":
        dataset_loader = datasets.CIFAR100
        normalize = (
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        )

    # decide data type.
    if is_train:
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop((32, 32), 4),
                transforms.ToTensor(),
            ]
            + ([normalize] if normalize is not None else [])
        )
    else:
        transform = transforms.Compose(
            [transforms.ToTensor()] + ([normalize] if normalize is not None else [])
        )
    return dataset_loader(
        root=root,
        train=is_train,
        transform=transform,
        target_transform=target_transform,
        download=download,
    )


def define_val_dataset(conf, train_dataset, test_dataset):
    assert conf.VAL_DATA_RATIO >= 0

    partition_sizes = [
        (1 - conf.VAL_DATA_RATIO) * conf.TRAIN_DATA_RATIO,
        (1 - conf.VAL_DATA_RATIO) * (1 - conf.TRAIN_DATA_RATIO),
        conf.VAL_DATA_RATIO,
    ]

    data_partitioner = DataPartitioner(
        conf,
        train_dataset,
        partition_sizes,
        partition_type="origin",
        consistent_indices=False,
    )
    train_dataset = data_partitioner.use(0)

    # split for val data.
    if conf.VAL_DATA_RATIO > 0:
        assert conf.PARTITIONED_BY_USER is False

        val_dataset = data_partitioner.use(2)
        return train_dataset, val_dataset, test_dataset
    else:
        return train_dataset, None, test_dataset


def _define_data_loader(config, dataset, localdata_id, is_train=True, shuffle=True, data_partitioner=None):
    if is_train:
        world_size = config.NUM_SITES
        partition_sizes = [1.0 / world_size for _ in range(world_size)]
        """
        if data_partitioner == None:
            data_partitioner = DataPartitioner(
                config, dataset, partition_sizes, partition_type=config.PARTITION_TYPE
            )
        """
        data_partitioner = DataPartitioner(
            config, dataset, partition_sizes, partition_type=config.PARTITION_TYPE
        )

        data_to_load = data_partitioner.use(int(localdata_id))
        print(f"Data partition for train (client_id={localdata_id}): partitioned data and use subdata.")
    else:
        data_to_load = dataset
        print("Data partition for validation/test.")

    data_loader = torch.utils.data.DataLoader(
        data_to_load,
        batch_size=config.TRAIN_BATCHSIZE,
        shuffle=shuffle,
        drop_last=False
    )

    # Some simple statistics.
    print(
        "\tData stat for {}: # of samples={} for {}. # of batches={}. The batch size={}".format(
            "train" if is_train else "validation/test",
            len(data_to_load),
            f"client_id={localdata_id}" if localdata_id is not None else "Master",
            len(data_loader),
            config.TRAIN_BATCHSIZE,
        )
    )

    return data_loader, data_partitioner