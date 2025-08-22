import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import Fed_Test


# 假设的Config类，用于提供batch_size
# class Config:
#     batch_size = 32


def get_data_loaders(num_clients):
    """
    加载 FashionMNIST 数据集，并为 num_clients 个客户端创建不相交的 Non-IID 数据加载器。

    改动说明:
    - 确保分得相同类别的客户端拥有该类别下互不相同的样本。
    - 返回每个客户端最终分配到的样本数量。

    Args:
        num_clients (int): 客户端的总数。

    Returns:
        tuple: 包含三个元素的元组:
               - client_loaders (list): 每个客户端的 DataLoader 列表。
               - test_loader (DataLoader): 用于测试的 DataLoader。
               - client_samples_count (list): 每个客户端分配到的样本数量列表。
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    # --- 1. 按类别对所有训练样本的索引进行分组 ---
    labels = np.array(train_dataset.targets)
    indices_by_class = {i: np.where(labels == i)[0] for i in range(10)}
    # 为了随机性，打乱每个类别内部的索引顺序
    for c in indices_by_class:
        np.random.shuffle(indices_by_class[c])

    # --- 2. 计算每个类别需要被分配给多少个客户端 ---
    # 例如，类别1会被分配给 client 0 (class2) 和 client 1 (class1)
    # 如果 num_clients=11, 类别1还会被分配给 client 10 (class2)
    class_split_counts = {i: 0 for i in range(10)}
    for i in range(num_clients):
        class1_idx = i % 10
        class2_idx = (i + 1) % 10
        class_split_counts[class1_idx] += 1
        class_split_counts[class2_idx] += 1

    # 如果某个类别没有被分配，将其计数值设为1以避免除以0的错误
    for c in class_split_counts:
        if class_split_counts[c] == 0:
            class_split_counts[c] = 1

    # --- 3. 将每个类别的索引列表切分成不相交的块(chunks) ---
    chunks_by_class = {c: np.array_split(indices, class_split_counts[c])
                       for c, indices in indices_by_class.items()}

    # --- 4. 按顺序为每个客户端分配不相交的样本块 ---
    client_indices = [[] for _ in range(num_clients)]
    # 用于追踪每个类别下一个应该分配哪个块
    chunk_counters = {i: 0 for i in range(10)}

    for i in range(num_clients):
        class1_idx = i % 10
        class2_idx = (i + 1) % 10

        # 为 class1 分配一个块
        chunk1_idx = chunk_counters[class1_idx]
        chunk1 = chunks_by_class[class1_idx][chunk1_idx]
        chunk_counters[class1_idx] += 1

        # 为 class2 分配一个块
        chunk2_idx = chunk_counters[class2_idx]
        chunk2 = chunks_by_class[class2_idx][chunk2_idx]
        chunk_counters[class2_idx] += 1

        # 合并两个类别的块作为该客户端的数据
        all_indices = np.concatenate((chunk1, chunk2))
        client_indices[i] = all_indices.astype(int).tolist()

    # --- 创建 DataLoader 和计算每个客户端的样本量 ---
    client_loaders = []
    client_samples_count = []
    for indices in client_indices:
        client_samples_count.append(len(indices))
        subset = Subset(train_dataset, indices)
        loader = DataLoader(subset, batch_size=Fed_Test.Config().batch_size, shuffle=True)
        client_loaders.append(loader)

    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    return client_loaders, test_loader, client_samples_count


# --- 使用示例 ---
if __name__ == '__main__':
    NUM_CLIENTS = 19  # 假设有20个客户端，这样每个类别会被分配4次

    # 调用调整后的函数
    client_loaders, test_loader, client_samples_count = get_data_loaders(NUM_CLIENTS)

    print(f"总共创建了 {len(client_loaders)} 个客户端的数据加载器。")
    print("-" * 30)
    print("每个客户端分配到的样本数量:")
    for i, count in enumerate(client_samples_count):
        print(f"  客户端 {i:<2}: {count:<4} 个样本")
    print("-" * 30)
    print(f"样本总量核对: {sum(client_samples_count)}")
    print(f"原始训练集大小: {len(datasets.FashionMNIST(root='./data', train=True, download=True))}")  # 注意：由于每个类别被分配了两次，总样本数会翻倍

    # client_samples_count 这个列表现在可以用于计算每个客户端的 `steps`
    # 例如，对于客户端 i:
    # steps_per_epoch = ceil(client_samples_count[i] / batch_size)
    # total_steps = communication_rounds * local_epochs * steps_per_epoch