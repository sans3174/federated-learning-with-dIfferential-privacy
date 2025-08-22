import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from opacus.accountants import RDPAccountant
from tqdm import tqdm
import math
import warnings
import utils

warnings.filterwarnings("ignore")


# --- 1. 配置参数 ---
class Config:
    def __init__(self):
        self.num_clients = 10
        self.num_rounds = 50
        self.local_epochs = 3
        self.batch_size = 64
        self.learning_rate = 0.01
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 差分隐私 (DP) 参数
        self.dp_enabled = True
        self.privacy_epsilon = 2.0  # 隐私预算 Epsilon
        self.max_grad_norm = 1.0  # 梯度裁剪范数
        self.delta = 1e-4  # 隐私误差 Delta

        # Fed-Kalman 特定参数
        self.kalman_process_noise = 1e-4  # 过程噪声 (Q)
        self.kalman_initial_cov = 1.0  # 初始状态协方差 (P)


# --- 2. 模型定义 (简单的CNN) ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 7 * 7, 512)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x


# --- 3. 数据加载与 Non-IID 划分 --- （分得相同类别的客户算具有完全系统的数据集，deprecated）
# def get_data_loaders(num_clients):
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.5,), (0.5,))
#     ])
#
#     train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
#     test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
#
#     # 创建 Non-IID 数据分布 (每个客户端只有2个类别的样本)
#     labels = np.array(train_dataset.targets)
#     client_indices = [[] for _ in range(num_clients)]
#
#     # 每两个类别分给一个客户端。注意这里实现的分割是分到同一类别的不同客户端都拥有该类别下的所有样本，实际在医疗场景中
#     for i in range(num_clients):
#         class1_idx = i % 10
#         class2_idx = (i + 1) % 10
#
#         idx1 = np.where(labels == class1_idx)[0]
#         idx2 = np.where(labels == class2_idx)[0]
#
#         all_indices = np.concatenate((idx1, idx2))
#         np.random.shuffle(all_indices)
#         client_indices[i] = all_indices.tolist()
#
#     client_loaders = []
#     for indices in client_indices:
#         subset = Subset(train_dataset, indices)
#         loader = DataLoader(subset, batch_size=Config().batch_size, shuffle=True)
#         client_loaders.append(loader)
#
#     test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
#
#     return client_loaders, test_loader


# --- 4. 客户端训练 ---
def client_update(client_model, optimizer, train_loader, config, noise_multiplier):
    client_model.train()

    if config.dp_enabled:
        privacy_engine = PrivacyEngine()
        client_model, optimizer, train_loader = privacy_engine.make_private(
            module=client_model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier= noise_multiplier,
            max_grad_norm=config.max_grad_norm,
        )

    for epoch in tqdm(range(config.local_epochs), desc='当前客户端训练进度：', leave=False):
        for data, target in train_loader:
            data, target = data.to(config.device), target.to(config.device)
            optimizer.zero_grad()
            output = client_model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            optimizer.step()

    # if config.dp_enabled:
    #     privacy_engine.detach()
    if config.dp_enabled:
        return client_model._module.state_dict()
    else:
        return client_model.state_dict()


def find_noise_multiplier(
        target_epsilon: float,
        target_delta: float,
        sample_rate: float,
        steps: int,
        epsilon_tolerance: float = 0.01
) -> float:
    """
    通过二分搜索找到满足目标 (epsilon, delta) 的 noise_multiplier。
    """
    low = 0.0
    high = 1000.0  # 增大上界以适应更严格的隐私或更多的步数
    accountant = RDPAccountant()

    # 二分搜索迭代次数可以固定，例如100次，以保证足够的精度
    for _ in range(100):
        mid_noise = (low + high) / 2
        if mid_noise == 0.0:
            # 噪声为0时epsilon无限大，不满足任何有限的目标epsilon
            # 将low设为一个极小值继续搜索
            low = 1e-5
            continue

        # 直接调用get_epsilon并传入所有参数，包括mid_noise
        epsilon = accountant.get_epsilon(
            delta=target_delta,
            noise_multiplier=mid_noise,
            sample_rate=sample_rate,
            steps=steps
        )

        if epsilon > target_epsilon:
            # Epsilon消耗过大 -> 噪声太小 -> 需要增大噪声，更新搜索下界
            low = mid_noise
        else:
            # Epsilon消耗达标 -> 噪声足够 -> 尝试减小噪声，更新搜索上界
            high = mid_noise

    # 返回high可以确保最终epsilon一定小于等于目标epsilon
    return high


# --- 5. 服务器聚合逻辑 ---
class Server:
    def __init__(self, config):
        self.config = config
        self.global_model = SimpleCNN().to(config.device)

        # Fed-Kalman 状态变量
        self.kalman_state = self.flatten_model(self.global_model)
        self.kalman_cov = torch.full_like(self.kalman_state, config.kalman_initial_cov)

    def flatten_model(self, model):
        return torch.cat([p.data.view(-1) for p in model.parameters()])

    def unflatten_model(self, flat_params, model):
        pointer = 0
        for param in model.parameters():
            num_elements = param.numel()
            param.data = flat_params[pointer:pointer + num_elements].view_as(param).data
            pointer += num_elements

    def aggregate_dp_fedavg(self, client_updates):
        """基线方法：DP-FedAvg"""
        global_dict = self.global_model.state_dict()
        for k in global_dict.keys():
            global_dict[k] = torch.stack([client_updates[i][k] for i in range(len(client_updates))], 0).mean(0)
        self.global_model.load_state_dict(global_dict)

    def aggregate_fed_kalman(self, client_updates, noise_multiplier):
        """提议方法：Fed-Kalman"""
        # 1. 预测步骤 (Predict)
        # 状态预测: x_k|k-1 = x_k-1|k-1 (假设没有控制输入)
        predicted_state = self.kalman_state
        # 协方差预测: P_k|k-1 = P_k-1|k-1 + Q
        predicted_cov = self.kalman_cov + self.config.kalman_process_noise

        # 2. 更新步骤 (Update)
        # 计算测量值 (Measurement): z_k
        # 这是对所有带噪客户端更新进行平均的结果
        avg_update_flat = torch.stack(
            [self.flatten_model(self.load_temp_model(update)) for update in client_updates]).mean(0)
        measurement = avg_update_flat

        # 计算测量噪声协方差 (R)
        # R = (sigma / C)^2, sigma是DP噪声标准差, C是客户端数量
        # Opacus中 noise_multiplier = sigma / max_grad_norm
        sigma = noise_multiplier * self.config.max_grad_norm
        measurement_noise_var = (sigma / self.config.num_clients) ** 2
        R = torch.full_like(self.kalman_state, measurement_noise_var)

        # 计算卡尔曼增益 (K)
        # K = P_k|k-1 / (P_k|k-1 + R)
        kalman_gain = predicted_cov / (predicted_cov + R)

        # 更新状态估计 (x_k|k)
        # x_k|k = x_k|k-1 + K * (z_k - x_k|k-1)
        innovation = measurement - predicted_state
        self.kalman_state = predicted_state + kalman_gain * innovation

        # 更新协方差估计 (P_k|k)
        # P_k|k = (1 - K) * P_k|k-1
        self.kalman_cov = (1 - kalman_gain) * predicted_cov

        # 将更新后的状态加载回全局模型
        self.unflatten_model(self.kalman_state, self.global_model)

    def load_temp_model(self, state_dict):
        model = SimpleCNN()
        model.load_state_dict(state_dict)
        return model


# --- 6. 评估函数 ---
def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return 100 * correct / total


# --- 7. 主训练循环 ---
def run_experiment(aggregation_method):
    print(f"\n--- Running Experiment: {aggregation_method} ---")

    config = Config()
    print(f"{aggregation_method}使用设备: {config.device}")

    server = Server(config)
    client_loaders, test_loader, client_samples_count = utils.get_data_loaders(config.num_clients)

    if config.dp_enabled:
        print("\n--- 2. 计算样本级差分隐私所需噪声 ---")
        # 计算平均每客户端样本数和批次数，用于估算
        avg_client_samples = sum(client_samples_count) / config.num_clients
        avg_steps_per_epoch = math.ceil(avg_client_samples / config.batch_size)

        # 计算批次采样率
        sample_rate = config.batch_size / avg_client_samples

        # 计算样本级DP的总步数
        record_level_steps = config.num_rounds * config.local_epochs * avg_steps_per_epoch

        print(f"平均采样率 (q): {sample_rate:.4f}")
        print(f"样本级总步数 (T): {record_level_steps}")
        print(f"目标隐私预算: Epsilon={config.privacy_epsilon}, Delta={config.delta}")

        global_noise_multiplier = find_noise_multiplier(
            target_epsilon=config.privacy_epsilon,
            target_delta=config.delta,
            sample_rate=sample_rate,
            steps=record_level_steps
        )
        print(f"计算得出的全局噪声乘数: {global_noise_multiplier:.4f}")

    accuracies = []

    for round_num in tqdm(range(config.num_rounds), desc='联邦聚合轮数：'):
        client_updates = []
        for i in tqdm(range(config.num_clients), desc='客户端训练总进度：', leave=False):
            client_model = SimpleCNN().to(config.device)
            client_model.load_state_dict(server.global_model.state_dict())

            # 确保模型是有效的Opacus模型
            if not ModuleValidator.is_valid(client_model):
                client_model = ModuleValidator.fix(client_model)

            optimizer = optim.SGD(client_model.parameters(), lr=config.learning_rate)

            # 获取客户端数据加载器
            sample_size = len(client_loaders[i].dataset)
            data_loader_with_sample_rate = DataLoader(
                client_loaders[i].dataset,
                batch_size=config.batch_size,
                generator=torch.Generator().manual_seed(i * 100 + round_num)  # 保证可复现
            )

            update = client_update(client_model, optimizer, data_loader_with_sample_rate, config, global_noise_multiplier)
            client_updates.append(update)

        # 服务器聚合
        if aggregation_method == 'DP-FedAvg':
            server.aggregate_dp_fedavg(client_updates)
        elif aggregation_method == 'Fed-Kalman':
            server.aggregate_fed_kalman(client_updates, global_noise_multiplier)

        # 评估
        acc = evaluate(server.global_model, test_loader, config.device)
        accuracies.append(acc)
        # print(f"Round {round_num+1}/{config.num_rounds}, Accuracy: {acc:.2f}%")

    return accuracies


# --- 8. 运行与绘图 ---
if __name__ == "__main__":

    # 运行基线实验
    fedavg_accuracies = run_experiment('DP-FedAvg')

    # 运行Fed-Kalman实验
    fedkalman_accuracies = run_experiment('Fed-Kalman')

    # 绘图比较
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, Config().num_rounds + 1), fedavg_accuracies, marker='o', linestyle='--',
             label='DP-FedAvg (Baseline)')
    plt.plot(range(1, Config().num_rounds + 1), fedkalman_accuracies, marker='s', linestyle='-',
             label='Fed-Kalman (Proposed)')
    plt.title('Comparison of DP-FedAvg and Fed-Kalman on Fashion-MNIST (Non-IID)')
    plt.xlabel('Communication Round')
    plt.ylabel('Test Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.ylim(10, 90)  # 设置Y轴范围以便更好地观察
    plt.show()

