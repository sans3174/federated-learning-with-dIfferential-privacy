import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from tqdm import tqdm
import warnings
import math
import utils

warnings.filterwarnings("ignore")


# --- 1. 配置参数 ---
class Config:
    def __init__(self):
        self.num_clients = 10
        self.num_rounds = 5
        self.local_epochs = 3
        self.batch_size = 64
        self.learning_rate = 0.01
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"当前使用设备: {self.device}")

        # 差分隐私 (DP) 参数
        self.dp_enabled = True
        self.privacy_epsilon = 5.0  # 隐私预算 Epsilon
        self.max_grad_norm = 1.0  # 梯度裁剪范数
        self.delta = 1e-5  # 隐私误差 Delta

        # Fed-Kalman 基本参数
        # self.kalman_process_noise = 1e-4  # 过程噪声 (Q)
        # self.kalman_initial_cov = 1.0  # 初始状态协方差 (P)

        # --- Fed-Kalman with Momentum & Adaptive Noise 参数 ---
        self.momentum_beta = 0.9  # 动量参数 Beta，通常取 0.9
        self.adaptive_q_base = 1e-5  # 自适应过程噪声的基础值
        self.adaptive_q_scale = 1e-3  # 自适应过程噪声的缩放因子

        # --- Fed-Kalman with IMM 参数 ---
        self.imm_num_models = 2  # IMM中的模型数量 (例如：一个高噪声，一个低噪声)
        self.imm_process_noises = [1e-2, 1e-5]  # 每个模型对应的过程噪声 Q
        # 状态转移概率矩阵 P_ij = P(从模型i转移到模型j)
        self.imm_transition_prob = torch.tensor([[0.95, 0.05],
                                                 [0.05, 0.95]], device=self.device)


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


# --- 3. 数据加载与 Non-IID 划分 放到 utils.py 中 ---



# --- 4. 客户端训练 ---
def client_update(client_model, optimizer, train_loader, config):
    client_model.train()

    if config.dp_enabled:
        privacy_engine = PrivacyEngine()
        client_model, optimizer, train_loader = privacy_engine.make_private(
            module=client_model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=calculate_noise_multiplier(config),
            max_grad_norm=config.max_grad_norm,
        )

    for epoch in tqdm(range(config.local_epochs), desc='当前客户端训练进度', position=2, leave = False):
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
        # Opacus v1.0+ uses _module attribute to access the original model
        return client_model._module.state_dict()
    else:
        return client_model.state_dict()


def calculate_noise_multiplier(config):
    # Opacus自带的工具函数需要一个隐私引擎实例，我们这里简化计算
    # 实际项目中建议使用更精确的库函数
    # 这是一个近似值，用于演示
    return 0.5 * np.sqrt(2 * np.log(1.25 / config.delta)) / config.privacy_epsilon


# --- 5. 服务器聚合逻辑 ---
class Server:
    def __init__(self, config):
        self.config = config
        self.global_model = SimpleCNN().to(config.device)
        self.noise_multiplier = calculate_noise_multiplier(config)
        model_param_dim = len(self.flatten_model(self.global_model))

        # Fed-Kalman 状态变量
        # self.kalman_state = self.flatten_model(self.global_model)
        # self.kalman_cov = torch.full_like(self.kalman_state, config.kalman_initial_cov)

        # 方案一：Adaptive Noise所需的状态 (模型参数 + 动量)
        # 状态向量维度是模型参数的两倍
        adaptive_state_dim = model_param_dim * 2
        self.kalman_state_adaptive = torch.zeros(adaptive_state_dim, device=config.device)
        self.kalman_state_adaptive[:model_param_dim] = self.flatten_model(self.global_model).clone()  # 初始化模型权重
        self.kalman_cov_adaptive = torch.full((adaptive_state_dim,), 1.0, device=config.device)  # 初始化协方差
        self.prev_momentum = torch.zeros(model_param_dim, device=config.device)  # 用于计算加速度

        # 方案二：IMM所需的状态
        self.imm_model_probs = torch.full((config.imm_num_models,), 1.0 / config.imm_num_models, device=config.device)
        self.imm_states = []
        self.imm_covs = []
        for _ in range(config.imm_num_models):
            # 每个模型都有自己的状态 (权重+动量) 和协方差
            state = torch.zeros(adaptive_state_dim, device=config.device)
            state[:model_param_dim] = self.flatten_model(self.global_model).clone()
            self.imm_states.append(state)
            self.imm_covs.append(torch.full((adaptive_state_dim,), 1.0, device=config.device))

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

    def aggregate_fed_kalman_adaptive_noise(self, client_updates):
        """方案一：基于动量和自适应过程噪声的Fed-Kalman"""
        model_param_dim = len(self.flatten_model(self.global_model))
        beta = self.config.momentum_beta
        noise_multiplier = self.noise_multiplier

        # --- 1. 预测步骤 (Predict) ---
        # 状态预测: w_k|k-1 = w_k-1|k-1 + m_k-1|k-1
        #           m_k|k-1 = beta * m_k-1|k-1
        weights_prev = self.kalman_state_adaptive[:model_param_dim]
        momentum_prev = self.kalman_state_adaptive[model_param_dim:]

        predicted_weights = weights_prev + momentum_prev
        predicted_momentum = beta * momentum_prev
        predicted_state = torch.cat([predicted_weights, predicted_momentum])

        # 协方差预测 (对角线近似):
        # P_w_pred = P_w + P_m
        # P_m_pred = beta^2 * P_m
        cov_w_prev = self.kalman_cov_adaptive[:model_param_dim]
        cov_m_prev = self.kalman_cov_adaptive[model_param_dim:]

        predicted_cov_w = cov_w_prev + cov_m_prev
        predicted_cov_m = (beta ** 2) * cov_m_prev

        # 动态调整过程噪声 Q
        acceleration = momentum_prev - self.prev_momentum
        # 使用 torch.linalg.norm 替代 torch.norm
        q_val = self.config.adaptive_q_base + self.config.adaptive_q_scale * torch.linalg.norm(acceleration) ** 2
        Q = torch.full_like(predicted_state, q_val)

        predicted_cov = torch.cat([predicted_cov_w, predicted_cov_m]) + Q

        # --- 2. 更新步骤 (Update) ---
        # 测量值 z_k 是聚合后的带噪更新，对应于对“速度/动量”的测量
        avg_update_params = torch.stack(
            [self.flatten_model(self.load_temp_model(update)) for update in client_updates]).mean(0)
        # 测量值是模型更新量 (delta), 而非模型参数本身
        measurement = avg_update_params - weights_prev

        # 测量噪声协方差 R
        sigma = noise_multiplier * self.config.max_grad_norm
        measurement_noise_var = (sigma / self.config.num_clients) ** 2
        R = torch.full((model_param_dim,), measurement_noise_var, device=self.config.device)

        # H = [0, I], H P H^T + R = P_m_pred + R
        innovation_cov = predicted_cov[model_param_dim:] + R

        # 卡尔曼增益 K = P H^T (H P H^T + R)^-1
        # K_w = P_cross / innovation_cov = P_m_pred / innovation_cov (近似)
        # K_m = P_m_pred / innovation_cov
        # P_cross 是 w 和 m 的交叉协方差，在对角近似下为0，但实际有关联。
        # 一个常用且有效的近似是 P_cross ≈ P_m_pred
        kalman_gain_m = predicted_cov[model_param_dim:] / innovation_cov
        kalman_gain_w = kalman_gain_m  # 使用近似
        kalman_gain = torch.cat([kalman_gain_w, kalman_gain_m])

        # 更新状态估计 x_k|k = x_k|k-1 + K * (z_k - H x_k|k-1)
        # H x_k|k-1 = m_k|k-1
        innovation = measurement - predicted_momentum
        updated_state = predicted_state + kalman_gain * innovation.repeat(2)

        # 更新协方差估计 P_k|k = (I - K H) P_k|k-1
        # (I - K H) 是一个分块矩阵，直接计算对角线元素
        updated_cov_w = (1 - kalman_gain_w) * predicted_cov[:model_param_dim]
        updated_cov_m = (1 - kalman_gain_m) * predicted_cov[model_param_dim:]
        updated_cov = torch.cat([updated_cov_w, updated_cov_m])

        # 保存状态
        self.prev_momentum = momentum_prev.clone()
        self.kalman_state_adaptive = updated_state
        self.kalman_cov_adaptive = updated_cov

        # 将更新后的状态加载回全局模型
        self.unflatten_model(self.kalman_state_adaptive[:model_param_dim], self.global_model)

    def aggregate_fed_kalman_imm(self, client_updates):
        """方案二：基于交互式多模型(IMM)的Fed-Kalman"""
        model_param_dim = len(self.flatten_model(self.global_model))
        beta = self.config.momentum_beta
        num_models = self.config.imm_num_models
        noise_multiplier = self.noise_multiplier


        # --- 1. 交互/混合步骤 (Interaction/Mixing) ---
        # c_bar_j = sum_i P(m_k=j | m_{k-1}=i) * p_i
        mixing_probs_numerator = self.config.imm_transition_prob * self.imm_model_probs
        mixing_denominators = mixing_probs_numerator.sum(dim=1, keepdim=True)
        # 避免除以零
        mixing_denominators[mixing_denominators == 0] = 1e-8
        mixing_probs = mixing_probs_numerator / mixing_denominators

        mixed_states = []
        mixed_covs = []
        for j in range(num_models):
            # 对每个模型j，混合所有模型i的上一轮状态
            mixed_state_j = torch.zeros_like(self.imm_states[0])
            for i in range(num_models):
                mixed_state_j += self.imm_states[i] * mixing_probs[j, i]
            mixed_states.append(mixed_state_j)

            mixed_cov_j = torch.zeros_like(self.imm_covs[0])
            for i in range(num_models):
                state_diff = self.imm_states[i] - mixed_state_j
                mixed_cov_j += mixing_probs[j, i] * (self.imm_covs[i] + state_diff ** 2)
            mixed_covs.append(mixed_cov_j)

        # --- 2. 滤波步骤 (Filtering) ---
        # 为每个模型并行运行一个卡尔曼滤波器
        updated_states_j = []
        updated_covs_j = []
        likelihoods_j = []

        # 测量值 (对所有模型都相同)
        weights_prev_overall = self.flatten_model(self.global_model)  # 使用上一轮的最终融合结果
        avg_update_params = torch.stack(
            [self.flatten_model(self.load_temp_model(update)) for update in client_updates]).mean(0)
        measurement = avg_update_params - weights_prev_overall

        sigma = noise_multiplier * self.config.max_grad_norm
        measurement_noise_var = (sigma / self.config.num_clients) ** 2
        R = torch.full((model_param_dim,), measurement_noise_var, device=self.config.device)

        for j in range(num_models):
            # a. 预测
            weights_prev_j = mixed_states[j][:model_param_dim]
            momentum_prev_j = mixed_states[j][model_param_dim:]
            predicted_weights_j = weights_prev_j + momentum_prev_j
            predicted_momentum_j = beta * momentum_prev_j
            predicted_state_j = torch.cat([predicted_weights_j, predicted_momentum_j])

            cov_w_prev_j = mixed_covs[j][:model_param_dim]
            cov_m_prev_j = mixed_covs[j][model_param_dim:]
            predicted_cov_w_j = cov_w_prev_j + cov_m_prev_j
            predicted_cov_m_j = (beta ** 2) * cov_m_prev_j

            Q_j = torch.full_like(predicted_state_j, self.config.imm_process_noises[j])
            predicted_cov_j = torch.cat([predicted_cov_w_j, predicted_cov_m_j]) + Q_j

            # b. 更新
            innovation_cov_j = predicted_cov_j[model_param_dim:] + R

            # 计算似然度 L_j = N(z_k; H x_k|k-1, S_k)
            innovation_j = measurement - predicted_momentum_j
            # 简化假设：各维度独立，计算对数似然以保证数值稳定性
            log_likelihood = -0.5 * torch.sum(
                torch.log(2 * math.pi * innovation_cov_j) + (innovation_j ** 2 / innovation_cov_j))
            likelihoods_j.append(torch.exp(log_likelihood))

            kalman_gain_m_j = predicted_cov_j[model_param_dim:] / innovation_cov_j
            kalman_gain_w_j = kalman_gain_m_j
            kalman_gain_j = torch.cat([kalman_gain_w_j, kalman_gain_m_j])

            updated_state_j = predicted_state_j + kalman_gain_j * innovation_j.repeat(2)
            updated_states_j.append(updated_state_j)

            updated_cov_w_j = (1 - kalman_gain_w_j) * predicted_cov_j[:model_param_dim]
            updated_cov_m_j = (1 - kalman_gain_m_j) * predicted_cov_j[model_param_dim:]
            updated_covs_j.append(torch.cat([updated_cov_w_j, updated_cov_m_j]))

        # --- 3. 模型概率更新 ---
        # 确保 likelihoods 是一个 tensor
        likelihoods = torch.stack(likelihoods_j) if isinstance(likelihoods_j, list) else likelihoods_j
        # 确保 mixing_denominators 维度正确
        mixing_denominators_squeezed = mixing_denominators.squeeze()
        if mixing_denominators_squeezed.dim() == 0:
            mixing_denominators_squeezed = mixing_denominators_squeezed.unsqueeze(0)

        new_model_probs = likelihoods * mixing_denominators_squeezed

        # 防止所有似然都为0导致NaN
        if new_model_probs.sum() == 0:
            # 如果所有模型概率都消失，则重置为均匀分布
            self.imm_model_probs = torch.full_like(self.imm_model_probs, 1.0 / num_models)
        else:
            self.imm_model_probs = new_model_probs / new_model_probs.sum()

        # 防止概率消失
        self.imm_model_probs = torch.clamp(self.imm_model_probs, min=1e-8)
        self.imm_model_probs /= self.imm_model_probs.sum()

        # --- 4. 状态融合 ---
        final_state = torch.zeros_like(self.imm_states[0])
        for j in range(num_models):
            final_state += updated_states_j[j] * self.imm_model_probs[j]

        final_cov = torch.zeros_like(self.imm_covs[0])
        for j in range(num_models):
            state_diff = updated_states_j[j] - final_state
            final_cov += self.imm_model_probs[j] * (updated_covs_j[j] + state_diff ** 2)

        # 保存每个模型的状态以备下一轮使用
        self.imm_states = updated_states_j
        self.imm_covs = updated_covs_j

        # 将最终融合后的状态加载回全局模型
        self.unflatten_model(final_state[:model_param_dim], self.global_model)

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
def run_experiment(aggregation_method, config):
    server = Server(config)
    client_loaders, test_loader, client_samples_count = utils.get_data_loaders(config)

    accuracies = []

    print(f"\n--- Running Experiment: {aggregation_method} ---")

    for round_num in tqdm(range(config.num_rounds), desc='联邦聚合轮数进度', position=0):
        client_updates = []
        for i in tqdm(range(config.num_clients), desc='单轮聚合内已训练客户端数目', position=1):
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

            update = client_update(client_model, optimizer, data_loader_with_sample_rate, config)
            client_updates.append(update)

        # 服务器聚合
        if aggregation_method == 'DP-FedAvg':
            server.aggregate_dp_fedavg(client_updates)
        elif aggregation_method == 'Fed-Kalman':
            server.aggregate_fed_kalman_adaptive_noise(client_updates)

        # 评估
        acc = evaluate(server.global_model, test_loader, config.device)
        accuracies.append(acc)
        # print(f"Round {round_num+1}/{config.num_rounds}, Accuracy: {acc:.2f}%")

    return accuracies


# --- 8. 运行与绘图 ---
if __name__ == "__main__":
    config = Config()
    # 运行基线实验
    fedavg_accuracies = run_experiment('DP-FedAvg', config)

    # 运行Fed-Kalman实验
    fedkalman_accuracies = run_experiment('Fed-Kalman', config)

    # 绘图比较
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, config.num_rounds + 1), fedavg_accuracies, marker='o', linestyle='--',
             label='DP-FedAvg (Baseline)')
    plt.plot(range(1, config.num_rounds + 1), fedkalman_accuracies, marker='s', linestyle='-',
             label='Fed-Kalman (Proposed)')
    plt.title('Comparison of DP-FedAvg and Fed-Kalman on Fashion-MNIST (Non-IID)')
    plt.xlabel('Communication Round')
    plt.ylabel('Test Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.ylim(10, 90)  # 设置Y轴范围以便更好地观察
    plt.show()

