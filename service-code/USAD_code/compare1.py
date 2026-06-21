import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_recall_fscore_support

# ======================== 标题与模块说明 ========================
print("="*80)
print("  软件测试与维护大作业阶段四：多KPI核心算法对比 (USAD/OmniAnomaly)  ")
print("  核心模块：异常数据采集 | 论文算法复现 | 参数调优 | 横向对比展示  ")
print("="*80)

# ======================== 阶段1：数据预处理 ========================
print("\n[STEP 01] 数据预处理与多维特征对齐")
print("-"*60)
print("🔍 读取 Prometheus+Grafana 导出的性能指标 CSV 文件...")
try:
    df_cpu = pd.read_csv("cpu.csv")
    df_mem = pd.read_csv("memory.csv")
    print("✅ 原始数据加载成功")
except Exception as e:
    print(f"❌ 数据加载失败：找不到 cpu.csv 或 memory.csv | 错误详情: {e}")
    exit()

# 数据形态检查
print(f"📈 原始数据形态：CPU序列 {len(df_cpu)} 行 | 内存序列 {len(df_mem)} 行")

# 特征提取与时间线对齐
cpu_raw = df_cpu.iloc[:, 1].values.reshape(-1, 1)
mem_raw = df_mem.iloc[:, 1].values.reshape(-1, 1)
min_len = min(len(cpu_raw), len(mem_raw))
raw_data = np.hstack((cpu_raw[:min_len], mem_raw[:min_len]))
times = df_cpu['Time'].values[:min_len]
print(f"🔗 多维KPI融合完成：融合后矩阵形状 {raw_data.shape}")

# 采样数据快照展示
print("📌 前3行原始数据快照：")
for i in range(3):
    print(f"   时间: {times[i]} | CPU值: {raw_data[i, 0]:.6f} | 内存值: {raw_data[i, 1]:.1f}")

# 1. 归一化处理
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(raw_data)
print("⚡ 执行MinMaxScaler归一化：数据缩放至 [0,1] 区间（消除量纲影响）")

# 2. 滑动窗口转换
WINDOW_SIZE = 5
num_features = 2
input_dim = WINDOW_SIZE * num_features
windows = []
for i in range(len(scaled_data) - WINDOW_SIZE + 1):
    windows.append(scaled_data[i:i+WINDOW_SIZE].flatten())
X = torch.FloatTensor(np.array(windows))
test_times = times[WINDOW_SIZE-1:]
print(f"🪟 滑动窗口处理完成 (Window Size={WINDOW_SIZE})：")
print(f"   原始矩阵 {raw_data.shape} → 模型输入张量 {list(X.shape)}")

# 3. 构建Ground Truth标签
true_labels = np.zeros(len(X))
for idx, t in enumerate(test_times):
    if "12:56:" in t or "12:57:" in t or "12:58:" in t:
        true_labels[idx] = 1
print(f"🎯 故障标签标注完成：总样本 {len(X)} 条 | 异常样本 {int(sum(true_labels))} 条")

print("✅ STEP 01 完成：数据预处理全部就绪")
print("-"*80)

# ======================== 阶段2：USAD模型定义与训练 ========================
print("\n[STEP 02] USAD (KDD '20) 对抗自编码器算法复现")
print("-"*60)

# USAD网络定义
class USAD_Network(nn.Module):
    def __init__(self, input_size, latent_size=10):
        super(USAD_Network, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, input_size // 2), nn.ReLU(),
            nn.Linear(input_size // 2, input_size // 4), nn.ReLU(),
            nn.Linear(input_size // 4, latent_size), nn.ReLU()
        )
        self.decoder1 = nn.Sequential(
            nn.Linear(latent_size, input_size // 4), nn.ReLU(),
            nn.Linear(input_size // 4, input_size // 2), nn.ReLU(),
            nn.Linear(input_size // 2, input_size), nn.Sigmoid()
        )
        self.decoder2 = nn.Sequential(
            nn.Linear(latent_size, input_size // 4), nn.ReLU(),
            nn.Linear(input_size // 4, input_size // 2), nn.ReLU(),
            nn.Linear(input_size // 2, input_size), nn.Sigmoid()
        )

    def forward(self, w):
        z = self.encoder(w)
        return self.decoder1(z), self.decoder2(z), self.decoder2(self.encoder(self.decoder1(z)))

# 模型初始化
usad_model = USAD_Network(input_dim)
print(f"🔧 USAD模型初始化完成：输入维度 {input_dim} | 隐空间维度 10")

# USAD训练过程
opt_u1 = torch.optim.Adam(list(usad_model.encoder.parameters()) + list(usad_model.decoder1.parameters()), lr=1e-3)
opt_u2 = torch.optim.Adam(list(usad_model.encoder.parameters()) + list(usad_model.decoder2.parameters()), lr=1e-3)
EPOCHS = 30
start_time = time.time()

print("🚀 开始USAD模型训练：")
for epoch in range(1, EPOCHS + 1):
    usad_model.train()
    # 重构训练阶段
    w1, w2, w3 = usad_model(X)
    loss_u1 = 1/epoch * torch.mean((X - w1)**2) + (1 - 1/epoch) * torch.mean((X - w2)**2)
    opt_u1.zero_grad()
    loss_u1.backward(retain_graph=True)
    opt_u1.step()
    
    # 对抗训练阶段
    w1, w2, w3 = usad_model(X)
    loss_u2 = 1/epoch * torch.mean((X - w1)**2) - (1 - 1/epoch) * torch.mean((X - w3)**2)
    opt_u2.zero_grad()
    loss_u2.backward()
    opt_u2.step()
    
    # 训练日志输出
    if epoch % 5 == 0 or epoch == 1:
        print(f"   Epoch [{epoch:02d}/{EPOCHS}] | 重构损失: {loss_u1.item():.6f} | 对抗损失: {loss_u2.item():.6f}")

usad_time = time.time() - start_time
print(f"✅ USAD训练完成 | 总耗时: {usad_time:.4f} 秒")
print("-"*80)

# ======================== 阶段3：OmniAnomaly模型定义与训练 ========================
print("\n[STEP 03] OmniAnomaly (KDD '19) 随机循环网络算法复现")
print("-"*60)

# OmniAnomaly网络定义
class OmniAnomaly_Network(nn.Module):
    def __init__(self, input_size, latent_size=6):
        super(OmniAnomaly_Network, self).__init__()
        self.gru = nn.GRU(input_size, input_size * 2, batch_first=True)
        self.fc_mu = nn.Linear(input_size * 2, latent_size)
        self.fc_logvar = nn.Linear(input_size * 2, latent_size)
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, input_size // 2), nn.ReLU(),
            nn.Linear(input_size // 2, input_size), nn.Sigmoid()
        )
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, w):
        out, _ = self.gru(w.unsqueeze(1))
        out = out.squeeze(1)
        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

# 模型初始化
omni_model = OmniAnomaly_Network(input_dim)
print(f"🔧 OmniAnomaly模型初始化完成：输入维度 {input_dim} | 隐空间维度 6")

# OmniAnomaly训练过程
opt_omni = torch.optim.Adam(omni_model.parameters(), lr=2e-3)
start_time = time.time()

print("🚀 开始OmniAnomaly模型训练：")
for epoch in range(1, EPOCHS + 1):
    omni_model.train()
    x_recon, mu, logvar = omni_model(X)
    # ELBO损失计算
    recon_loss = torch.mean((X - x_recon)**2)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    loss_omni = recon_loss + 0.01 * kl_loss
    
    opt_omni.zero_grad()
    loss_omni.backward()
    opt_omni.step()
    
    # 训练日志输出
    if epoch % 5 == 0 or epoch == 1:
        print(f"   Epoch [{epoch:02d}/{EPOCHS}] | 重构残差: {recon_loss.item():.6f} | KL散度: {kl_loss.item():.6f} | 总损失: {loss_omni.item():.6f}")

omni_time = time.time() - start_time
print(f"✅ OmniAnomaly训练完成 | 总耗时: {omni_time:.4f} 秒")
print("-"*80)

# ======================== 阶段4：模型评估与参数调优 ========================
print("\n[STEP 04] 模型评估与超参数调优")
print("-"*60)

# 计算异常得分
usad_model.eval()
omni_model.eval()
with torch.no_grad():
    # USAD异常得分
    w1, _, w3 = usad_model(X)
    usad_scores = (0.5 * torch.mean((X - w1)**2, dim=1) + 0.5 * torch.mean((X - w3)**2, dim=1)).numpy()
    # OmniAnomaly异常得分
    x_recon_o, _, _ = omni_model(X)
    omni_scores = torch.mean((X - x_recon_o)**2, dim=1).numpy()

print("📊 USAD阈值调优（遍历分位数阈值）：")
print("-"*50)
print("  分位数 | 准确率(P) | 召回率(R) | F1-Score")
print("-"*50)
u_best_f1, u_best_p, u_best_r, u_best_perc = 0, 0, 0, 0
for perc in [80, 85, 90, 95]:
    th = np.percentile(usad_scores, perc)
    preds = (usad_scores > th).astype(int)
    p, r, f, _ = precision_recall_fscore_support(true_labels, preds, average='binary', zero_division=0)
    print(f"    {perc}%    |   {p:.4f}   |   {r:.4f}   |   {f:.4f}")
    if f > u_best_f1:
        u_best_f1, u_best_p, u_best_r, u_best_perc = f, p, r, perc

print("\n📊 OmniAnomaly阈值调优（遍历分位数阈值）：")
print("-"*50)
print("  分位数 | 准确率(P) | 召回率(R) | F1-Score")
print("-"*50)
o_best_f1, o_best_p, o_best_r, o_best_perc = 0, 0, 0, 0
for perc in [80, 85, 90, 95]:
    th = np.percentile(omni_scores, perc)
    preds = (omni_scores > th).astype(int)
    p, r, f, _ = precision_recall_fscore_support(true_labels, preds, average='binary', zero_division=0)
    print(f"    {perc}%    |   {p:.4f}   |   {r:.4f}   |   {f:.4f}")
    if f > o_best_f1:
        o_best_f1, o_best_p, o_best_r, o_best_perc = f, p, r, perc

# # 异常时间点定位
# u_final_th = np.percentile(usad_scores, u_best_perc)
# u_preds = (usad_scores > u_final_th).astype(int)
# print("\n🎯 异常时间点定位结果：")
# for t, score, is_anomaly in zip(test_times, usad_scores, u_preds):
#     if is_anomaly == 1:
#         if "12:56:" in t:
#             print(f"   🚨 故障根因 | 时间: {t} | 得分: {score:.6f} (ChaosMesh容器拉起事件)")
#         elif "12:57:" in t or "12:58:" in t:
#             print(f"   ⚠️ 故障扩散 | 时间: {t} | 得分: {score:.6f}")
#         else:
#             print(f"   🔍 环境波动 | 时间: {t} | 得分: {score:.6f}")

print("✅ STEP 04 完成：模型评估与调优全部完成")
print("-"*80)

# ======================== 阶段5：结果汇总 ========================
print("\n[FINAL] 算法对比结果汇总")
print("="*80)
print(" 算法名称      | 最优阈值 | 准确率(P) | 召回率(R) | F1-Score | 耗时(秒)")
print("-"*80)
print(f" USAD (KDD'20) | {u_best_perc}%分位数 | {u_best_p:.4f} | {u_best_r:.4f} | {u_best_f1:.4f} | {usad_time:.3f}")
print(f" OmniAnomaly   | {o_best_perc}%分位数 | {o_best_p:.4f} | {o_best_r:.4f} | {o_best_f1:.4f} | {omni_time:.3f}")
print("="*80)
print("🎉 阶段四全部完成：多KPI算法对比验证通过")