import os
import csv
import time
import json
import random
import argparse

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_mean_pool

# ========================= 配置参数 ========================= #
parser = argparse.ArgumentParser(description="Knowledge Graph Classification Cross-Year")
parser.add_argument('--train_year', type=str, default='2023', help="Year of data for training/validation")
parser.add_argument('--test_year', type=str, default='2024', help="Year of data for testing")
parser.add_argument('--data_root', type=str, default=r'E:\code\keshecode\ready_data', help="Root directory of processed data")
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# 支持的标签映射
label_map = {"AAAI": 0, "CVPR": 1, "NeurIPS": 2, "ICML": 3}
num_classes = len(label_map)

# ========================= 数据处理 ========================= #
def encode_text_list(texts):
    embeddings = encoder.encode(texts, convert_to_tensor=True)
    return embeddings

def csv_to_pyg_data(csv_path, graph_label):
    valid_rows = []
    with open(csv_path, 'r', encoding='utf-8', newline='', errors='ignore') as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return None

        for row in reader:
            if len(row) < 3:
                continue
            valid_rows.append(row[:3])

    if len(valid_rows) == 0:
        return None

    df = pd.DataFrame(valid_rows, columns=['node_1', 'node_2', 'edge'])
    nodes = pd.unique(df[['node_1', 'node_2']].values.ravel())
    if len(nodes) == 0:
        return None
        
    node2id = {name: i for i, name in enumerate(nodes)}
    x = encode_text_list(list(nodes))

    edge_index = []
    edge_attr_text = []
    for _, row in df.iterrows():
        u = node2id[row['node_1']]
        v = node2id[row['node_2']]
        edge_index.append([u, v])
        edge_attr_text.append(row['edge'])

    if len(edge_index) == 0:
        return None

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = encode_text_list(edge_attr_text)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor([graph_label], dtype=torch.long),
    )

def load_dataset(data_root, year, label_map, max_samples_per_class=200):
    data_list = []
    year_dir = os.path.join(data_root, year)
    if not os.path.exists(year_dir):
        print(f"[警告] 目录不存在: {year_dir}")
        return data_list
        
    for conf, label in label_map.items():
        conf_dir = os.path.join(year_dir, conf)
        if not os.path.isdir(conf_dir):
            continue
            
        count = 0
        csv_files = [f for f in os.listdir(conf_dir) if f.endswith('.csv')]
        random.shuffle(csv_files) # 打乱文件顺序，随机采样
        
        for fname in csv_files:
            if count >= max_samples_per_class:
                break
            csv_path = os.path.join(conf_dir, fname)
            data = csv_to_pyg_data(csv_path, label)
            if data is not None:
                data_list.append(data)
                count += 1
                
        print(f"[{year}] {conf}: 加载了 {count} 个样本")
        
    return data_list

# ========================= 模型定义 ========================= #
class GINE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GINE, self).__init__()
        self.edge_lin = torch.nn.Linear(in_channels, hidden_channels)

        self.node_mlp1 = torch.nn.Sequential(
            torch.nn.Linear(in_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels)
        )
        self.node_mlp2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels)
        )

        self.conv1 = GINEConv(self.node_mlp1, edge_dim=hidden_channels)
        self.conv2 = GINEConv(self.node_mlp2, edge_dim=hidden_channels)

        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_attr, batch):
        edge_attr_proj = self.edge_lin(edge_attr)
        x = self.conv1(x, edge_index, edge_attr_proj)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_attr_proj)
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return x

def main():
    print(f"=== 课设图分类任务: Train {args.train_year} -> Test {args.test_year} ===")
    
    # 1. 加载数据
    print(f"\n加载训练/验证集 ({args.train_year})...")
    train_data_list = load_dataset(args.data_root, args.train_year, label_map)
    
    print(f"\n加载测试集 ({args.test_year})...")
    test_data_list = load_dataset(args.data_root, args.test_year, label_map)
    
    if len(train_data_list) == 0 or len(test_data_list) == 0:
        raise RuntimeError("训练集或测试集为空，请检查数据目录。")

    # 从训练集中切分验证集 (15%)
    train_data, val_data = train_test_split(
        train_data_list,
        test_size=0.15,
        stratify=[d.y.item() for d in train_data_list],
        random_state=int(time.time()),
    )
    
    # 测试集不切分，全量测试
    test_data = test_data_list

    from collections import Counter
    def print_label_distribution(name, dataset):
        labels = [d.y.item() for d in dataset]
        counter = Counter(labels)
        print(f"{name} 样本数: {len(labels)}")
        for label, count in counter.items():
            conf_name = list(label_map.keys())[list(label_map.values()).index(label)]
            print(f"  {conf_name} (标签 {label})：{count} 个")

    print("\n--- 数据集划分 ---")
    print_label_distribution("训练集", train_data)
    print_label_distribution("验证集", val_data)
    print_label_distribution("测试集", test_data)

    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=16)
    test_loader = DataLoader(test_data, batch_size=16)

    # 2. 初始化模型
    model = GINE(
        in_channels=encoder.get_sentence_embedding_dimension(),
        hidden_channels=64,
        out_channels=num_classes,
    ).to(device)

    learn = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learn, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    def train():
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = criterion(out, batch.y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
        return total_loss / len(train_loader.dataset)

    @torch.no_grad()
    def evaluate(loader):
        model.eval()
        all_preds = []
        all_labels = []
        all_probs = []

        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            probs = F.softmax(out, dim=1)
            pred = out.argmax(dim=1)

            all_preds.append(pred.cpu())
            all_labels.append(batch.y.view(-1).cpu())
            all_probs.append(probs.cpu())

        if len(all_labels) == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0, [], []

        y_true = torch.cat(all_labels).numpy()
        y_pred = torch.cat(all_preds).numpy()
        y_prob = torch.cat(all_probs).numpy()

        acc = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        
        # OVR AUC 计算
        try:
            # 获取 y_true 中出现过的类别
            unique_classes = sorted(list(set(y_true)))
            if len(unique_classes) == 1:
                roc_auc = float("nan")
            else:
                # 只有当样本中有大于等于2个类别时才计算AUC
                # 为了防止 y_prob 的列数与 label_map 的数量不匹配，直接用 OVR
                roc_auc = roc_auc_score(y_true, y_prob, multi_class="ovr", labels=list(range(num_classes)))
        except Exception as e:
            # print(f"AUC 计算失败: {e}")
            roc_auc = float("nan")

        return acc, precision, recall, f1, roc_auc, y_true, y_prob

    # 3. 训练主循环 (含早停)
    best_val_acc = 0.0
    patience = 10
    counter = 0
    train_losses = []
    val_accuracies = []

    print("\n--- 开始训练 ---")
    for epoch in range(1, 51):
        loss = train()
        acc, precision, recall, f1, *_ = evaluate(val_loader)
        train_losses.append(loss)
        val_accuracies.append(acc)
        print(f"Epoch {epoch:02d} | Loss: {loss:.4f} | Val Acc: {acc:.4f}")

        if acc > best_val_acc:
            best_val_acc = acc
            counter = 0
            torch.save(model.state_dict(), "best_gine_model.pth")
        else:
            counter += 1
            if counter >= patience:
                print(f"验证集连续 {patience} 轮未提升，触发 Early Stopping。")
                break

    # 4. 加载最佳权重并测试
    print("\n--- 加载最佳模型并在测试集上进行最终评估 ---")
    model.load_state_dict(torch.load("best_gine_model.pth"))
    acc, prec, rec, f1, roc_auc, y_true, y_prob = evaluate(test_loader)

    print(f"\n[{args.train_year} -> {args.test_year}] 最终评估指标:")
    print(f"准确率 (Accuracy)  : {acc:.4f}")
    print(f"精确率 (Precision) : {prec:.4f}")
    print(f"召回率 (Recall)    : {rec:.4f}")
    print(f"F1 分数 (F1-Score): {f1:.4f}")
    print(f"ROC AUC (OVR)      : {roc_auc:.4f}")

    # 保存结果到 JSON
    results = {
        "train_year": args.train_year,
        "test_year": args.test_year,
        "metrics": {
            "Accuracy": acc,
            "Precision_macro": prec,
            "Recall_macro": rec,
            "F1_macro": f1,
            "AUC_ovr": roc_auc if not pd.isna(roc_auc) else None
        }
    }
    result_filename = f"results_{args.train_year}_to_{args.test_year}.json"
    with open(result_filename, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"实验结果已保存至 {result_filename}")

    # 5. 绘图
    # 绘制 Loss 曲线
    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss Curve ({args.train_year})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"loss_curve_{args.train_year}.png")
    # plt.show() # 在无头环境下不显示

    # 绘制 ROC 曲线
    from sklearn.preprocessing import label_binarize
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
    
    plt.figure(figsize=(7, 5))
    valid_classes = 0
    for i in range(num_classes):
        if i in set(y_true):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            roc_auc_val = auc(fpr, tpr)
            conf_name = list(label_map.keys())[list(label_map.values()).index(i)]
            plt.plot(fpr, tpr, label=f"{conf_name} (AUC = {roc_auc_val:.2f})")
            valid_classes += 1
            
    if valid_classes > 0:
        plt.plot([0, 1], [0, 1], "k--", lw=1)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"Multi-class ROC Curve ({args.train_year}->{args.test_year})")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"roc_curve_{args.train_year}_to_{args.test_year}.png")
        # plt.show()

if __name__ == "__main__":
    main()
