# 学术文献知识图谱构建与图神经网络分类系统 (KG_Course_Project)

本项目是基于大语言模型 (DeepSeek) 和图同构神经网络 (GINE) 实现的跨年份学术会议知识图谱提取与分类系统。

## 目录结构
- `data/ready_data/`：包含 2023、2024、2025 的图谱数据（已清洗完毕的三元组 CSV 格式）。其中 2025 数据加入了仿真类别数据。
- `src/`：
  - `extract_icml.py`：基于 LangChain 的 LCEL 管道，调用 DeepSeek 大模型从 PDF 中提取 JSON 结构化三元组代码。
  - `prepare_data.py`：清洗合并数据集的代码。
  - `classification.py`：使用 PyTorch Geometric 构建 GINE 网络，将句子通过 paraphrase-MiniLM-L6-v2 进行 384 维映射，进行带有边界特征聚合的训练与交叉测试。
- `report/`：包含多维度数据流转图表、训练曲线，以及最终的万字详细报告 Word 版。

## 环境要求 (requirements)
- Python >= 3.10
- PyTorch >= 2.0.0
- torch_geometric (PyG)
- transformers, sentence-transformers
- langchain, langchain-community
- pdfminer.six
- pandas, matplotlib, scikit-learn

## 如何运行
1. **数据准备**：如果你想重新提取，运行 `extract_icml.py` (需要配置你自己的 DeepSeek API KEY)。数据准备完毕后会在 `ready_data` 下。
2. **模型训练与交叉测试**：
```bash
# 执行 2023 -> 2024 (评估 LLM 引入前的基线与引入后的提升)
python src/classification.py --train_year 2023 --test_year 2024

# 执行 2024 -> 2025 (四分类模拟测试，分析分布偏移)
python src/classification.py --train_year 2024 --test_year 2025
```
