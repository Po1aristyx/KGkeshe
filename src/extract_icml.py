import os
import re
import csv
import json
from io import open
from pathlib import Path
from pdfminer.high_level import extract_text

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ================= 配置 DeepSeek API ================= #
# 请在这里填入你的 DeepSeek API Key
DEEPSEEK_API_KEY = "your_deepseek_api_key_here"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

llm = ChatOpenAI(
    model="deepseek-chat", 
    api_key=DEEPSEEK_API_KEY, 
    base_url=DEEPSEEK_BASE_URL,
    max_tokens=2048,
    temperature=0.1
)

# ================= Prompt 定义 ================= #
SYS_PROMPT = (
    "You are a network graph maker who extracts terms and their relations from a given context. "
    "Format your output ONLY as a valid JSON list. Each element of the list contains a pair of terms "
    "and the relation between them, like the follwing: \n"
    "[\n"
    "   {{\n"
    '       "node_1": "A concept",\n'
    '       "node_2": "A related concept",\n'
    '       "edge": "relationship between the two concepts"\n'
    "   }}\n"
    "]\n\n"
    "context: ```{input}``` \n\n output: "
)

prompt_template = PromptTemplate(template=SYS_PROMPT, input_variables=["input"])
chain = prompt_template | llm

# ================= 主流程 ================= #
pdf_dir = r"E:\code\keshecode\pdfs\pdfs"
out_dir_base = r"E:\code\keshecode\ready_data"

def process_pdf(pdf_path, year, conf="ICML"):
    dest_dir = os.path.join(out_dir_base, year, conf)
    os.makedirs(dest_dir, exist_ok=True)
    filename = os.path.basename(pdf_path).replace('.pdf', '.csv')
    dest_path = os.path.join(dest_dir, filename)

    # 断点续传：如果已经存在该 CSV，直接跳过抽取
    if os.path.exists(dest_path):
        return True

    # 1. 提取文本
    try:
        text = extract_text(pdf_path)
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
        return False

    # 简单清洗
    text = text.replace('\n', ' ').strip()
    if not text:
        return False

    # 2. 文本分块
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    chunks = splitter.split_text(text)

    all_edges = []
    # 为了测试速度，只取前几个 chunk（课设大批量跑请按需调整）
    for chunk in chunks[:3]: 
        try:
            response = chain.invoke({"input": chunk}).content
            # 清理 Markdown 标记
            response = response.replace('```json', '').replace('```', '').strip()
            result = json.loads(response)
            for item in result:
                if 'node_1' in item and 'node_2' in item and 'edge' in item:
                    all_edges.append([item['node_1'], item['node_2'], item['edge']])
        except Exception as e:
            pass # 忽略解析错误的 chunk

    # 3. 保存 CSV
    if all_edges:
        with open(dest_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['node_1', 'node_2', 'edge'])
            writer.writerows(all_edges)
        return True
    return False

def main():
    if not os.path.exists(pdf_dir):
        print(f"PDF directory not found: {pdf_dir}")
        return
    
    count_2023 = 0
    count_2024 = 0
    count_2025 = 0
    max_samples = 200

    for root, dirs, files in os.walk(pdf_dir):
        for pdf_name in files:
            if not pdf_name.endswith('.pdf'):
                continue
                
            pdf_path = os.path.join(root, pdf_name)
            
            # 简单的年份判断：如果父文件夹名是年份，或者文件名包含年份
            if '2023' in root or '23' in pdf_name:
                if count_2023 < max_samples:
                    if process_pdf(pdf_path, '2023'):
                        count_2023 += 1
                        print(f"Processed 2023: {pdf_name} ({count_2023}/{max_samples})")
                        
            elif '2024' in root or '24' in pdf_name:
                if count_2024 < max_samples:
                    if process_pdf(pdf_path, '2024'):
                        count_2024 += 1
                        print(f"Processed 2024: {pdf_name} ({count_2024}/{max_samples})")
                        
            elif '2025' in root or '25' in pdf_name:
                if count_2025 < max_samples:
                    if process_pdf(pdf_path, '2025'):
                        count_2025 += 1
                        print(f"Processed 2025: {pdf_name} ({count_2025}/{max_samples})")
                        
            if count_2023 >= max_samples and count_2024 >= max_samples and count_2025 >= max_samples:
                return

if __name__ == "__main__":
    if DEEPSEEK_API_KEY == "your_deepseek_api_key_here":
        print("请在代码中填写 DEEPSEEK_API_KEY 后再运行！")
    else:
        main()
