# RAG-Travel-Insurance-Chatbot

A prototype RAG-based chatbot for travel inconvenience insurance policies, capable of retrieving policy clauses and generating accurate, source-referenced answers. Includes data chunking, metadata design, retrieval methods, prompt strategies, and sample QA.

---

##  Features

- **Preloaded Engine (Singleton Mode)**: 單例初始化檢索引擎，避免每次查詢都重建，節省時間並提升性能。
- **Batch Query**: 可一次處理多個問題，從檔案讀取並生成對應答案。
- **Interactive Mode**: 支援連續對話，保留上下文歷史。
- **Index Building**: 自動生成條文 chunks 與意圖索引，並整合意圖與條款。
- **Statistics**: 查看索引檔案大小、條文結構與意圖分布。
- **Similarity Scores**: 每個條文來源顯示語義相似度分數。

---

##  Environment Configuration

在專案根目錄下建立 `.env` 檔案，填入你的 OpenAI API Key：

```
OPENAI_API_KEY=your_api_key_here
```
---
## Installation
```
git clone <repo_url>
cd <project_dir>

# 建議使用虛擬環境
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# 安裝依賴
pip install -r requirements.txt
```
## Usage

### 1. Build Index

生成條文與意圖索引，必須先執行此步驟：
```
python main.py build
python main.py build --regenerate   # 強制重新生成索引
```

```
輸出檔案：

chunks_structured.json
intents.json
chunks_structured_with_intents.json
```
### 2. Single Query

查詢單個問題並返回答案與來源：
```
python main.py query "什麼情況下可以申請旅遊延誤賠償？"
python main.py query "..." --top-k-intents 5 --top-k-clauses 3 --details
python main.py query "..." --no-sources
```
```
參數說明：

--top-k-intents: 檢索的意圖數量（預設: 5）
--top-k-clauses: 返回條文數量（預設: 3）
--details: 顯示意圖分數等詳細訊息
--no-sources: 不顯示來源條文
```
### 3. Interactive Mode

支援連續對話，保留上下文歷史：
```
python main.py interactive
```

```
操作提示：

輸入問題並按 Enter
clear: 清除對話歷史
quit / exit / q: 退出互動模式
```

### 4. Batch Query
從檔案讀取多個問題並生成答案，可選擇輸出 JSON：
```
python main.py batch questions.txt
python main.py batch questions.txt -o output.json
```
```
questions.txt: 每行一個問題
output.json: 選擇性輸出結果檔案
```
### 5. View Statistics

檢視索引統計資訊，包括條文結構、意圖數量、意圖分類及索引大小：
```
python main.py stats
```

# Project Structure
```
.
├─ main.py                    # 主程序
├─ config.py                  # 設定索引路徑等
├─ core/
│   ├─ chunk_generator.py     # 條文拆分與儲存
│   ├─ intent_generator.py    # 意圖生成
│   ├─ retrieval_engine.py    # 檢索引擎
│   └─ answer_generator.py    # 答案生成器
├─ data/                      # 原始 PDF / 條款檔案
└─ index/                     # 生成索引
```

```
Notes

API Key: 確保 .env 裡填入正確的 OPENAI_API_KEY，否則無法呼叫 GPT 模型生成答案。

索引更新: 每次新增或修改保險條款 PDF，都建議重新執行 python main.py build --regenerate。

Top-K 設定: 可以調整 --top-k-intents 與 --top-k-clauses 來控制檢索精度與答案豐富度。

虛擬環境: 建議在虛擬環境中運行，避免系統 Python 套件衝突。
```