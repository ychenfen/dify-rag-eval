# CRUD-RAG 数据集快速入门教程

## 教程目标

本教程介绍如何使用CRUD-RAG数据集来测试Dify知识库的检索效果。完成本教程后,可以:
- 准备测试文档库
- 创建评测集
- 运行评测脚本
- 分析评测结果

---

## 第一步: 准备测试文档

### 1.1 提取部分文档用于测试

CRUD-RAG包含80,000+篇文档,建议先用1000篇进行测试。

创建测试文档目录:
```bash
cd CRUD_RAG/data
mkdir test_docs
```

提取前1000篇文档:
```bash
# 复制前10个文档文件(每个文件约100篇文档)
cp 80000_docs/documents_dup_part_1_part_1 test_docs/
cp 80000_docs/documents_dup_part_1_part_2 test_docs/
cp 80000_docs/documents_dup_part_1_part_3 test_docs/
cp 80000_docs/documents_dup_part_2_part_1 test_docs/
cp 80000_docs/documents_dup_part_2_part_2 test_docs/
cp 80000_docs/documents_dup_part_2_part_3 test_docs/
cp 80000_docs/documents_dup_part_3_part_1 test_docs/
cp 80000_docs/documents_dup_part_3_part_2 test_docs/
cp 80000_docs/documents_dup_part_3_part_3 test_docs/
cp 80000_docs/documents_dup_part_4_part_1 test_docs/
```

### 1.2 查看文档内容

```bash
# 查看第一个文档文件的内容
head -50 test_docs/documents_dup_part_1_part_1
```

文档格式示例:
```
[ 2023-07-28 16：40 ] ，正文：人民网柏林7月28日电...
2023-08-11 15:10，正文：中新网约翰内斯堡8月10日电...
```

---

## 第二步: 上传文档到Dify

### 2.1 创建知识库

1. 登录Dify控制台
2. 点击左侧菜单「知识库」
3. 点击「创建知识库」按钮
4. 输入知识库名称,例如:
   - `CRUD_测试_通用分块`
   - `CRUD_测试_父子分块`
   - `CRUD_测试_QA分块`

### 2.2 上传文档

1. 进入刚创建的知识库
2. 点击「上传文档」
3. 选择 `test_docs` 目录下的所有文件
4. 选择分块策略:
   - 通用分块: chunk_size=500, overlap=50
   - 父子分块: 使用默认设置
   - QA分块: 使用默认设置
5. 点击「开始处理」

### 2.3 获取知识库ID

1. 进入知识库详情页
2. 查看浏览器地址栏
3. URL格式: `https://xxx.dify.ai/datasets/【这里是知识库ID】/documents`
4. 复制知识库ID,例如: `abc123def456`

---

## 第三步: 创建评测集

### 3.1 方式一: 使用自动生成工具

回到RAG评测项目目录,运行:

```bash
python build_evaluation_set.py \
  --action build \
  --dataset-id abc123def456 \
  --output candidates.xlsx
```

这会自动从知识库中提取候选问题。

### 3.2 方式二: 手工创建

创建一个Excel文件 `evaluation_set.xlsx`,包含以下列:

| id | query | gold_doc_id | gold_chunk_text | category | difficulty |
|----|-------|-------------|-----------------|----------|------------|
| 1  | 中国驻德国大使馆举行招待会是为了庆祝什么? | doc_001 | 中国人民解放军建军96周年 | 政治 | easy |
| 2  | 王文涛在南非表示中南经贸合作前景如何? | doc_002 | 前景光明、大有可为 | 经济 | medium |
| 3  | 华硕无畏联名的是哪个潮流品牌? | doc_003 | BAPE | 科技 | easy |

注意事项:
- `gold_doc_id` 必须是Dify知识库中实际的文档ID
- 建议先创建50-100个问题进行测试

### 3.3 人工审核

打开 `candidates.xlsx`,逐条检查:
1. 问题是否自然、像真实用户会问的
2. `gold_doc_id` 是否正确
3. 标注 `is_valid` 列: Y表示保留, N表示删除
4. 填写 `category` 和 `difficulty`

---

## 第四步: 配置API密钥

### 4.1 获取Dify API Key

1. 登录Dify控制台
2. 点击左下角「设置」
3. 找到「API密钥」
4. 点击「创建新密钥」
5. 复制生成的API Key

### 4.2 创建配置文件

在项目根目录创建 `.env` 文件:

```bash
# Dify API配置
DIFY_API_BASE=https://api.dify.ai/v1
DIFY_API_KEY=你的API_Key

# 知识库ID配置
DATASET_ID_GENERAL=通用分块知识库的ID
DATASET_ID_PARENT_CHILD=父子分块知识库的ID
DATASET_ID_QA=QA分块知识库的ID
```

示例:
```
DIFY_API_BASE=https://api.dify.ai/v1
DIFY_API_KEY=app-abc123def456xyz789
DATASET_ID_GENERAL=ds-general-001
DATASET_ID_PARENT_CHILD=ds-parent-002
DATASET_ID_QA=ds-qa-003
```

---

## 第五步: 运行评测

### 5.1 单次评测

测试单个知识库:

```bash
python rag_evaluator.py \
  --dataset-id ds-general-001 \
  --eval-set evaluation_set.xlsx \
  --top-k 5 \
  --config-name "通用分块_top5" \
  --output-dir ./results
```

参数说明:
- `--dataset-id`: 知识库ID
- `--eval-set`: 评测集文件路径
- `--top-k`: 检索返回的结果数量(建议3、5、10)
- `--config-name`: 本次评测的名称
- `--output-dir`: 结果保存目录

### 5.2 批量对比评测

对比多个知识库和配置:

```bash
python batch_evaluation.py
```

这会自动运行所有配置组合,包括:
- 3种分块策略 × 3种TopK值 × 2种重排选项 = 18次评测

### 5.3 查看评测进度

评测过程中会显示进度条:
```
评测进度: 100%|████████████| 100/100 [02:30<00:00, 1.50s/it]
```

---

## 第六步: 分析结果

### 6.1 查看指标摘要

评测完成后,终端会显示:

```
==================================================
配置: 通用分块_top5
==================================================
总查询数:      100
命中数:        85
Recall@K:      0.8500 (85.00%)
Precision@K:   0.1700 (17.00%)
F1@K:          0.2833
MRR:           0.7234
平均延迟:      245.67 ms
--------------------------------------------------
命中排名分布:
  Rank 1: 65 (65.0%)
  Rank 2: 12 (12.0%)
  Rank 3: 5 (5.0%)
  Rank 4: 2 (2.0%)
  Rank 5: 1 (1.0%)
==================================================
```

### 6.2 查看详细结果

打开 `results` 目录,包含:
- `metrics_xxx.json`: 指标汇总(JSON格式)
- `detailed_xxx.xlsx`: 每个问题的详细结果
- `summary_xxx.xlsx`: 所有配置的对比汇总

### 6.3 生成可视化图表

```bash
python visualization.py \
  --input results/summary_20240205_120000.json \
  --output-dir ./results
```

生成的图表包括:
- `recall_comparison.png`: 召回率对比图
- `metrics_heatmap.png`: 指标热力图
- `comprehensive_comparison.png`: 综合对比图
- `evaluation_report.md`: 文字分析报告

---

## 第七步: 解读结果

### 7.1 关键指标含义

Recall@K (召回率)
- 含义: 在Top K结果中找到正确答案的问题占比
- 目标: 越高越好,建议 > 80%
- 示例: Recall@5 = 85% 表示100个问题中有85个在Top5中找到了答案

Precision@K (精确率)
- 含义: 返回结果中相关文档的占比
- 目标: 越高越好
- 计算: 命中数 / (K × 总问题数)

MRR (平均倒数排名)
- 含义: 正确答案排名的倒数平均值
- 目标: 越高越好,建议 > 0.7
- 示例: 答案在第1位贡献1.0,第2位贡献0.5,第5位贡献0.2

F1 Score
- 含义: 召回率和精确率的调和平均
- 目标: 越高越好
- 用途: 综合评估检索效果

### 7.2 优化建议

如果召回率低 (< 70%):
1. 增大TopK值(从5增到10)
2. 调整chunk_size(尝试300-800)
3. 增加chunk_overlap(尝试10%-20%)
4. 启用重排器(Reranker)

如果MRR低但召回率高:
1. 启用重排器提升排序质量
2. 优化embedding模型
3. 调整检索策略

如果延迟高 (> 500ms):
1. 减小TopK值
2. 优化向量索引
3. 考虑是否真的需要重排器

---

## 第八步: 对比不同配置

### 8.1 分块策略对比

查看 `summary_xxx.xlsx`,对比:
- 通用分块 vs 父子分块 vs QA分块
- 哪种策略的召回率最高
- 哪种策略的延迟最低

### 8.2 TopK值对比

观察规律:
- TopK从3到5到10,召回率如何变化
- 延迟增加了多少
- 最佳的性价比是多少

### 8.3 重排器效果

对比启用/不启用重排器:
- 召回率提升了多少
- MRR提升了多少
- 延迟增加了多少
- 是否值得启用

---

## 常见问题

Q1: 评测集应该准备多少个问题?
A: 初步测试50-100个,正式评测300-500个,持续监控100-200个核心问题

Q2: 如何判断评测结果是否可信?
A: 评测集规模足够大(>100),问题覆盖多个领域,标注准确无误,多次运行结果稳定

Q3: 召回率100%是好事吗?
A: 不一定,可能说明评测集太简单,问题与文档高度重复,需要增加难度更高的问题

Q4: 不同知识库的结果能直接对比吗?
A: 可以,但要确保使用相同的评测集,文档内容相同,只有分块策略不同

Q5: 如何处理评测失败的情况?
A: 检查API Key是否正确,知识库ID是否正确,网络连接是否正常,评测集格式是否正确

---

## 下一步行动

完成本教程后,建议:

1. 小规模验证: 用50个问题快速验证流程
2. 扩大规模: 准备300+个问题进行正式评测
3. 持续优化: 根据结果调整配置
4. 定期监控: 知识库更新后重新评测

---

## 需要帮助

如果遇到问题:
1. 检查 `.env` 配置是否正确
2. 查看 `results` 目录下的错误日志
3. 确认评测集格式符合要求
4. 验证API Key和知识库ID
