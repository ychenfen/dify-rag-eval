# 常见问题（FAQ）

本 FAQ 用于回答使用 Dify 知识库评测脚本过程中最常见的疑问，尤其是“多知识库对比 + 重排 + 评测集构建”这几个点。

## Q1：`extract_candidate_questions` 规则太少，生成的候选问题很多是“返回搜狐/责任编辑”这类噪音，怎么办？

原因通常是：知识库文档本身包含新闻站点水印、页眉页脚、免责声明、导航、广告等文本，分块后这些噪音会被当成“标题/短句”，从而被规则命中。

本仓库已在 `build_evaluation_set.py` 做了两类改进：
- **噪音行过滤**：对“返回搜狐/查看更多/责任编辑/免责声明/导航/广告”等常见模式直接过滤，不参与问题生成。
- **更稳的候选生成**：优先抽取 `问：/Q:`、带问号的句子、疑问词开头的句子；对疑似标题行再用更严格的模板生成问题（避免把正文残句硬转成问题）。

你仍然可以进一步增强（建议按优先级）：
1. 在上传知识库前清理文档：去掉重复页眉页脚、免责声明、导航、版权、广告等。
2. 按你业务领域扩展噪音过滤关键词（在 `build_evaluation_set.py` 的 `noise_line_re` 正则里追加）。
3. 若你允许引入 LLM，可把“候选问题生成”改成：对每段调用一个小模型生成 1-2 个高质量问题（但这会增加成本与耗时，也引入模型偏差）。

## Q2：我有 3 个知识库（通用/父子/QA），构建评测集时基于哪个知识库？会不会导致另外 2 个评测不准？

关键点：**跨知识库对比时不要用 `gold_doc_id` 做 gold**。

原因：Dify 的 `document_id` 是“知识库内唯一”，同一份文件上传到不同知识库，`document_id` 往往不同。  
所以如果评测集用 `gold_doc_id`，会出现：
- 在 A 知识库评测正常
- 切换到 B/C 知识库时，gold 的 doc_id 根本不存在，导致指标被低估

正确做法：
- 评测集填 `gold_doc_name`（文档名），并确保 3 个知识库里文档名一致
- 评测时使用 `--gold-match doc_name`（脚本会把 `gold_doc_name` 映射到当前知识库的 `document_id`）

示例：
```bash
python3 rag_evaluator.py --dataset-id <DATASET_ID> --eval-set evaluation_set.xlsx --top-k 5 --gold-match doc_name
```

结论：
- 构建评测集可以基于任意一个知识库（甚至只基于其中一个也可以）
- 但最终评测集应以 `gold_doc_name` 为准，这样才能对比通用/父子/QA 三种分块策略

## Q3：能否提供一个成功跑通的示例数据集/结果作参考？

仓库里提供了一个最小可复现示例（用于“跑通流程”而非“得出固定指标”）：
- `examples/docs/`：可上传到 Dify 的 demo 文档
- `examples/evaluation_set_example.xlsx`：示例评测集（用 `gold_doc_name`）
- `examples/README.md`：如何使用示例

说明：
- 指标结果与 Dify 的 embedding/reranker/索引状态相关，不同环境数值不完全一致是正常的。

## Q4：Dify 知识库用 QA 分块时一直卡在“嵌入处理中/索引中”两个多小时，资源看起来也正常，怎么排查？

这类问题通常不是脚本导致，而是知识库侧的索引流水线卡住。常见原因：
- embedding provider 额度/鉴权问题（401/403/429）导致不断重试
- worker/队列堆积或某个任务异常退出
- 文档过大或包含异常格式导致解析/抽取阶段卡住
- QA 分块通常涉及额外的 Q/A 抽取步骤，比纯文本分块更耗时

排查建议（从快到慢）：
1. 先上传 1-2 个小文件，验证 QA 分块是否能在可接受时间内完成。
2. 在 Dify 后台查看该文档的处理阶段与报错日志（重点看 embedding 调用是否报 401/429/5xx）。
3. 若为自部署，检查 worker/队列是否积压、向量库写入是否异常慢、相关容器是否重启/崩溃。
4. 如果是第三方 embedding/reranker 服务，检查服务侧日志与限流策略。

## Q5：`run_evaluation.py` 批量跑配置了第三方 reranker，但报“找不到重排模型”；单跑 `rag_evaluator.py` 传 `siliconflow` 却没问题，为什么？

根因是：之前脚本把 `reranking_provider_name` 固定写死为 `local`，导致批量跑时始终按 `local` 去找模型。

本仓库已修复：
- `rag_evaluator.py` 增加 `--rerank-provider`
- `batch_evaluation.py` / `run_evaluation.py` 会从 `.env` 读取并透传：
  - `RERANK_PROVIDER_NAME`
  - `RERANK_MODEL_NAME`

你需要确保：
- `RERANK_PROVIDER_NAME` 与 Dify「系统模型设置」里的 provider 名完全一致（大小写/拼写都要一致）
- `RERANK_MODEL_NAME` 与系统模型设置中的模型名完全一致

示例（硅基流动 + bge reranker）：
```bash
python3 rag_evaluator.py \
  --dataset-id <DATASET_ID> \
  --eval-set evaluation_set.xlsx \
  --top-k 5 \
  --gold-match doc_name \
  --use-rerank \
  --rerank-provider siliconflow \
  --rerank-model BAAI/bge-reranker-v2-m3
```

## Q6：评测集 `evaluation_set.xlsx` 最少需要哪些列？

最少需要：
- `query`
- `gold_doc_id` 或 `gold_doc_name`（二选一）

推荐加上（方便排查与统计）：
- `category`
- `difficulty`

可选（方便人工回看/追溯）：
- `gold_chunk_text`
- `gold_segment_id`
- `is_valid`：如果存在此列且至少一行标 `Y`，评测时会自动只评测 `Y` 行

## Q7：创建评测集之前，上传的文档要不要先预处理（清噪）？

不属于“硬性要求”，但强烈建议做基础清理，因为它会显著提升检索稳定性与评测集质量：
- 去掉重复页眉页脚、免责声明、导航、版权、广告
- 去掉无意义空行、目录、重复段落
- 尽量保留正文结构（标题/小节），对分块与检索更友好

## Q8：Dify 知识库默认是否支持“上下文扩展（expanded/contextual）”？如何实现？

默认的向量/全文检索一般只返回“命中的分段”，不会自动把上下文段（前后段）拼接成 expanded context。

常见实现方式有两种：
1. **父子分块**：让“子 chunk 命中后”天然关联到更大的父 chunk，父 chunk 作为上下文。
2. **应用侧扩展**：检索命中 segment 后，再拉取同文档相邻 segments（上/下文）拼接成最终上下文再喂给 LLM。

评测脚本当前关注的是“检索命中是否包含 gold 文档”，不直接实现 expanded context；如果你要评测 expanded 策略，需要在“检索后拼接上下文”的那层增加评测维度。

