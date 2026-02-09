#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG 知识库检索评测工具
支持 Dify 知识库的召回率、精确率、F1、MRR 等指标评测
"""

import os
import json
import time
import re
import requests
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv
from tqdm import tqdm
from datetime import datetime

# 加载环境变量
load_dotenv()


@dataclass
class RetrievalResult:
    """单次检索结果"""
    query: str
    retrieved_docs: List[Dict]  # 检索到的文档列表
    gold_doc_ids: List[str]     # 标准答案文档ID列表
    is_hit: bool = False        # 是否命中
    hit_rank: int = -1          # 首次命中的排名（-1表示未命中）
    latency_ms: float = 0       # 检索耗时（毫秒）


@dataclass
class EvaluationMetrics:
    """评测指标"""
    total_queries: int = 0
    hit_count: int = 0
    recall_at_k: float = 0.0
    precision_at_k: float = 0.0
    f1_at_k: float = 0.0
    mrr: float = 0.0
    avg_latency_ms: float = 0.0
    hit_distribution: Dict[int, int] = field(default_factory=dict)  # 各排名命中数分布


class DifyRetriever:
    """Dify 知识库检索器"""

    def __init__(self, api_base: str = None, api_key: str = None):
        """
        初始化检索器

        Args:
            api_base: API基础URL
            api_key: API密钥
        """
        self.api_base = api_base or os.getenv('DIFY_API_BASE', 'https://api.dify.ai/v1')
        self.api_key = api_key or os.getenv('DIFY_API_KEY')

        if not self.api_key:
            raise ValueError("未配置 DIFY_API_KEY，请在 .env 文件中设置")

        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

        # 缓存：避免重复拉取同一知识库的文档列表
        self._documents_cache: Dict[str, List[Dict]] = {}
        self._doc_name_to_ids_cache: Dict[str, Dict[str, List[str]]] = {}

    def list_documents(self, dataset_id: str) -> List[Dict]:
        """
        获取知识库中的所有文档列表（带分页）

        Args:
            dataset_id: 知识库ID

        Returns:
            文档列表（原始API字段）
        """
        if dataset_id in self._documents_cache:
            return self._documents_cache[dataset_id]

        url = f"{self.api_base}/datasets/{dataset_id}/documents"
        all_documents: List[Dict] = []
        page = 1

        while True:
            params = {'page': page, 'limit': 20}
            response = requests.get(url, headers=self.headers, params=params, timeout=30)
            if response.status_code != 200:
                print(f"获取文档列表失败: {response.status_code} - {response.text}")
                break

            data = response.json()
            documents = data.get('data', [])
            if not documents:
                break

            all_documents.extend(documents)
            if not data.get('has_more', False):
                break
            page += 1

        self._documents_cache[dataset_id] = all_documents
        return all_documents

    def get_doc_name_to_ids(self, dataset_id: str) -> Dict[str, List[str]]:
        """
        构建/获取文档名 -> 文档ID列表 的映射。

        注意：同名文档可能存在，因此 value 是 List[str]。
        """
        if dataset_id in self._doc_name_to_ids_cache:
            return self._doc_name_to_ids_cache[dataset_id]

        mapping: Dict[str, List[str]] = {}
        for doc in self.list_documents(dataset_id):
            doc_id = doc.get('id')
            name = (doc.get('name') or '').strip()
            if not doc_id or not name:
                continue
            mapping.setdefault(name, []).append(doc_id)

        self._doc_name_to_ids_cache[dataset_id] = mapping
        return mapping

    def retrieve(self, dataset_id: str, query: str, top_k: int = 5,
                 score_threshold: float = 0.0) -> Tuple[List[Dict], float]:
        """
        执行检索

        Args:
            dataset_id: 知识库ID
            query: 查询问题
            top_k: 返回结果数量
            score_threshold: 分数阈值

        Returns:
            (检索结果列表, 耗时毫秒)
        """
        url = f"{self.api_base}/datasets/{dataset_id}/retrieve"

        payload = {
            "query": query,
            "retrieval_model": {
                "search_method": "semantic_search",  # 语义检索
                "reranking_enable": False,  # 是否启用重排
                "top_k": top_k,
                "score_threshold_enabled": score_threshold > 0,
                "score_threshold": score_threshold
            }
        }

        start_time = time.time()

        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=30)
            latency_ms = (time.time() - start_time) * 1000

            if response.status_code != 200:
                print(f"检索失败: {response.status_code} - {response.text}")
                return [], latency_ms

            data = response.json()
            records = data.get('records', [])

            # 标准化结果格式
            results = []
            for i, record in enumerate(records):
                results.append({
                    'rank': i + 1,
                    'document_id': record.get('segment', {}).get('document_id', ''),
                    'segment_id': record.get('segment', {}).get('id', ''),
                    'content': record.get('segment', {}).get('content', ''),
                    'score': record.get('score', 0),
                    'document_name': record.get('segment', {}).get('document', {}).get('name', ''),
                })

            return results, latency_ms

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            print(f"检索异常: {str(e)}")
            return [], latency_ms

    def retrieve_with_rerank(
        self,
        dataset_id: str,
        query: str,
        top_k: int = 5,
        rerank_model: str = "bge-reranker-base",
        rerank_provider: str = "local",
    ) -> Tuple[List[Dict], float]:
        """
        执行带重排的检索

        Args:
            dataset_id: 知识库ID
            query: 查询问题
            top_k: 返回结果数量
            rerank_model: 重排模型名称
            rerank_provider: 重排模型提供方名称（需与 Dify 系统模型设置中的 provider 名一致）

        Returns:
            (检索结果列表, 耗时毫秒)
        """
        url = f"{self.api_base}/datasets/{dataset_id}/retrieve"

        payload = {
            "query": query,
            "retrieval_model": {
                "search_method": "semantic_search",
                "reranking_enable": True,
                "reranking_model": {
                    "reranking_provider_name": rerank_provider,
                    "reranking_model_name": rerank_model
                },
                "top_k": top_k,
                "score_threshold_enabled": False
            }
        }

        start_time = time.time()

        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=60)
            latency_ms = (time.time() - start_time) * 1000

            if response.status_code != 200:
                print(f"检索失败: {response.status_code} - {response.text}")
                return [], latency_ms

            data = response.json()
            records = data.get('records', [])

            results = []
            for i, record in enumerate(records):
                results.append({
                    'rank': i + 1,
                    'document_id': record.get('segment', {}).get('document_id', ''),
                    'segment_id': record.get('segment', {}).get('id', ''),
                    'content': record.get('segment', {}).get('content', ''),
                    'score': record.get('score', 0),
                    'document_name': record.get('segment', {}).get('document', {}).get('name', ''),
                })

            return results, latency_ms

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            print(f"检索异常: {str(e)}")
            return [], latency_ms


class RAGEvaluator:
    """RAG 检索评测器"""

    def __init__(self, retriever: DifyRetriever):
        """
        初始化评测器

        Args:
            retriever: 检索器实例
        """
        self.retriever = retriever
        self.results: List[RetrievalResult] = []

    def load_evaluation_set(self, file_path: str) -> pd.DataFrame:
        """
        加载评测集

        Args:
            file_path: 评测集文件路径（支持xlsx、csv）

        Returns:
            评测集DataFrame
        """
        if file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path, engine='openpyxl')
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {file_path}")

        # 验证必要列
        if 'query' not in df.columns:
            raise ValueError("评测集缺少必要列: query")
        if ('gold_doc_id' not in df.columns) and ('gold_doc_name' not in df.columns):
            raise ValueError("评测集至少需要一列: gold_doc_id 或 gold_doc_name")

        print(f"加载评测集: {len(df)} 条记录")
        return df

    def evaluate(self, dataset_id: str, evaluation_set: pd.DataFrame,
                 top_k: int = 5, use_rerank: bool = False,
                 rerank_model: str = None,
                 rerank_provider: str = "local",
                 gold_match: str = "auto") -> EvaluationMetrics:
        """
        执行评测

        Args:
            dataset_id: 知识库ID
            evaluation_set: 评测集DataFrame
            top_k: TopK值
            use_rerank: 是否使用重排
            rerank_model: 重排模型（use_rerank=True时需要）
            rerank_provider: 重排 provider 名称（use_rerank=True时可用）
            gold_match: gold 匹配方式
                - auto: 优先使用 gold_doc_id；如果为空则使用 gold_doc_name
                - doc_id: 仅使用 gold_doc_id（适用于单知识库评测）
                - doc_name: 仅使用 gold_doc_name（适用于多知识库对比，依赖文档名一致）

        Returns:
            评测指标
        """
        if gold_match not in {"auto", "doc_id", "doc_name"}:
            raise ValueError("gold_match 必须为 auto/doc_id/doc_name")

        eval_df = evaluation_set.copy()

        # 如果存在 is_valid 列，且包含至少一个 Y，则默认只评测 Y 的行
        if 'is_valid' in eval_df.columns:
            normalized = eval_df['is_valid'].astype(str).str.strip().str.lower()
            valid_mask = normalized.isin({'y', 'yes', 'true', '1'})
            if valid_mask.any():
                eval_df = eval_df[valid_mask].copy()
                print(f"检测到 is_valid 列，已过滤为 {len(eval_df)} 条 (is_valid=Y)")
            else:
                print("检测到 is_valid 列，但未发现任何 Y；将按全量数据评测（如需过滤请在表中标注 Y）。")

        # 丢弃空 query
        eval_df['query'] = eval_df['query'].astype(str).str.strip()
        eval_df = eval_df[eval_df['query'] != ""].copy()

        # 预加载 doc_name -> ids 映射（仅在 doc_name 模式或 auto 可能用到时）
        doc_name_to_ids: Optional[Dict[str, List[str]]] = None
        if gold_match in {"doc_name", "auto"} and 'gold_doc_name' in eval_df.columns:
            doc_name_to_ids = self.retriever.get_doc_name_to_ids(dataset_id)

        self.results = []
        hit_count = 0
        total_rr = 0  # 用于计算MRR
        total_latency = 0
        hit_distribution = {i: 0 for i in range(1, top_k + 1)}
        missing_gold_count = 0

        print(f"\n开始评测...")
        print(f"知识库ID: {dataset_id}")
        print(f"TopK: {top_k}")
        print(f"使用重排: {use_rerank}")
        if use_rerank:
            print(f"重排 provider: {rerank_provider}")
            print(f"重排模型: {rerank_model}")
        print(f"Gold 匹配: {gold_match}")
        print(f"评测集大小: {len(eval_df)}")
        print("-" * 50)

        for idx, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc="评测进度"):
            query = str(row['query']).strip()

            # 处理 gold（可能是单个或多个，用逗号分隔）
            gold_doc_ids: List[str] = []
            gold_doc_names: List[str] = []

            def _split_csv(value: object) -> List[str]:
                if value is None:
                    return []
                s = str(value).strip()
                if not s or s.lower() == 'nan':
                    return []
                return [x.strip() for x in s.split(',') if x.strip()]

            if gold_match in {"auto", "doc_id"} and 'gold_doc_id' in eval_df.columns:
                gold_doc_ids = _split_csv(row.get('gold_doc_id'))

            if (gold_match == "doc_name") or (gold_match == "auto" and not gold_doc_ids):
                if 'gold_doc_name' not in eval_df.columns:
                    raise ValueError("gold_match=doc_name 需要评测集包含 gold_doc_name 列")
                gold_doc_names = _split_csv(row.get('gold_doc_name'))
                if doc_name_to_ids is None:
                    doc_name_to_ids = self.retriever.get_doc_name_to_ids(dataset_id)
                for name in gold_doc_names:
                    mapped = doc_name_to_ids.get(name, [])
                    gold_doc_ids.extend(mapped)

            gold_doc_ids = [x for x in gold_doc_ids if x]  # 清理空值
            if not gold_doc_ids:
                missing_gold_count += 1

            # 执行检索
            if use_rerank and rerank_model:
                retrieved_docs, latency = self.retriever.retrieve_with_rerank(
                    dataset_id, query, top_k, rerank_model, rerank_provider
                )
            else:
                retrieved_docs, latency = self.retriever.retrieve(dataset_id, query, top_k)

            total_latency += latency

            # 判断是否命中
            is_hit = False
            hit_rank = -1

            for doc in retrieved_docs:
                doc_id = doc.get('document_id', '')
                if doc_id in gold_doc_ids:
                    is_hit = True
                    hit_rank = doc['rank']
                    break

            if is_hit:
                hit_count += 1
                total_rr += 1.0 / hit_rank
                hit_distribution[hit_rank] = hit_distribution.get(hit_rank, 0) + 1

            # 记录结果
            result = RetrievalResult(
                query=query,
                retrieved_docs=retrieved_docs,
                gold_doc_ids=gold_doc_ids,
                is_hit=is_hit,
                hit_rank=hit_rank,
                latency_ms=latency
            )
            self.results.append(result)

            # 避免请求过快
            time.sleep(0.1)

        # 计算指标
        total_queries = len(eval_df)
        recall = hit_count / total_queries if total_queries > 0 else 0
        precision = hit_count / (top_k * total_queries) if total_queries > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        mrr = total_rr / total_queries if total_queries > 0 else 0
        avg_latency = total_latency / total_queries if total_queries > 0 else 0

        metrics = EvaluationMetrics(
            total_queries=total_queries,
            hit_count=hit_count,
            recall_at_k=recall,
            precision_at_k=precision,
            f1_at_k=f1,
            mrr=mrr,
            avg_latency_ms=avg_latency,
            hit_distribution=hit_distribution
        )

        if missing_gold_count > 0:
            print(f"\n警告: 有 {missing_gold_count}/{total_queries} 条评测数据无法解析到 gold_doc_ids（将视为未命中）。")
            if gold_match == "doc_name":
                print("  请确认 evaluation_set.xlsx 中 gold_doc_name 与该知识库中的文档名完全一致。")
            elif gold_match == "auto":
                print("  auto 模式下会优先用 gold_doc_id；若跨知识库对比请使用 --gold-match doc_name。")

        return metrics

    def get_detailed_results(self) -> pd.DataFrame:
        """
        获取详细结果

        Returns:
            详细结果DataFrame
        """
        data = []
        for r in self.results:
            data.append({
                'query': r.query,
                'gold_doc_ids': ','.join(r.gold_doc_ids),
                'is_hit': r.is_hit,
                'hit_rank': r.hit_rank if r.hit_rank > 0 else 'N/A',
                'latency_ms': round(r.latency_ms, 2),
                'top1_doc_id': r.retrieved_docs[0]['document_id'] if r.retrieved_docs else '',
                'top1_score': round(r.retrieved_docs[0]['score'], 4) if r.retrieved_docs else '',
                'top1_content': r.retrieved_docs[0]['content'][:100] if r.retrieved_docs else '',
            })
        return pd.DataFrame(data)

    def save_results(self, metrics: EvaluationMetrics, output_dir: str,
                     config_name: str = "default"):
        """
        保存评测结果

        Args:
            metrics: 评测指标
            output_dir: 输出目录
            config_name: 配置名称（用于区分不同实验）
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存指标摘要
        summary = {
            'config_name': config_name,
            'timestamp': timestamp,
            'total_queries': metrics.total_queries,
            'hit_count': metrics.hit_count,
            'recall_at_k': round(metrics.recall_at_k, 4),
            'precision_at_k': round(metrics.precision_at_k, 4),
            'f1_at_k': round(metrics.f1_at_k, 4),
            'mrr': round(metrics.mrr, 4),
            'avg_latency_ms': round(metrics.avg_latency_ms, 2),
            'hit_distribution': metrics.hit_distribution
        }

        summary_file = os.path.join(output_dir, f"metrics_{config_name}_{timestamp}.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"指标摘要已保存: {summary_file}")

        # 保存详细结果
        detailed_df = self.get_detailed_results()
        detailed_file = os.path.join(output_dir, f"detailed_{config_name}_{timestamp}.xlsx")
        detailed_df.to_excel(detailed_file, index=False, engine='openpyxl')
        print(f"详细结果已保存: {detailed_file}")

        return summary_file, detailed_file


def print_metrics(metrics: EvaluationMetrics, config_name: str = ""):
    """
    打印评测指标

    Args:
        metrics: 评测指标
        config_name: 配置名称
    """
    print("\n" + "=" * 50)
    if config_name:
        print(f"配置: {config_name}")
    print("=" * 50)
    print(f"总查询数:      {metrics.total_queries}")
    print(f"命中数:        {metrics.hit_count}")
    print(f"Recall@K:      {metrics.recall_at_k:.4f} ({metrics.recall_at_k*100:.2f}%)")
    print(f"Precision@K:   {metrics.precision_at_k:.4f} ({metrics.precision_at_k*100:.2f}%)")
    print(f"F1@K:          {metrics.f1_at_k:.4f}")
    print(f"MRR:           {metrics.mrr:.4f}")
    print(f"平均延迟:      {metrics.avg_latency_ms:.2f} ms")
    print("-" * 50)
    print("命中排名分布:")
    for rank, count in sorted(metrics.hit_distribution.items()):
        if count > 0:
            print(f"  Rank {rank}: {count} ({count/metrics.total_queries*100:.1f}%)")
    print("=" * 50)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='RAG 知识库检索评测工具')
    parser.add_argument('--dataset-id', type=str, required=True, help='知识库ID')
    parser.add_argument('--eval-set', type=str, required=True, help='评测集文件路径')
    parser.add_argument('--top-k', type=int, default=5, help='TopK值')
    parser.add_argument('--use-rerank', action='store_true', help='是否使用重排')
    parser.add_argument('--rerank-provider', type=str,
                        default=os.getenv('RERANK_PROVIDER_NAME', 'local'),
                        help='重排provider（需与Dify系统模型设置一致），如 local/siliconflow')
    parser.add_argument('--rerank-model', type=str, default='bge-reranker-base', help='重排模型')
    parser.add_argument('--gold-match', type=str, choices=['auto', 'doc_id', 'doc_name'],
                        default=os.getenv('GOLD_MATCH_MODE', 'auto'),
                        help='gold 匹配方式：auto/doc_id/doc_name（多知识库对比建议用 doc_name）')
    parser.add_argument('--output-dir', type=str, default='./results', help='输出目录')
    parser.add_argument('--config-name', type=str, default='default', help='配置名称')

    args = parser.parse_args()

    # 初始化
    retriever = DifyRetriever()
    evaluator = RAGEvaluator(retriever)

    # 加载评测集
    eval_set = evaluator.load_evaluation_set(args.eval_set)

    # 执行评测
    metrics = evaluator.evaluate(
        dataset_id=args.dataset_id,
        evaluation_set=eval_set,
        top_k=args.top_k,
        use_rerank=args.use_rerank,
        rerank_model=args.rerank_model if args.use_rerank else None,
        rerank_provider=args.rerank_provider,
        gold_match=args.gold_match
    )

    # 打印结果
    print_metrics(metrics, args.config_name)

    # 保存结果
    evaluator.save_results(metrics, args.output_dir, args.config_name)
