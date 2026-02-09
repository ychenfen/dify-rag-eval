#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量对比评测脚本
支持多知识库、多TopK、是否重排等多维度对比
"""

import os
import json
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from rag_evaluator import DifyRetriever, RAGEvaluator, print_metrics, EvaluationMetrics

load_dotenv()


def run_batch_evaluation(config: dict) -> dict:
    """
    运行批量评测

    Args:
        config: 评测配置，包含：
            - evaluation_set_path: 评测集路径
            - datasets: 知识库配置列表
            - top_k_list: TopK值列表
            - use_rerank_options: 是否使用重排选项列表
            - rerank_provider: 重排 provider 名称（如 local/siliconflow）
            - output_dir: 输出目录

    Returns:
        所有评测结果
    """
    retriever = DifyRetriever()
    evaluator = RAGEvaluator(retriever)

    # 加载评测集
    eval_set = evaluator.load_evaluation_set(config['evaluation_set_path'])

    all_results = []

    # 遍历所有配置组合
    for dataset_config in config['datasets']:
        dataset_id = dataset_config['id']
        dataset_name = dataset_config['name']

        for top_k in config['top_k_list']:
            for use_rerank in config['use_rerank_options']:
                config_name = f"{dataset_name}_top{top_k}"
                if use_rerank:
                    rerank_provider = config.get('rerank_provider', os.getenv('RERANK_PROVIDER_NAME', 'local'))
                    config_name += f"_rerank_{rerank_provider}"
                else:
                    rerank_provider = config.get('rerank_provider', os.getenv('RERANK_PROVIDER_NAME', 'local'))

                print(f"\n{'='*60}")
                print(f"正在评测: {config_name}")
                print(f"{'='*60}")

                # 执行评测
                metrics = evaluator.evaluate(
                    dataset_id=dataset_id,
                    evaluation_set=eval_set,
                    top_k=top_k,
                    use_rerank=use_rerank,
                    rerank_model=config.get('rerank_model', os.getenv('RERANK_MODEL_NAME', 'bge-reranker-base')),
                    rerank_provider=rerank_provider,
                    gold_match=config.get('gold_match', os.getenv('GOLD_MATCH_MODE', 'auto'))
                )

                # 打印结果
                print_metrics(metrics, config_name)

                # 保存单次结果
                evaluator.save_results(metrics, config['output_dir'], config_name)

                # 收集结果
                all_results.append({
                    'config_name': config_name,
                    'dataset_name': dataset_name,
                    'dataset_id': dataset_id,
                    'top_k': top_k,
                    'use_rerank': use_rerank,
                    'total_queries': metrics.total_queries,
                    'hit_count': metrics.hit_count,
                    'recall': metrics.recall_at_k,
                    'precision': metrics.precision_at_k,
                    'f1': metrics.f1_at_k,
                    'mrr': metrics.mrr,
                    'avg_latency_ms': metrics.avg_latency_ms
                })

    # 保存汇总结果
    summary_df = pd.DataFrame(all_results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = os.path.join(config['output_dir'], f"summary_{timestamp}.xlsx")
    summary_df.to_excel(summary_file, index=False, engine='openpyxl')
    print(f"\n汇总结果已保存: {summary_file}")

    # 保存JSON格式
    json_file = os.path.join(config['output_dir'], f"summary_{timestamp}.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"JSON结果已保存: {json_file}")

    return all_results


if __name__ == '__main__':
    # 评测配置
    config = {
        # 评测集路径
        'evaluation_set_path': 'evaluation_set.xlsx',

        # 要对比的知识库（不同分块策略）
        'datasets': [
            {
                'id': os.getenv('DATASET_ID_GENERAL', 'your_general_dataset_id'),
                'name': 'general'  # 通用分块
            },
            {
                'id': os.getenv('DATASET_ID_PARENT_CHILD', 'your_parent_child_dataset_id'),
                'name': 'parent_child'  # 父子分块
            },
            {
                'id': os.getenv('DATASET_ID_QA', 'your_qa_dataset_id'),
                'name': 'qa'  # QA分块
            },
        ],

        # TopK值列表
        'top_k_list': [3, 5, 10],

        # 是否使用重排选项
        'use_rerank_options': [False, True],

        # gold 匹配方式（跨知识库对比建议用 doc_name）
        'gold_match': os.getenv('GOLD_MATCH_MODE', 'auto'),

        # 重排 provider
        'rerank_provider': os.getenv('RERANK_PROVIDER_NAME', 'local'),

        # 重排模型
        'rerank_model': os.getenv('RERANK_MODEL_NAME', 'bge-reranker-base'),

        # 输出目录
        'output_dir': './results'
    }

    # 运行批量评测
    results = run_batch_evaluation(config)

    # 打印最终汇总
    print("\n" + "="*80)
    print("评测完成！最终汇总：")
    print("="*80)

    df = pd.DataFrame(results)
    print(df.to_string(index=False))
