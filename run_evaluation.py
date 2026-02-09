#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键运行完整评测流程
"""

import os
import sys
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


def main():
    print("="*60)
    print("RAG 知识库检索评测 - 完整流程")
    print("="*60)

    # 检查必要文件
    if not os.path.exists('.env'):
        print("错误: 未找到 .env 配置文件")
        print("请创建 .env 文件并配置 DIFY_API_KEY 等参数")
        sys.exit(1)

    eval_set_file = 'evaluation_set.xlsx'
    if not os.path.exists(eval_set_file):
        print(f"未找到评测集文件: {eval_set_file}")
        print("正在创建评测集模板...")
        from build_evaluation_set import create_empty_template
        create_empty_template(eval_set_file)
        print(f"请编辑 {eval_set_file} 填入评测数据后重新运行")
        sys.exit(0)

    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n输出目录: {output_dir}")

    # 配置评测参数
    config = {
        'evaluation_set_path': eval_set_file,
        'datasets': [
            {'id': os.getenv('DATASET_ID_GENERAL'), 'name': 'general'},
            {'id': os.getenv('DATASET_ID_PARENT_CHILD'), 'name': 'parent_child'},
            {'id': os.getenv('DATASET_ID_QA'), 'name': 'qa'},
        ],
        'top_k_list': [3, 5, 10],
        'use_rerank_options': [False, True],
        'gold_match': os.getenv('GOLD_MATCH_MODE', 'auto'),
        'rerank_provider': os.getenv('RERANK_PROVIDER_NAME', 'local'),
        'rerank_model': os.getenv('RERANK_MODEL_NAME', 'bge-reranker-base'),
        'output_dir': output_dir
    }

    # 过滤掉未配置的知识库
    config['datasets'] = [d for d in config['datasets'] if d['id']]

    if not config['datasets']:
        print("错误: 未配置任何知识库ID")
        print("请在 .env 文件中配置 DATASET_ID_GENERAL 等参数")
        sys.exit(1)

    print(f"\n将评测以下知识库:")
    for d in config['datasets']:
        print(f"  - {d['name']}: {d['id']}")

    # 运行批量评测
    print("\n" + "="*60)
    print("开始批量评测...")
    print("="*60)

    from batch_evaluation import run_batch_evaluation
    results = run_batch_evaluation(config)

    # 找到生成的JSON文件
    import glob
    json_files = glob.glob(os.path.join(output_dir, 'summary_*.json'))
    if not json_files:
        print("错误: 未找到汇总JSON文件")
        sys.exit(1)

    json_file = sorted(json_files)[-1]  # 取最新的

    # 生成可视化图表
    print("\n" + "="*60)
    print("生成可视化图表...")
    print("="*60)

    from visualization import main as vis_main
    vis_main(json_file, output_dir)

    print("\n" + "="*60)
    print("评测完成！")
    print("="*60)
    print(f"\n请查看输出目录: {output_dir}")
    print("包含以下文件:")
    for f in os.listdir(output_dir):
        print(f"  - {f}")


if __name__ == '__main__':
    main()
