#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评测结果可视化
生成对比图表和分析报告
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import matplotlib.font_manager as fm

def _configure_chinese_font():
    """
    Configure a usable Chinese font across environments.

    Matplotlib will silently fall back to fonts that may not contain CJK glyphs,
    producing warnings like: "Glyph xxxx missing from font(s) Arial".
    """
    preferred = [
        # macOS common
        'PingFang SC',
        'Hiragino Sans GB',
        'STHeiti',
        'Songti SC',
        # Windows common
        'Microsoft YaHei',
        'SimHei',
        # Broad Unicode fallback (may or may not exist)
        'Arial Unicode MS',
        # Linux common
        'Noto Sans CJK SC',
        'WenQuanYi Zen Hei',
        # final fallback
        'DejaVu Sans',
    ]

    available = {f.name for f in fm.fontManager.ttflist}
    usable = [name for name in preferred if name in available]

    # Ensure we prefer fonts that actually exist on the machine.
    if usable:
        plt.rcParams['font.sans-serif'] = usable
        plt.rcParams['font.family'] = 'sans-serif'

    # Fix minus sign rendering
    plt.rcParams['axes.unicode_minus'] = False


# 设置中文字体（自动选择本机可用字体）
_configure_chinese_font()

# 设置绘图风格
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 150


def load_summary_data(json_file: str) -> pd.DataFrame:
    """
    加载汇总数据

    Args:
        json_file: JSON格式的汇总文件路径

    Returns:
        DataFrame
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return pd.DataFrame(data)


def plot_recall_comparison(df: pd.DataFrame, output_dir: str):
    """
    绘制召回率对比图

    Args:
        df: 评测结果DataFrame
        output_dir: 输出目录
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 图1: 不同分块策略的召回率对比（按TopK分组）
    ax1 = axes[0]

    # 准备数据：不使用重排的结果
    df_no_rerank = df[df['use_rerank'] == False]

    pivot_data = df_no_rerank.pivot(index='top_k', columns='dataset_name', values='recall')

    x = np.arange(len(pivot_data.index))
    width = 0.25

    colors = ['#2ecc71', '#3498db', '#e74c3c']

    for i, col in enumerate(pivot_data.columns):
        bars = ax1.bar(x + i*width, pivot_data[col], width, label=col, color=colors[i % len(colors)])
        # 添加数值标签
        for bar, val in zip(bars, pivot_data[col]):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.2%}', ha='center', va='bottom', fontsize=9)

    ax1.set_xlabel('Top K', fontsize=12)
    ax1.set_ylabel('Recall@K', fontsize=12)
    ax1.set_title('不同分块策略的召回率对比（无重排）', fontsize=14, fontweight='bold')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels([f'Top{k}' for k in pivot_data.index])
    ax1.legend(title='分块策略')
    ax1.set_ylim(0, 1.1)

    # 图2: 重排前后对比
    ax2 = axes[1]

    # 选择 Top5 的数据进行重排前后对比
    df_top5 = df[df['top_k'] == 5]

    datasets = df_top5['dataset_name'].unique()
    x = np.arange(len(datasets))
    width = 0.35

    recall_no_rerank = df_top5[df_top5['use_rerank'] == False].set_index('dataset_name')['recall']
    recall_with_rerank = df_top5[df_top5['use_rerank'] == True].set_index('dataset_name')['recall']

    bars1 = ax2.bar(x - width/2, [recall_no_rerank.get(d, 0) for d in datasets],
                    width, label='无重排', color='#3498db')
    bars2 = ax2.bar(x + width/2, [recall_with_rerank.get(d, 0) for d in datasets],
                    width, label='有重排', color='#e74c3c')

    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{bar.get_height():.2%}', ha='center', va='bottom', fontsize=9)

    ax2.set_xlabel('分块策略', fontsize=12)
    ax2.set_ylabel('Recall@5', fontsize=12)
    ax2.set_title('重排前后召回率对比（Top5）', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets)
    ax2.legend()
    ax2.set_ylim(0, 1.1)

    plt.tight_layout()

    output_file = os.path.join(output_dir, 'recall_comparison.png')
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    print(f"召回率对比图已保存: {output_file}")


def plot_metrics_heatmap(df: pd.DataFrame, output_dir: str):
    """
    绘制指标热力图

    Args:
        df: 评测结果DataFrame
        output_dir: 输出目录
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 热力图1: Recall
    ax1 = axes[0]
    df_no_rerank = df[df['use_rerank'] == False]
    pivot_recall = df_no_rerank.pivot(index='dataset_name', columns='top_k', values='recall')

    sns.heatmap(pivot_recall, annot=True, fmt='.2%', cmap='YlGnBu', ax=ax1,
                vmin=0, vmax=1, cbar_kws={'label': 'Recall'})
    ax1.set_title('召回率热力图（无重排）', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Top K')
    ax1.set_ylabel('分块策略')

    # 热力图2: MRR
    ax2 = axes[1]
    pivot_mrr = df_no_rerank.pivot(index='dataset_name', columns='top_k', values='mrr')

    sns.heatmap(pivot_mrr, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax2,
                vmin=0, vmax=1, cbar_kws={'label': 'MRR'})
    ax2.set_title('MRR热力图（无重排）', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Top K')
    ax2.set_ylabel('分块策略')

    plt.tight_layout()

    output_file = os.path.join(output_dir, 'metrics_heatmap.png')
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    print(f"指标热力图已保存: {output_file}")


def plot_latency_comparison(df: pd.DataFrame, output_dir: str):
    """
    绘制延迟对比图

    Args:
        df: 评测结果DataFrame
        output_dir: 输出目录
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # 按配置名排序
    df_sorted = df.sort_values(['dataset_name', 'top_k', 'use_rerank'])

    # 创建配置标签
    labels = []
    for _, row in df_sorted.iterrows():
        rerank_str = '+rerank' if row['use_rerank'] else ''
        labels.append(f"{row['dataset_name']}\nTop{row['top_k']}{rerank_str}")

    colors = ['#3498db' if not r else '#e74c3c' for r in df_sorted['use_rerank']]

    bars = ax.barh(range(len(labels)), df_sorted['avg_latency_ms'], color=colors)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('平均延迟 (ms)', fontsize=12)
    ax.set_title('检索延迟对比', fontsize=14, fontweight='bold')

    # 添加数值标签
    for bar, val in zip(bars, df_sorted['avg_latency_ms']):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                f'{val:.0f}ms', ha='left', va='center', fontsize=9)

    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#3498db', label='无重排'),
                      Patch(facecolor='#e74c3c', label='有重排')]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()

    output_file = os.path.join(output_dir, 'latency_comparison.png')
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    print(f"延迟对比图已保存: {output_file}")


def plot_comprehensive_comparison(df: pd.DataFrame, output_dir: str):
    """
    绘制综合对比图（类似用户提供的示例图）

    Args:
        df: 评测结果DataFrame
        output_dir: 输出目录
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 选择不使用重排的数据
    df_no_rerank = df[df['use_rerank'] == False]

    # 图1: 召回率折线图
    ax1 = axes[0, 0]
    for dataset in df_no_rerank['dataset_name'].unique():
        data = df_no_rerank[df_no_rerank['dataset_name'] == dataset]
        ax1.plot(data['top_k'], data['recall'], marker='o', linewidth=2,
                markersize=8, label=dataset)
    ax1.set_xlabel('Top K', fontsize=12)
    ax1.set_ylabel('Recall@K', fontsize=12)
    ax1.set_title('召回率随TopK变化趋势', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)

    # 图2: F1分数对比
    ax2 = axes[0, 1]
    pivot_f1 = df_no_rerank.pivot(index='top_k', columns='dataset_name', values='f1')
    pivot_f1.plot(kind='bar', ax=ax2, color=['#2ecc71', '#3498db', '#e74c3c'])
    ax2.set_xlabel('Top K', fontsize=12)
    ax2.set_ylabel('F1 Score', fontsize=12)
    ax2.set_title('F1分数对比', fontsize=14, fontweight='bold')
    ax2.legend(title='分块策略')
    ax2.set_xticklabels([f'Top{k}' for k in pivot_f1.index], rotation=0)

    # 图3: MRR对比
    ax3 = axes[1, 0]
    pivot_mrr = df_no_rerank.pivot(index='top_k', columns='dataset_name', values='mrr')
    pivot_mrr.plot(kind='bar', ax=ax3, color=['#9b59b6', '#f39c12', '#1abc9c'])
    ax3.set_xlabel('Top K', fontsize=12)
    ax3.set_ylabel('MRR', fontsize=12)
    ax3.set_title('MRR (Mean Reciprocal Rank) 对比', fontsize=14, fontweight='bold')
    ax3.legend(title='分块策略')
    ax3.set_xticklabels([f'Top{k}' for k in pivot_mrr.index], rotation=0)

    # 图4: 综合雷达图（选择Top5数据）
    ax4 = axes[1, 1]

    df_top5 = df_no_rerank[df_no_rerank['top_k'] == 5]

    categories = ['Recall', 'Precision', 'F1', 'MRR']

    # 计算角度
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    ax4 = fig.add_subplot(2, 2, 4, polar=True)

    colors = ['#2ecc71', '#3498db', '#e74c3c']
    for i, (_, row) in enumerate(df_top5.iterrows()):
        values = [row['recall'], row['precision'], row['f1'], row['mrr']]
        values += values[:1]
        ax4.plot(angles, values, 'o-', linewidth=2, color=colors[i % len(colors)],
                label=row['dataset_name'])
        ax4.fill(angles, values, alpha=0.1, color=colors[i % len(colors)])

    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(categories)
    ax4.set_title('Top5综合指标雷达图', fontsize=14, fontweight='bold', pad=20)
    ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    plt.tight_layout()

    output_file = os.path.join(output_dir, 'comprehensive_comparison.png')
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    print(f"综合对比图已保存: {output_file}")


def generate_report(df: pd.DataFrame, output_dir: str):
    """
    生成文字报告

    Args:
        df: 评测结果DataFrame
        output_dir: 输出目录
    """
    df_no_rerank = df[df['use_rerank'] == False]
    df_with_rerank = df[df['use_rerank'] == True]

    # 找出最佳配置
    best_recall_idx = df_no_rerank['recall'].idxmax()
    best_recall_config = df_no_rerank.loc[best_recall_idx]

    best_mrr_idx = df_no_rerank['mrr'].idxmax()
    best_mrr_config = df_no_rerank.loc[best_mrr_idx]

    report = f"""
# RAG 知识库检索评测报告

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 评测概要

- 总评测配置数: {len(df)}
- 评测查询数: {df['total_queries'].iloc[0]}
- 分块策略: {', '.join(df['dataset_name'].unique())}
- TopK范围: {', '.join([f'Top{k}' for k in sorted(df['top_k'].unique())])}

## 最佳配置

### 召回率最高配置
- 配置: {best_recall_config['dataset_name']} + Top{best_recall_config['top_k']}
- Recall@K: {best_recall_config['recall']:.2%}
- MRR: {best_recall_config['mrr']:.4f}

### MRR最高配置
- 配置: {best_mrr_config['dataset_name']} + Top{best_mrr_config['top_k']}
- Recall@K: {best_mrr_config['recall']:.2%}
- MRR: {best_mrr_config['mrr']:.4f}

## 分块策略对比（Top5, 无重排）

| 分块策略 | Recall@5 | Precision@5 | F1@5 | MRR | 平均延迟 |
|---------|----------|-------------|------|-----|---------|
"""

    df_top5_no_rerank = df_no_rerank[df_no_rerank['top_k'] == 5]
    for _, row in df_top5_no_rerank.iterrows():
        report += f"| {row['dataset_name']} | {row['recall']:.2%} | {row['precision']:.4f} | {row['f1']:.4f} | {row['mrr']:.4f} | {row['avg_latency_ms']:.0f}ms |\n"

    # 重排效果分析
    if len(df_with_rerank) > 0:
        report += """
## 重排效果分析（Top5）

| 分块策略 | 无重排Recall | 有重排Recall | 提升 |
|---------|-------------|-------------|------|
"""
        for dataset in df['dataset_name'].unique():
            no_rerank = df_no_rerank[(df_no_rerank['dataset_name'] == dataset) & (df_no_rerank['top_k'] == 5)]
            with_rerank = df_with_rerank[(df_with_rerank['dataset_name'] == dataset) & (df_with_rerank['top_k'] == 5)]

            if len(no_rerank) > 0 and len(with_rerank) > 0:
                recall_no = no_rerank['recall'].iloc[0]
                recall_with = with_rerank['recall'].iloc[0]
                improvement = recall_with - recall_no
                report += f"| {dataset} | {recall_no:.2%} | {recall_with:.2%} | {improvement:+.2%} |\n"

    report += """
## 结论与建议

"""

    # 自动生成建议
    datasets = df_no_rerank['dataset_name'].unique()
    recall_by_dataset = df_no_rerank.groupby('dataset_name')['recall'].mean()
    best_dataset = recall_by_dataset.idxmax()

    report += f"1. **推荐分块策略**: {best_dataset}（平均召回率最高: {recall_by_dataset[best_dataset]:.2%}）\n\n"

    # TopK建议
    recall_by_topk = df_no_rerank.groupby('top_k')['recall'].mean()
    best_topk = recall_by_topk.idxmax()
    report += f"2. **推荐TopK值**: Top{best_topk}（该TopK下平均召回率: {recall_by_topk[best_topk]:.2%}）\n\n"

    # 重排建议
    if len(df_with_rerank) > 0:
        avg_improvement = (df_with_rerank['recall'].mean() - df_no_rerank[df_no_rerank['top_k'].isin(df_with_rerank['top_k'].unique())]['recall'].mean())
        if avg_improvement > 0.05:
            report += f"3. **重排效果显著**: 平均提升 {avg_improvement:.2%}，建议启用重排器\n\n"
        else:
            report += f"3. **重排效果有限**: 平均提升仅 {avg_improvement:.2%}，可根据延迟要求决定是否启用\n\n"

    report += """
---
报告由 RAG 评测工具自动生成
"""

    # 保存报告
    report_file = os.path.join(output_dir, f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"评测报告已保存: {report_file}")

    return report


def main(summary_json_path: str, output_dir: str = './results'):
    """
    主函数：生成所有可视化图表和报告

    Args:
        summary_json_path: 汇总JSON文件路径
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)

    # 加载数据
    df = load_summary_data(summary_json_path)

    print("开始生成可视化图表...")

    # 生成各类图表
    plot_recall_comparison(df, output_dir)
    plot_metrics_heatmap(df, output_dir)
    plot_latency_comparison(df, output_dir)
    plot_comprehensive_comparison(df, output_dir)

    # 生成文字报告
    report = generate_report(df, output_dir)

    print("\n所有图表和报告已生成完成！")
    print(f"输出目录: {output_dir}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='评测结果可视化工具')
    parser.add_argument('--input', type=str, required=True, help='汇总JSON文件路径')
    parser.add_argument('--output-dir', type=str, default='./results', help='输出目录')

    args = parser.parse_args()

    main(args.input, args.output_dir)
