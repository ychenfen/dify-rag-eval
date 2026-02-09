#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CRUD-RAG 文档提取工具
用于从80000篇文档中提取指定数量的文档用于测试
"""

import os
import sys
import argparse
from pathlib import Path


def extract_documents(source_dir, output_dir, num_files=10):
    """
    从源目录提取指定数量的文档文件

    Args:
        source_dir: 源文档目录 (80000_docs)
        output_dir: 输出目录
        num_files: 要提取的文件数量
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有文档文件
    source_path = Path(source_dir)
    if not source_path.exists():
        print(f"错误: 源目录不存在: {source_dir}")
        sys.exit(1)

    # 获取所有文档文件并排序
    doc_files = sorted([f for f in source_path.iterdir() if f.is_file()])

    if not doc_files:
        print(f"错误: 源目录中没有找到文档文件: {source_dir}")
        sys.exit(1)

    print(f"找到 {len(doc_files)} 个文档文件")
    print(f"将提取前 {num_files} 个文件到: {output_dir}")
    print("-" * 50)

    # 提取指定数量的文件
    extracted_count = 0
    total_size = 0

    for i, doc_file in enumerate(doc_files[:num_files]):
        try:
            # 读取源文件
            with open(doc_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # 写入目标文件
            output_file = Path(output_dir) / doc_file.name
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(content)

            file_size = doc_file.stat().st_size
            total_size += file_size
            extracted_count += 1

            print(f"✓ [{i+1}/{num_files}] {doc_file.name} ({file_size/1024/1024:.2f} MB)")

        except Exception as e:
            print(f"✗ 提取失败: {doc_file.name} - {str(e)}")

    print("-" * 50)
    print(f"提取完成!")
    print(f"成功提取: {extracted_count} 个文件")
    print(f"总大小: {total_size/1024/1024:.2f} MB")
    print(f"输出目录: {output_dir}")


def count_documents(doc_dir):
    """
    统计文档目录中的文档数量

    Args:
        doc_dir: 文档目录
    """
    doc_path = Path(doc_dir)
    if not doc_path.exists():
        print(f"错误: 目录不存在: {doc_dir}")
        return

    total_docs = 0
    total_size = 0

    for doc_file in doc_path.iterdir():
        if doc_file.is_file():
            try:
                with open(doc_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # 简单统计: 按日期分隔符计数
                    doc_count = content.count('，正文：')
                    total_docs += doc_count
                    total_size += doc_file.stat().st_size
            except Exception as e:
                print(f"读取失败: {doc_file.name} - {str(e)}")

    print(f"文档统计:")
    print(f"  文件数: {len(list(doc_path.iterdir()))}")
    print(f"  文档数: {total_docs} 篇")
    print(f"  总大小: {total_size/1024/1024:.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description='CRUD-RAG 文档提取工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 提取前10个文件(约1000篇文档)
  python extract_docs.py --num 10

  # 提取前30个文件(约3000篇文档)
  python extract_docs.py --num 30 --output test_docs_3k

  # 统计文档数量
  python extract_docs.py --count
        """
    )

    parser.add_argument(
        '--source',
        type=str,
        default='CRUD_RAG/data/80000_docs',
        help='源文档目录 (默认: CRUD_RAG/data/80000_docs)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='CRUD_RAG/data/test_docs',
        help='输出目录 (默认: CRUD_RAG/data/test_docs)'
    )

    parser.add_argument(
        '--num',
        type=int,
        default=10,
        help='要提取的文件数量 (默认: 10, 约1000篇文档)'
    )

    parser.add_argument(
        '--count',
        action='store_true',
        help='统计文档数量'
    )

    args = parser.parse_args()

    if args.count:
        count_documents(args.source)
    else:
        extract_documents(args.source, args.output, args.num)


if __name__ == '__main__':
    main()
