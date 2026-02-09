#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评测集构建工具
用于从知识库文档中生成候选问题，辅助人工标注
"""

import os
import json
import pandas as pd
from dotenv import load_dotenv
import requests
from tqdm import tqdm

# 加载环境变量
load_dotenv()

class EvaluationSetBuilder:
    """评测集构建器"""

    def __init__(self):
        self.api_base = os.getenv('DIFY_API_BASE', 'https://api.dify.ai/v1')
        self.api_key = os.getenv('DIFY_API_KEY')
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

    def get_dataset_documents(self, dataset_id: str) -> list:
        """
        获取知识库中的所有文档列表

        Args:
            dataset_id: 知识库ID

        Returns:
            文档列表
        """
        url = f"{self.api_base}/datasets/{dataset_id}/documents"
        all_documents = []
        page = 1

        while True:
            params = {'page': page, 'limit': 20}
            response = requests.get(url, headers=self.headers, params=params)

            if response.status_code != 200:
                print(f"获取文档列表失败: {response.text}")
                break

            data = response.json()
            documents = data.get('data', [])

            if not documents:
                break

            all_documents.extend(documents)

            if not data.get('has_more', False):
                break

            page += 1

        print(f"共获取到 {len(all_documents)} 个文档")
        return all_documents

    def get_document_segments(self, dataset_id: str, document_id: str) -> list:
        """
        获取文档的所有分段

        Args:
            dataset_id: 知识库ID
            document_id: 文档ID

        Returns:
            分段列表
        """
        url = f"{self.api_base}/datasets/{dataset_id}/documents/{document_id}/segments"
        all_segments = []
        page = 1

        while True:
            params = {'page': page, 'limit': 100}
            response = requests.get(url, headers=self.headers, params=params)

            if response.status_code != 200:
                print(f"获取分段失败: {response.text}")
                break

            data = response.json()
            segments = data.get('data', [])

            if not segments:
                break

            all_segments.extend(segments)

            if not data.get('has_more', False):
                break

            page += 1

        return all_segments

    def extract_candidate_questions(self, segment_text: str) -> list:
        """
        从文本段落中提取候选问题
        基于规则的提取（偏高精度、低召回，便于人工筛选）。
        目标：尽量避免从噪音文本（如新闻站点水印/责任编辑/广告）生成“伪问题”。

        Args:
            segment_text: 段落文本

        Returns:
            候选问题列表
        """
        import re
        from collections import OrderedDict

        max_questions = 5

        if not segment_text:
            return []

        # 规范化
        text = str(segment_text)
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        text = re.sub(r'<[^>]+>', '', text)  # 去HTML标签
        text = re.sub(r'[ \t]+', ' ', text).strip()

        # 常见噪音/水印行过滤（覆盖“返回搜狐/责任编辑”等）
        noise_line_re = re.compile(
            r'(?:返回搜狐|查看更多|责任编辑|原标题|来源[:：]|免责声明|版权声明|'
            r'点击.*?关注|阅读原文|扫码|二维码|更多精彩|关注我们|公众号|微信|微博|'
            r'广告|推广|转载|版权归|导航|目录|上一篇|下一篇|相关推荐)',
            re.IGNORECASE
        )

        def has_meaningful_text(s: str) -> bool:
            return bool(re.search(r'[\u4e00-\u9fffA-Za-z]', s))

        # 过滤后行列表
        lines = []
        for raw in text.split('\n'):
            line = raw.strip()
            if not line:
                continue
            if noise_line_re.search(line):
                continue
            if not has_meaningful_text(line):
                continue
            # 过短的单行通常是标点/残片
            if len(line) <= 2:
                continue
            lines.append(line)

        if not lines:
            return []

        cleaned_text = "\n".join(lines)

        # 候选问题按“来源类型”从高到低写入，最后去重截断
        candidates = []

        def normalize_question(q: str) -> str:
            q = q.strip()
            q = re.sub(r'\s+', ' ', q)
            # 去掉常见 Q/A 前缀
            q = re.sub(r'^(?:问|问题|Q)\s*[：:]\s*', '', q, flags=re.IGNORECASE)
            q = q.strip('：:;；')
            # 统一问号
            if q and not q.endswith(('?', '？')):
                q += '？'
            # 中文问号优先
            q = q.replace('?', '？')
            return q

        def add_candidate(q: str):
            q = normalize_question(q)
            if not q:
                return
            # 长度过滤
            if len(q) < 6 or len(q) > 80:
                return
            # 二次噪音过滤
            if noise_line_re.search(q):
                return
            candidates.append(q)

        # 规则1：显式 Q/A 结构：问： / Q:
        qa_prefix_re = re.compile(r'^(?:问|问题|Q)\s*[：:]\s*(.+)$', re.IGNORECASE)
        for line in lines:
            m = qa_prefix_re.match(line)
            if not m:
                continue
            add_candidate(m.group(1))

        # 规则2：直接包含问号的“行内”句子（避免跨行把整段拼进来）
        for line in lines:
            if '？' in line or '?' in line:
                q = re.split(r'[？?]', line, maxsplit=1)[0].strip()
                add_candidate(q)

        # 规则3：疑问词开头的句子，转成问句（逐行处理更稳）
        question_starters = ('如何', '怎么', '怎样', '为什么', '为何', '什么是', '什么叫', '哪些', '是否', '能否', '可否')
        for line in lines:
            if line.startswith(question_starters):
                add_candidate(line)

        # 规则4：标题/小节行 -> 模板化生成问题（比之前更严格）
        heading_prefix_re = re.compile(
            r'^(?:'
            r'第?[一二三四五六七八九十\d]+[章节部分]\s*|'
            r'[\(\（]?\d+[\)\）]?\s*[\.、\)]\s*|'
            r'[\(\（]?[一二三四五六七八九十]+[\)\）]?\s*[\.、\)]\s*'
            r')'
        )
        process_hint_re = re.compile(r'(?:流程|步骤|方法|指南|说明|规范|规则|注意事项|常见问题|FAQ|Q&A)', re.IGNORECASE)
        action_hint_re = re.compile(r'(?:配置|安装|部署|接入|对接|申请|办理|使用|创建|上传|下载|更新|删除|查询|设置|开通)')

        def looks_like_heading(line: str) -> bool:
            if len(line) < 6 or len(line) > 40:
                return False
            # 含逗号的更可能是正文/水印（如“返回搜狐，查看更多…”）
            if '，' in line or ',' in line:
                return False
            if re.search(r'[。！？!?]$', line):
                return False
            if not has_meaningful_text(line):
                return False
            # 典型标题提示：编号/冒号/关键字
            if heading_prefix_re.match(line):
                return True
            if line.endswith(('：', ':')):
                return True
            if process_hint_re.search(line) or action_hint_re.search(line):
                return True
            return False

        for line in lines:
            if not looks_like_heading(line):
                continue

            clean = heading_prefix_re.sub('', line).strip().rstrip('：:')
            if not clean:
                continue

            # 已经是问句/疑问式标题
            if clean.startswith(question_starters) or clean.endswith(('？', '?')):
                add_candidate(clean)
                continue

            if action_hint_re.search(clean):
                add_candidate(f"如何{clean}")
            elif process_hint_re.search(clean):
                add_candidate(f"{clean}有哪些步骤")
                add_candidate(f"{clean}是什么")
            else:
                add_candidate(f"什么是{clean}")

        # 规则5（可选）：关键词补充（依赖 jieba；没装也不影响主流程）
        try:
            import jieba.analyse  # type: ignore
            keywords = jieba.analyse.textrank(cleaned_text, topK=5, withWeight=False)
            for kw in keywords:
                kw = str(kw).strip()
                if len(kw) < 2 or len(kw) > 12:
                    continue
                if noise_line_re.search(kw):
                    continue
                add_candidate(f"什么是{kw}")
        except Exception:
            pass

        # 去重（保留顺序）
        uniq = list(OrderedDict.fromkeys(candidates))
        return uniq[:max_questions]  # 每个段落最多返回 max_questions 个候选问题

    def build_candidate_set(self, dataset_id: str, output_file: str = 'candidate_questions.xlsx'):
        """
        构建候选评测集

        Args:
            dataset_id: 知识库ID
            output_file: 输出文件名
        """
        print("开始构建候选评测集...")

        # 获取所有文档
        documents = self.get_dataset_documents(dataset_id)

        candidates = []

        for doc in tqdm(documents, desc="处理文档"):
            doc_id = doc['id']
            doc_name = doc.get('name', 'unknown')

            # 获取文档分段
            segments = self.get_document_segments(dataset_id, doc_id)

            for seg in segments:
                seg_id = seg.get('id', '')
                seg_text = seg.get('content', '')

                if not seg_text:
                    continue

                # 提取候选问题
                questions = self.extract_candidate_questions(seg_text)

                for q in questions:
                    candidates.append({
                        'id': len(candidates) + 1,
                        'query': q,
                        'gold_doc_id': doc_id,
                        'gold_doc_name': doc_name,
                        'gold_segment_id': seg_id,
                        'gold_chunk_text': seg_text[:200],  # 截取前200字符
                        'category': '',  # 待人工填写
                        'difficulty': '',  # 待人工填写
                        'is_valid': '',  # 待人工确认（Y/N）
                    })

        # 保存到Excel
        df = pd.DataFrame(candidates)
        df.to_excel(output_file, index=False, engine='openpyxl')

        print(f"候选评测集已保存到: {output_file}")
        print(f"共生成 {len(candidates)} 条候选问题")
        print("请人工审核并标注 is_valid、category、difficulty 列")

        return df


def create_empty_template(output_file: str = 'evaluation_set_template.xlsx'):
    """
    创建空的评测集模板

    Args:
        output_file: 输出文件名
    """
    template = pd.DataFrame({
        'id': [1, 2, 3],
        'query': ['示例问题1：年假有多少天', '示例问题2：怎么申请报销', '示例问题3：试用期多长'],
        'gold_doc_id': ['doc_001', 'doc_002', 'doc_003'],
        'gold_doc_name': ['员工手册', '费用报销制度', '入职指南'],
        'gold_chunk_text': ['员工入职满一年可享受5天带薪年假', '报销流程：填写报销单后提交审批', '试用期一般为3个月'],
        'category': ['请假', '报销', '入职'],
        'difficulty': ['easy', 'medium', 'easy'],
    })

    template.to_excel(output_file, index=False, engine='openpyxl')
    print(f"评测集模板已创建: {output_file}")
    print("请按模板格式填写您的评测数据")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='评测集构建工具')
    parser.add_argument('--action', choices=['template', 'build'], default='template',
                        help='操作类型: template-创建空模板, build-从知识库构建')
    parser.add_argument('--dataset-id', type=str, help='知识库ID（build模式需要）')
    parser.add_argument('--output', type=str, default=None, help='输出文件名（build 默认 candidates.xlsx；template 默认 evaluation_set_template.xlsx）')

    args = parser.parse_args()

    if args.action == 'template':
        out = args.output or 'evaluation_set_template.xlsx'
        create_empty_template(out)
    elif args.action == 'build':
        if not args.dataset_id:
            print("错误: build模式需要指定 --dataset-id")
            exit(1)
        builder = EvaluationSetBuilder()
        out = args.output or 'candidates.xlsx'
        builder.build_candidate_set(args.dataset_id, out)
