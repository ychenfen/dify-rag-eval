# Dify Knowledge Base Notes (Demo)

## Chunking Strategies

- General chunking: splits text into fixed-size segments.
- Parent-child chunking: stores small child chunks for retrieval and links them to a larger parent chunk for context.
- QA chunking: extracts question-answer pairs as chunks.

## Reranking

Reranking is an optional step that reorders retrieved chunks based on a reranker model.
In Dify, the reranker provider and model name must match what is configured in "System Model Settings".

## Context Expansion

If the retrieval result returns only a small chunk, you can expand context by concatenating neighboring chunks from the same document.
This is typically done in the application layer after retrieval.

