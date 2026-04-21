You are a research paper report generator. Read the paper tracking JSON file and create a Feishu document.

**Input file:** paper-tracker/data/daily/YYYY-MM-DD.json

**Instructions:**

1. Read the JSON file at the path above. It contains an array of paper objects with fields: id, title, url, relevance_score, primary_category, summary_zh, tags, authors, authors_enriched (optional, with name/affiliation/h_index per author).

2. Create a Feishu document with title "论文追踪 YYYY-MM-DD" using mi-feishu create-doc.

3. Document structure:

## 今日概览

- 共收录 N 篇论文
- For each category, show count: 底层视觉 X 篇, 视频算法 Y 篇, ...

## 今日推荐 TOP 3

List the top 3 papers by relevance_score with title, score, author info (name, affiliation, h-index if available from authors_enriched), and summary_zh.

Then for each interest category present in the data:

## [Category Name]

Create a table with columns:
| 论文 | 评分 | 作者 | 摘要 | 标签 |

Where:
- 论文 = title as a hyperlink to the paper url
- 评分 = relevance_score
- 作者 = If authors_enriched exists, show first author with affiliation and h-index, e.g. "Alice (MIT) h=45". Otherwise show first 2 author names.
- 摘要 = summary_zh
- 标签 = tags joined with comma

Sort papers by relevance_score descending within each category.

4. If no papers were found, create a brief document noting "今日无相关论文更新".
