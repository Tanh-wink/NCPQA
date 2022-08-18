English | [简体中文](README_zh.md)

## Datafountain-Epidemic government affairs quiz assistant competition
[Official Competition](https://www.datafountain.cn/competitions/424)  
## Data Declaration
Contain 3 files: corpus.csv, train.csv, test.csv.

1.corpus.csv: content of the policy file, UTF-8, separated by Tab.  

|Fieldname|Field Description  |
|  ----  | ----  |
|docid	|Policy document's id |
|text	|Content of the Policy document  |

2.train.csv：train set，UTF-8，separated by Tab. 
|Fieldname|Field Description  |
|  ----  | ----  |
|qid	|query id |
|query	|user's query |
|docid	|Id of Policy document containing answer text | 
|answer	|answer text |

3.test.csv：test set，UTF-8，separated by Tab. 
|Fieldname|Field Description  |
|  ----  | ----  |
|qid|	query id |
|query|	user's query | 

Submit file is a csv file，encoding by UTF-8，separated by Tab.  
|Fieldname|Field Description  |
|  ----  | ----  |
|qid	|query id  |
|docID|	Id of Policy document containing answer text | 
|answer|	answer text  | 

## Mission Introduction
给定的8943条政策文件， 根据用户问题，先在8900多条数据中检索出答案所在的政策文件，再对检索出来的政策文件进行提取答案片段作为回答返回给用户。  

## Implementation Details
本任务将其分为两部分：检索文件和问答抽取
1. 检索部分  
  采用 BM25 检索算法和es 根据用户问题进行检索出相关性 top10 政策文件。 
  预处理8943文档，按字分割，并得到对应词性。
2. 问答抽取  
  对top10 政策文件通过滑动窗口切成成若干子文档，然后与用户问题拼接输入到模型中进行答案抽取。
  选取 Albert 作为最终机器阅读理解模型。通过结合多任务训练、 答案选择与模型融合等技术对模型进行优化。

## Model
Albert为baseline，改进策略：多任务训练、 答案选择。  
最后10折交叉，然后模型融合。

## Criterion  
  Rouge-L：0.7044
  
