English | [简体中文](README_zh.md)

## Datafountain-Epidemic government affairs quiz assistant competition
[Official Competition](https://www.datafountain.cn/competitions/424)  
## Data Declaration
Contains 3 files: corpus.csv, train.csv, test.csv.

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
According to user questions, the policy document containing the answer span is retrieved from 8943 policy documents, and then the answer span is extracted from the retrieved policy document and returned to the user as an answer.

## Implementation Details
We divided this task into two parts: document retrieval and answer extraction

1. Document Retrieval  
  use BM25 model and es to retrieve relevant top10 policy documents containing answer from 8932 epidemic policy documents according to user question. 
  For preprocessed 8943 documents, they were segmented by word. And get the corresponding part of speech of each word.
2. Answer Extraction  
  对top10 政策文件通过滑动窗口切成成若干子文档，然后与用户问题拼接输入到模型中进行答案抽取。
  选取 Albert 作为最终机器阅读理解模型。通过结合多任务训练、 答案选择与模型融合等技术对模型进行优化。

## Model
We choose Albert as the baseline, and optimize the model by combining multiple strategies such as multi-task training and answer selection.  
Then use 10-fold cross training, and finally model fusion.

## Criterion  
  Rouge-L：0.7044
  
