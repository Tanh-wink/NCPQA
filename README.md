# NCPQA  
## Datafountain平台-疫情政务问答助手竞赛
[比赛官网](https://www.datafountain.cn/competitions/424)  
## 数据说明
数据包含3个文件，corpus.csv, train.csv, test.csv.

1.corpus.csv 政策文件内容，使用UTF-8编码，用Tab分隔。  
字段名称	字段说明  
docid	政策文件id  
text	政策内容  

2.train.csv：训练集，使用UTF-8编码，用Tab分隔。  
字段名称	字段说明  
qid	训练问题的id  
query	用户查询的问题  
docid	答案参考的政策文件id  
answer	答案  

3.test.csv：测试集，使用UTF-8编码，用Tab分隔。  
字段名称	字段说明  
qid	测试问题的id  
query	用户查询的问题  

提交测试格式为csv格式，使用UTF-8编码，Tab分隔，包含两个字段。  
字段名称	字段说明  
qid	测试问题的id  
docID	提交答案所在的政策文件id  
answer	预测的答案  

## 任务简介
给定的8943条政策文件， 根据用户问题，先在8900多条数据中检索出答案所在的政策文件，再对检索出来的政策文件进行提取答案片段作为回答返回给用户。  

## 实现
本任务将其分为两部分：检索文件和问答抽取
1. 检索部分使用es对用户问题进行检索出top10个文件
2. 问答抽取对top10文件进行答案片段抽取。

## 模型
Albert和10折交叉，然后模型融合。
