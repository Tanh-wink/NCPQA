from data_utils import split_text
import os
import pickle
from tqdm import tqdm
import json

data_path = '../data'
train_path = os.path.join(data_path, 'train_data1.pkl')
doc_path = os.path.join(data_path, 'policies_context.pkl')

def get_data(path):
    with open(path, 'rb') as fp:
        data = pickle.load(fp)
    return data


def build_docs(docs, save_path):
    text_id = 1000
    all_docs = []
    for doc in tqdm(docs):
        docid = doc['docid']
        text = doc['text']
        sub_texts, starts = split_text(text, maxlen=384, greedy=False)
        for sub_text, start in zip(sub_texts, starts):
            cur_doc = {}
            cur_doc["docid"] = docid
            cur_doc["text_id"] = text_id
            cur_doc["text"] = sub_text
            cur_doc["start"] = start
            text_id += 1
            all_docs.append(cur_doc)
    print('number of all docs:{}'.format(len(all_docs)))
    with open(save_path, 'wb') as fp:
        pickle.dump(all_docs, fp)

def build_train(train, save_path):
    text_id = 1000
    all_train = []
    for data in tqdm(train):
        docid = data['docid']
        question = data['question']
        qid = data['qid']
        text_ids = data['text_ids']  # list
        answer = data['answer']
        answer_span = data['answer_span']
        sub_texts, starts = split_text(text, maxlen=384, greedy=False)
        for sub_text, start in zip(sub_texts, starts):
            cur_doc = {}
            cur_doc["docid"] = docid
            cur_doc["text_id"] = text_id
            cur_doc["text"] = sub_text
            cur_doc["start"] = start
            text_id += 1
            all_docs.append(cur_doc)
    print('number of all docs:{}'.format(len(all_docs)))
    with open(save_path, 'wb') as fp:
        pickle.dump(all_docs, fp)


if __name__ == '__main__':
    train_path = '../data/rank/train_with_doc_top30_2.pkl'
    train = get_data(train_path)

    save_path = os.path.join(data_path, 'documents.pkl')
    build_train(train, save_path)


