from utils import *
import pickle
import gensim
import re
from pyhanlp import *
import pynlpir
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import operator as op
from gensim.summarization.bm25 import BM25

def check_q_and_context(train_ids, train_dict, context_dict):
    while True:
        print('='*30)
        qid = input('输入问题的id (1~5000) 【退出按 Q】')
        if qid == 'Q':
            break
        qid = train_ids[int(qid)-1]
        question = train_dict.get(qid).get('question')
        docid = train_dict.get(qid).get('docid')
        context = context_dict.get(docid)
        print('Q: {}'.format(question))
        print('context: {}'.format(context))
        print('=' * 30)



def cal_sim_QandC_by_dfs(question_f, context_vocab, restore=True):
    save_path = path.get('result')
    if os.path.exists(save_path) and restore:
        print('./result_dfs.pkl is existed')
        result = load_pkl_data(save_path)
    else:
        result = {}
        with tqdm(total=len(question_f)) as pbar:
            for q_item in question_f:
                question = q_item.get('question')
                isBad = q_item.get('isBad')
                true_docid = q_item.get('docid')
                qid = q_item.get('id')

                if isBad:
                    continue
                else:
                    doc_scores = []
                    for docid, doc_dfs in context_vocab.items():
                        score = 0
                        for q_word in question:
                            score += doc_dfs.get(q_word, 0)
                        doc_scores.append([docid, score])
                    doc_scores.sort(key=op.itemgetter(1), reverse=True)
                    result[qid] = {
                        'true_docid': true_docid,
                        'id': qid,
                        'doc_scores': doc_scores,
                    }
                    pbar.update(1)
        save_pkl_data(result, save_path)
    return result


def cal_top_acc(result, context_dict, K):
    correct = [0 for _ in range(K)]
    bad_samples = []
    with tqdm(total=len(result)) as pbar:
        for k in range(K):
            pbar.reset()
            pbar.set_description('compute top-{} acc'.format(k+1))
            for _id, item in enumerate(result):
                true_docid = item['docid']
                # answer_start, answer_end = item['answer_span']
                docids = item['text_ids']
                top_K_docids = [text_id for text_id in docids[:k+1]]
                for docid in top_K_docids:

                    if true_docid == docid:
                        correct[k] += 1
                        break
                        # doc_start = doc["start"]
                        # text_length = len(doc["text"])
                        # doc_end = doc_start + text_length - 1
                        # if answer_start >= doc_start and answer_start <= doc_end:
                        #     if answer_end <= doc_end:
                        #         correct[k] += 1
                        #         break

                pbar.update(1)
    for k in range(K):
        print('top-{} acc is {:.2%}'.format(k+1, correct[k]*1.0/len(result)))


def cal_MAP(result):
    with tqdm(total=len(result)) as pbar:
        AP = []
        for _id, item in result.items():
            true_docid = item['true_docid']
            doc_scores = item['doc_scores']
            doc_scores = [doc_id for doc_id, _ in doc_scores]
            true_docid_rank = doc_scores.index(true_docid) + 1
            P = 1 / true_docid_rank
            AP.append(P)
            pbar.update(1)
        MAP = np.array(AP).mean()
        print('MAP is {:.2%}'.format(MAP))


def build_list_and_idmap(train_dict, context_dict, path=None):
    question_list = []
    question_idmap = {}

    context_list = []
    context_idmap = {}

    with tqdm(total=len(train_dict)) as pbar:
        pbar.set_description('build list_and_idmap of train_dict')
        for index, item in enumerate(train_dict.items()):
            _id, _item = item
            question = _item['question']
            word_li = HanLP.segment(question)
            for word in word_li:
                question_list.append(word.word)

            # question_list.append(pynlpir.segment(question, pos_tagging=False))
            question_idmap[_id] = str(index)
            question_idmap[str(index)] = _id
            pbar.update(1)

    if path and os.path.exists(path.get('context_idmap')) and os.path.exists(path.get('context_list')):
        context_list = load_pkl_data(path.get('context_list'))
        context_idmap = load_pkl_data(path.get('context_idmap'))
    else:
        with tqdm(total=len(context_dict)) as pbar:
            pbar.set_description('build list_and_idmap of context_dict')
            for index, item in enumerate(context_dict.items()):
                _id, doc = item
                # word_li = HanLP.segment(doc["text"])
                # for word in word_li:
                #     context_list.append(word.word)
                context_list.append(pynlpir.segment(doc["text"], pos_tagging=False))
                # context_list.append(pynlpir.get_key_words(doc["text"]))
                context_idmap[_id] = str(index)
                context_idmap[str(index)] = _id
                pbar.update(1)
        save_pkl_data(context_list, path.get('context_list'))
        save_pkl_data(context_idmap, path.get('context_idmap'))


    return question_list, question_idmap, context_list, context_idmap


def get_test_with_doc(test, bm25_model, context_idmap, k=5):
    test_with_doc = []
    with tqdm(total=len(test)) as pbar:
        pbar.set_description('build test_with doc in top-{}'.format(k))
        for item in test:
            question = item['question']
            qid = item['id']
            q_cut = []
            word_li = HanLP.segment(question)
            for word in word_li:
                q_cut.append(word.word)

            # q_cut = pynlpir.segment(question, pos_tagging=False)
            bm25_score = bm25_model.get_scores(q_cut)
            bm25_score = [[context_idmap[str(index)], score] for index, score in enumerate(bm25_score)]
            bm25_score.sort(key=op.itemgetter(1), reverse=True)
            best_doc_id = [item[0] for item in bm25_score[:k]]
            test_sample = {
                'qid': qid,
                'question': question,
                'contexts': best_doc_id,
            }
            test_with_doc.append(test_sample)
            pbar.update(1)
        save_pkl_data(test_with_doc, './test_with_doc_top{}.pkl'.format(k))


def get_train_with_doc(train, bm25_model, context_idmap, k=5):
    save_path = '../data/rank/train_with_doc_top{}.pkl'.format(k)
    if os.path.exists(save_path):
        train_with_doc = load_pkl_data(save_path)
    else:
        train_with_doc = []
        with tqdm(total=len(train)) as pbar:
            pbar.set_description('build train with doc in top-{}'.format(k))
            for item in train:
                question = item['question']
                qid = item['qid']
                # q_cut = []
                # # word_li = HanLP.segment(question)
                # for word in word_li:
                #     q_cut.append(word.word)
                q_cut = pynlpir.segment(question, pos_tagging=False)
                # q_cut = pynlpir.get_key_words(question)
                bm25_score = bm25_model.get_scores(q_cut)
                bm25_score = [[context_idmap[str(index)], score] for index, score in enumerate(bm25_score)]
                bm25_score.sort(key=op.itemgetter(1), reverse=True)
                best_text_id = [item[0] for item in bm25_score[:k]]
                # if item['docid'] in best_doc_id:
                #     answer = item['answer']

                train_sample = {
                    'qid': qid,
                    'question': question,
                    'text_ids': best_text_id,
                    'answer': item['answer'],
                    'answer_span': item['answer_span'],
                    "docid":item['docid']
                }

                train_with_doc.append(train_sample)
                pbar.update(1)
            save_pkl_data(train_with_doc, save_path)
    return train_with_doc



if __name__ == '__main__':
    pynlpir.open()
    path = {
        'train': '../data/train_data1.pkl',
        'test': '../data/test_data.pkl',
        'documents': '../data/policies_context.pkl',
        'train_dict': '../data/rank/train_dict_y.pkl',
        'context_dict': './data/rank/context_dict2_y.pkl',
        'context_corpus': '../data/rank/context_corpus_y.pkl',
        'result': '../data/rank/result_bm25_y.pkl',
        'bad_samples': '../data/rank/bad_samples_y.pkl',
        'context_list': '../data/rank/context_list_y.pkl',
        'context_idmap': '../data/rank/context_idmap_y.pkl',
        'BM25_model': '../data/rank/BM25_model_y.gen'
    }

    if os.path.exists(path.get('result')) and True:
        result = load_pkl_data(path.get('result'))
    else:
        train = load_pkl_data(path.get('train'))
        test = load_pkl_data(path.get('test'))
        context = load_pkl_data(path.get('documents'))

        train_dict, train_ids = get_train_dict_and_ids(path.get('train_dict'), train)
        context_dict = get_context_dict(path.get('context_dict'), context)
        question_list, question_idmap, context_list, context_idmap = build_list_and_idmap(train_dict, context_dict, path)

        bm25_model = BM25(context_list)

        train_data = get_train_with_doc(train, bm25_model, context_idmap, 30)

        # result = {}
        # with tqdm(total=len(question_list)) as pbar:
        #     pbar.set_description('compute bm25 for each question')
        #     for index, question in enumerate(question_list):
        #         qid = question_idmap[str(index)]
        #         true_docid = train_dict[qid]['docid']
        #         bm25_score = bm25_model.get_scores(question)
        #         bm25_score = [[context_idmap[str(index)], score] for index, score in enumerate(bm25_score)]
        #         bm25_score.sort(key=op.itemgetter(1), reverse=True)
        #
        #         result[qid] = {
        #             'true_docid': true_docid,
        #             'id': qid,
        #             'doc_scores': bm25_score,
        #         }
        #         pbar.update(1)
        #
        # save_pkl_data(result, path.get('result'))
        cal_top_acc(train_data, context_dict, K=30)
    # cal_MAP(result)

    print('end')
    pynlpir.close()