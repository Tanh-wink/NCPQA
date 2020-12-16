from utils import *
import gensim
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


def cal_top_acc(result, K):
    correct = [0 for _ in range(K)]
    bad_samples = []
    with tqdm(total=len(result)) as pbar:
        for k in range(K):
            pbar.reset()
            pbar.set_description('compute top-{} acc'.format(k+1))
            for _id, item in result.items():
                true_docid = item['true_docid']
                doc_scores = item['doc_scores']
                top_K_dcot = [doc_id for doc_id, _ in doc_scores[:k+1]]
                if true_docid in top_K_dcot:
                    correct[k] += 1
                else:
                    if k == 9:
                        bad_samples.append(item)
                pbar.update(1)
    for k in range(K):
        print('top-{} acc is {:.2%}'.format(k+1, correct[k]*1.0/len(result)))

    save_pkl_data(bad_samples, path.get('bad_samples'))

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
            question_list.append(pynlpir.segment(question, pos_tagging=False))
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
                _id, _context = item
                context_list.append(pynlpir.segment(_context["text"], pos_tagging=False))
                context_idmap[_id] = str(index)
                context_idmap[str(index)] = _id
                pbar.update(1)
        save_pkl_data(context_list, path.get('context_list'))
        save_pkl_data(context_idmap, path.get('context_idmap'))


    return question_list, question_idmap, context_list, context_idmap


def build_list_and_idmap_addPos(train_dict, context_dict, path=None):
    question_list = []
    question_idmap = {}

    context_list = []
    context_idmap = {}

    with tqdm(total=len(train_dict)) as pbar:
        pbar.set_description('build list_and_idmap of train_dict')
        for index, item in enumerate(train_dict.items()):
            _id, _item = item
            question = _item['question']
            question_list.append(pynlpir.segment(question, pos_tagging=True, pos_english=False))
            question_idmap[_id] = str(index)
            question_idmap[str(index)] = _id
            pbar.update(1)

    if path and os.path.exists(path.get('context_idmap')) and os.path.exists(path.get('context_list_with_pos')):
        context_list = load_pkl_data(path.get('context_list_with_pos'))
        context_idmap = load_pkl_data(path.get('context_idmap'))
    else:
        with tqdm(total=len(context_dict)) as pbar:
            pbar.set_description('build list_and_idmap of context_dict')
            for index, item in enumerate(context_dict.items()):
                _id, _context = item
                context_list.append(pynlpir.segment(_context["text"], pos_tagging=True, pos_english=False))
                context_idmap[_id] = str(index)
                context_idmap[str(index)] = _id
                pbar.update(1)
        save_pkl_data(context_list, path.get('context_list_with_pos'))
        save_pkl_data(context_idmap, path.get('context_idmap'))


    return question_list, question_idmap, context_list, context_idmap



if __name__ == '__main__':
    pynlpir.open()
    path = {
        'train': '../data/train_data1.pkl',
        'test': '../data/test_data.pkl',
        'context': '../data/policies_context.pkl',
        'train_dict': '../data/rank/train_dict3.pkl',
        'context_dict': '../data/rank/context_dict3.pkl',
        'context_corpus': '../data/rank/context_corpus3.pkl',
        'result': '../data/rank/result_bm25_wordProcess_addPos3.pkl',
        'bad_samples': '../data/rank/bad_samples3.pkl',
        'context_list': '../data/rank/context_list3.pkl',
        'context_idmap': '../data/rank/context_idmap3.pkl',
        'context_list_with_pos': '../data/rank/context_list_with_pos3.pkl',
    }

    if os.path.exists(path.get('result')) and True:
        result = load_pkl_data(path.get('result'))
    else:
        train = load_pkl_data(path.get('train'))
        test = load_pkl_data(path.get('test'))
        context = load_pkl_data(path.get('context'))

        train_dict, train_ids = get_train_dict_and_ids(path.get('train_dict'), train)
        context_dict = get_context_dict(path.get('context_dict'), context)
        question_list, question_idmap, context_list, context_idmap = build_list_and_idmap(train_dict, context_dict, path)
        question_withPos_list, _, context_withPos_list, _ = build_list_and_idmap_addPos(train_dict, context_dict,
                                                                                        path)
        specical_pos = {
            '。': '标点符号', ' ': '标点符号', '？': '标点符号', '\u3000': '标点符号', '；': '标点符号', '！': '标点符号', '!': '标点符号',
            '?': '标点符号', '[': '标点符号', '大': '名词', '中': '名词', '小': '名词',
        }
        question_list_concat_pos = []
        for index, item in enumerate(question_withPos_list):
            try:
                question_concat_pos = []
                for word, pos in item:
                    pos = specical_pos.get(word, pos)
                    question_concat_pos.append(word+'_'+pos)
                question_list_concat_pos.append(question_concat_pos)
            except:
                print(index)
                print(item)
                break

        context_list_concat_pos = []
        stop = False
        for index, item in enumerate(context_withPos_list):
            context_concat_pos = []
            for word, pos in item:
                try:
                    pos = specical_pos.get(word, pos)
                    context_concat_pos.append(word + '_' + pos)
                except:
                    print(word)
                    print(pos)
                    stop = True
                    break
            if stop:
                break
            context_list_concat_pos.append(context_concat_pos)

        max_freq = 1000

        all_sentences = question_list_concat_pos + context_list_concat_pos
        vocab_all = gensim.corpora.Dictionary(all_sentences)

        dfs_all = []
        for _id, _dfs in vocab_all.dfs.items():
            dfs_all.append(_dfs)
        dfs_all = pd.DataFrame(dfs_all)

        dfs_all_filter = {}
        for _id, _dfs in vocab_all.dfs.items():
            if _dfs <= max_freq:
                dfs_all_filter[vocab_all[_id]] = True


        good_pos = ['名词', '时间词', '处所词', '方位词', '动词', '数词', '量词', '副词', '形容词']
        bad_word = ['新冠', "疫情", "防控", "病毒", "战疫", "肺炎", "新冠病毒"]
        question_f = []
        question_f_bad = []
        with tqdm(total=len(train)) as pbar:
            pbar.set_description('filter question by fs and pos')
            for item in train:
                qid = item['qid']
                q_cut = question_list_concat_pos[int(question_idmap[qid])]
                question = []
                for word in q_cut:
                    _, pos = word.split('_')
                    if dfs_all_filter.get(word, False) or pos in good_pos:
                        if word not in bad_word:
                            question.append(word)
                    else:
                        continue
                if len(question) == 0:
                    isBad = True
                    q = {
                        'question': q_cut,
                        'isBad': isBad,
                        'docid': item['docid'],
                        'id': qid,
                    }
                    question_f_bad.append(q)
                else:
                    isBad = False
                    q = {
                        'question': question,
                        'isBad': isBad,
                        'docid': item['docid'],
                        'id': qid,
                    }
                    question_f.append(q)
                pbar.update(1)
        if True:
            print('Empty in q after filtering, have {} empty q'.format(len(question_f_bad)))

        raw_num = 0
        new_num = 0
        for index, raw_text in enumerate(question_list):
            raw_num += len(raw_text)
            new_num += len(question_f[index])
        print('raw_num:{}, new_num:{}, diff: {} in question'.format(raw_num, new_num, raw_num - new_num))


        context_f = []
        context_f_bad = []
        with tqdm(total=len(context_list_concat_pos)) as pbar:
            pbar.set_description('filter context by fs and pos')
            for item in context_list_concat_pos:
                context = []
                for word in item:
                    pos = word.split('_')[-1]
                    if dfs_all_filter.get(word, False) or pos in good_pos:
                        context.append(word)
                    else:
                        continue
                if len(context) == 0:
                    context_f_bad.append(context)
                else:
                    context_f.append(context)
                pbar.update(1)
        if True:
            print('Empty in context after filtering, have {} empty context'.format(len(context_f_bad)))

        raw_num = 0
        new_num = 0
        for index, raw_text in enumerate(context_list):
            raw_num += len(raw_text)
            new_num += len(context_f[index])
        print('raw_num:{}, new_num:{}, diff: {} in context'.format(raw_num, new_num, raw_num - new_num))

        bm25_model = BM25(context_f)

        result = {}
        with tqdm(total=len(question_f)) as pbar:
            pbar.set_description('compute bm25 for each question')
            for index, q_item in enumerate(question_f):

                question = q_item.get('question')
                isBad = q_item.get('isBad')
                true_docid = q_item.get('docid')
                qid = q_item.get('id')

                if isBad:
                    pass
                else:
                    bm25_score = bm25_model.get_scores(question)
                    bm25_score = [[context_idmap[str(index)], score] for index, score in enumerate(bm25_score)]
                    bm25_score.sort(key=op.itemgetter(1), reverse=True)

                    result[qid] = {
                        'true_docid': true_docid,
                        'id': qid,
                        'doc_scores': bm25_score,
                    }
                pbar.update(1)
        save_pkl_data(result, path.get('result'))
    cal_top_acc(result, K=100)
    cal_MAP(result)

    print('end')
    pynlpir.close()