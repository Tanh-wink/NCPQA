import pickle
import os
from tqdm import tqdm
import gensim
import pynlpir


def load_pkl_data(filePath):
    with open(filePath, 'rb') as fp:
        data_pkl = fp.read()
    return pickle.loads(data_pkl)


def save_pkl_data(data, filePath):
    data_pkl = pickle.dumps(data)
    with open(filePath, 'wb') as fp:
        fp.write(data_pkl)


def get_train_dict_and_ids(path, train=None):
    if os.path.exists(path):
        train_dict = load_pkl_data(path)
    else:
        train_dict = {}
        for item in train:
            train_dict[item['qid']] = item
        save_pkl_data(train_dict, path)
    train_ids = [qid for qid, _ in train_dict.items()]
    return train_dict, train_ids

def get_context_dict(path, context=None):
    if os.path.exists(path):
        context_dict = load_pkl_data(path)
    else:
        context_dict = {}
        for item in context:
            context_dict[item['docid']] = item
    return context_dict

def get_context_corpus(path, context_dict=None):
    if os.path.exists(path):
        context_corpus = load_pkl_data(path)
    else:
        context_corpus = {}
        with tqdm(total=len(context_dict)) as pbar:
            pbar.set_description('build context_corpus')
            for _id, _context in context_dict.items():
                context_cut_word = pynlpir.segment(_context, pos_tagging=False)
                context_corpus[_id] = context_cut_word
                pbar.update(1)
        save_pkl_data(context_corpus, path)

    return context_corpus

def build_context_vocab(context_corpus):
    # doc frequent 如果单词出现在文档中则为 1 ，不记录文档的频数
    context_vocab = {}
    with tqdm(total=len(context_corpus)) as pbar:
        pbar.set_description('build context_vocab by dfs')
        for _docid, _corpus in context_corpus.items():
            vocab = gensim.corpora.Dictionary([_corpus])
            vocab_dfs_word = {}
            for _id, _dfs in vocab.dfs.items():
                vocab_dfs_word[vocab[_id]] = _dfs
            context_vocab[_docid] = vocab_dfs_word
            pbar.update(1)

    return context_vocab

def build_context_vocab_fs(context_corpus):
    # BOW-like
    context_vocab_fs = {}
    with tqdm(total=len(context_corpus)) as pbar:
        pbar.set_description('build context_vocab_fs by df')
        for _docid, _corpus in context_corpus.items():
            vocab = gensim.corpora.Dictionary([_corpus])
            vocab_fs_word = {}
            for vid in vocab.keys():
                word = vocab[vid]
                fs = _corpus.count(word)
                vocab_fs_word[word] = fs
            context_vocab_fs[_docid] = vocab_fs_word
            pbar.update(1)

    return context_vocab_fs

def build_context_vocab_tfidf(context_corpus_list, idmap):
    # use tfidf
    context_vocab_tfidf = {}
    vocab = gensim.corpora.Dictionary(context_corpus_list)
    corpus_bow = [vocab.doc2bow(text) for text in context_corpus_list]
    tfidf_model = gensim.models.TfidfModel(corpus_bow)
    with tqdm(total=len(corpus_bow)) as pbar:
        pbar.set_description('build context_vocab_tfidf by tfidf')
        for index, _bow in enumerate(corpus_bow):
            _tfidf = tfidf_model[_bow]
            vocab_tfidf_word = {}
            for word_id, w_tfidf in _tfidf:
                word = vocab[word_id]
                vocab_tfidf_word[word] = w_tfidf
            docid = idmap[str(index)]
            context_vocab_tfidf[docid] = vocab_tfidf_word
            pbar.update(1)

    return context_vocab_tfidf


def get_context_corpus_list(context_corpus):
    context_corpus_list = []
    idmap = {}
    index = 0
    for _docid, _context in context_corpus.items():
        idmap[str(index)] = _docid
        idmap[_docid] = str(index)
        context_corpus_list.append(_context)
        index += 1

    return context_corpus_list, idmap



def get_question_corpus(train):
    question_corpus = []
    for item in train:
        cut_words = pynlpir.segment(item.get('question'), pos_tagging=False)
        if 'delete stop word':
            pass
        question_corpus.append(cut_words)
    return question_corpus

def filter_q_by_word_freq(dfs_q_filters, train=None):
    question_f = []
    with tqdm(total=len(train)) as pbar:
        pbar.set_description('filter question by q_dfs')
        for item in train:
            q_cut = pynlpir.segment(item.get('question'), pos_tagging=False)
            question = [word for word in q_cut if word in dfs_q_filters]
            if len(question) == 0:
                isBad = True
            else:
                isBad = False
            q = {
                'question': question,
                'isBad': isBad,
                'docid': item['docid'],
                'id': item['id'],
                }
            question_f.append(q)
            pbar.update(1)
    if False:
        q_empty = []
        for item in question_f:
            if len(item['question']) == 0:
                q_empty.append(item)
        if len(q_empty) != 0:
            raise Exception('Empty in q after filtering, have {} empty q'.format(len(q_empty)))

    return question_f


def filter_q_by_word_freq_dict(dfs_q_filters, train=None):
    question_f_dict = {}
    with tqdm(total=len(train)) as pbar:
        pbar.set_description('filter question by q_dfs')
        for item in train:
            q_cut = pynlpir.segment(item.get('question'), pos_tagging=False)
            question = [word for word in q_cut if word in dfs_q_filters]
            if len(question) == 0:
                isBad = True
            else:
                isBad = False
            q = {
                'question': question,
                'isBad': isBad,
                'docid': item['docid'],
                'id': item['id'],
                }
            question_f_dict[item['id']] = q
            pbar.update(1)

    return question_f_dict


