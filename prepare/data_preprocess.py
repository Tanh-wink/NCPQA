import pandas as pd
import copy
import pickle
import re
from tqdm import tqdm
import jieba

def del_repeat(raw_data):
    """
    去重
    :param raw_data:
    :param save_path:
    :return:
    """
    repeat_data = []
    unique_data = []
    with tqdm(total=len(raw_data), desc='del repeat data') as pbar:
        for idx, data in enumerate(raw_data):
            if data in unique_data:
                repeat_data.append(data)
            else:
                unique_data.append(data)
            pbar.update(1)

    return unique_data, repeat_data


def chinaPuncSubEngPunc(data):
    '''
        常见的英文标点符号 替换为 对应的中文标点符号

        return dataset, subSample

        Bug:
            小数点的 '.' 应该不可以被替换为 '。'

    '''
    data = copy.deepcopy(data)
    usualChinaPunc = ['，', '。', '：', '；', '？', '！', '（', '）', '【', '】']
    usualEngPunc = [',', '.', ':', ';', '?', '!', '(', ')', '[', ']']
    with tqdm(total=len(data), desc='Eng puncts to Chinese puncts') as pbar:
        for index in range(len(data)):
            for curEngIndex, curEngPunc in enumerate(usualEngPunc):
                data[index][1] = re.sub('[{}]'.format(curEngPunc), '{}'.format(usualChinaPunc[curEngIndex]),
                                        data[index][1])
            pbar.update(1)
    return data


def del_stopwords(data):
    '''
        去除停用词
        data： a list of data

        stopwords: a list of all the stopwords

    '''
    data = copy.deepcopy(data)

    for index in range(len(data)):
        data[index][1].strip()
        data[index][1] = list(jieba.cut(data[index][1]))
        for word in data[index][1]:
            if word in stopwords:
                data[index][1].remove(word)
        data[index][1] = ''.join(data[index][1])
    return data

def clean_text(input_data):
    """
    清除非法字符
    :param input_data:
    :return:
    """
    illegal_puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=',
                  '#', '*', '+', '\\', '•',  '~', '@', '£', '“', '”', '：', '）', '（', '-', '！', '？', '|', '¬',
                  '；', '￥','·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←','【','】',
                  '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '░',
                  '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', '▒', '：', '¼', '⊕', '▼', '▪',
                  '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞',
                  '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤',
                  'ï']
    data = copy.deepcopy(input_data)
    with tqdm(total=len(data), desc='del illegal puncts') as pbar:
        for index in range(len(data)):
            for punct in illegal_puncts:
                data[index][1] = data[index][1].replace(punct, '')
            pbar.update(1)
    return data

def data_preprocess(data):

    data = del_stopwords(data)
    data = clean_text(data)
    # data = clean_text(data)
    return data


if __name__ == '__main__':
    raw_data = pd.read_csv("./data/train.csv").values.tolist()
    data = data_preprocess(raw_data, is_train=True)
    with open('./data/train_data_preprocess.pkl', 'wb') as fout:
        pickle.dump(data, fout)
        print('saved the preprocessed data into ./data/train_data_preprocess.pkl')
