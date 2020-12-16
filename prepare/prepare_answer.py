import pandas as pd
import numpy as np
import collections
from prepare.data_preprocess import data_preprocess
from torch.utils.data.distributed import DistributedSampler
from sklearn.model_selection import KFold
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch
import logging
import os
import pickle
from tqdm import tqdm
import time

logger = logging.getLogger(__name__)


class QAExample(object):
    def __init__(self, example_id=None, qid=None, qusetion=None, docid=None, answer=None):

        self.example_id = example_id
        self.qid = qid
        self.qusetion = qusetion
        self.docid = docid
        self.answer = answer

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qid: %s" % (self.qid)
        s += ", context: %s" % (self.contexts)
        if self.answer is not None:
            s += ", label: %s" % self.answer
        return s


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_id, qid, tokens, input_ids, input_mask, segment_ids, docid, answer,
                  label):
        self.unique_id = unique_id,
        self.qid = qid,
        self.tokens = tokens,
        self.input_ids = input_ids,
        self.input_mask = input_mask,
        self.segment_ids = segment_ids,
        self.answer = answer,
        self.docid = docid,
        self.label = label





class Data_processor(object):
    def __init__(self, tokenizer, policies_file,  max_seq_length=384, max_query_length=64):

        self.max_seq_length = max_seq_length
        self.max_query_length = max_query_length
        self.tokenizer = tokenizer
        self._read_policies_context(policies_file)

    def _read_policies_context(self, data_file):
        with open(data_file, 'rb') as fin:
            self.plicies = pickle.load(fin)
        self.id2doc = {}
        for c_data in self.plicies:
            context = c_data['text']
            self.id2doc[c_data['docid']] = context

    def build_sub_docs_dict(self, args):
        print('build_sub_docs_dict')
        with open(args.sub_docs_dict, 'rb') as fin:
            self.sub_docs_dict = pickle.load(fin)


    def _train_data_preprocess(self, train_data):
        '''

        :param train_data:
        :return: train_datas: [dict,dict,...,dict]
        '''
        train_datas = []
        for c_data in train_data:
            answer = c_data['answer']
            docid = c_data['docid']
            c_data['contexts'] = [docid]
            context = self.id2doc[docid]
            start_idx = context.find(answer)
            if start_idx != -1:
                end_idx = start_idx + len(answer) - 1
                c_data['answer_span'] = [start_idx, end_idx]
                c_data['context'] = context
                train_datas.append(c_data)
        return train_datas

    def read_QA_examples(self, data_file, is_train=True, k=1):
        with open(data_file, 'rb') as fin:
            data_set = pickle.load(fin)

        examples = []
        example_id = 1000000
        with tqdm(total=len(data_set), desc="reading examples:") as pbar:
            for qid, data in data_set.items():
                if len(data) > k:
                    data = data[:k]
                for d in data:
                    docid = d['answer_docid']
                    qusetion = d['question']
                    answer = d['text']

                    if is_train:
                        examples.append(QAExample(
                            example_id=example_id,
                            qid=qid,
                            qusetion=qusetion,
                            docid=docid,
                            answer=answer
                        ))
                        example_id += 1
                    else:
                        examples.append(QAExample(
                            example_id=example_id,
                            qid=qid,
                            qusetion=qusetion,
                            docid=docid,
                            answer=answer
                        ))
                        example_id += 1
                    pbar.update(1)
        return examples


    def get_train_examples(self, train_file, train_examples_file):
        if os.path.exists(train_examples_file):
            train_examples = pickle.load(open(train_examples_file, mode='rb'))
        else:
            train_examples = self.read_QA_examples(train_file, is_train=True)
            pickle.dump(train_examples, open(train_examples_file, mode='wb'))

        np.random.shuffle(train_examples)  # shuffle data
        return train_examples

    def _convert_examples_to_features(self, examples, is_train=True):
        """Loads a data file into a list of `InputBatch`s."""
        unique_id = 1000000000
        features = []
        with tqdm(total=len(examples), desc="convert examples to features:") as pbar:
            for example_id, example in enumerate(examples):
                qid = example.qid
                qusetion = example.qusetion
                docid = example.docid
                answer = example.answer
                label = None
                if is_train:
                    label = example.label

                qusetion_tokens = self.tokenizer.tokenize(qusetion) if len(qusetion) > 0 else []
                if len(qusetion_tokens) > self.max_query_length: # cut at tail
                    qusetion_tokens = qusetion_tokens[0:self.max_query_length]
                tokens = ["[CLS]"] + qusetion_tokens + ["[SEP]"]
                segment_ids = [0] * len(tokens)
                input_mask = [1] * len(tokens)
                pads = ["[PAD]"] * (self.max_query_length - len(tokens))
                tokens += pads
                input_mask += [0] * len(pads)
                segment_ids += [0] * len(pads)

                max_answer_length = self.max_seq_length - self.max_query_length - 3
                answer_tokens = self.tokenizer.tokenize(answer) if len(answer) > 0 else []
                if len(answer_tokens) > max_answer_length:  # cut at tail
                    answer_tokens = answer_tokens[0:max_answer_length]
                tokens += answer_tokens + ["[SEP]"]
                segment_ids += [1] * len(answer_tokens + ["[SEP]"])
                input_mask += [1] * len(answer_tokens + ["[SEP]"])



                input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

                padding = [0] * (self.max_seq_length - len(input_ids))
                input_ids += padding
                input_mask += padding
                segment_ids += padding

                assert len(input_ids) == self.max_seq_length
                assert len(input_mask) == self.max_seq_length
                assert len(segment_ids) == self.max_seq_length

                features.append(InputFeatures(
                    unique_id=unique_id,
                    qid=qid,
                    tokens=tokens,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    answer=answer,
                    docid=docid,
                    label=label
                ))
                unique_id += 1
                pbar.update(1)
        return features


    def get_train_features(self, train_examples, train_features_file):

        if os.path.exists(train_features_file):
            with open(train_features_file, "rb") as reader:
                train_features = pickle.load(reader)
        else:
            train_features = self._convert_examples_to_features(
                examples=train_examples,
                is_train=True
            )
            logger.info("  Saving train features into file %s", train_features_file)
            with open(train_features_file, "wb") as writer:
                pickle.dump(train_features, writer)
        logger.info("Num train features:{}".format(len(train_features)))
        return train_features

    def get_valid_features(self, valid_examples, valid_features_file):

        if os.path.exists(valid_features_file):
            with open(valid_features_file, "rb") as reader:
                valid_features = pickle.load(reader)
        else:
            valid_features = self._convert_examples_to_features(
                examples=valid_examples,
                is_train=True
            )
            logger.info("  Saving valid features into file %s", valid_features_file)
            with open(valid_features_file, "wb") as writer:
                pickle.dump(valid_features, writer)
        logger.info("Num valid features:{}".format(len(valid_features)))
        return valid_features

    def get_pred_features(self, pred_examples, pred_features_file, doc_stride):
        if os.path.exists(pred_features_file):
            with open(pred_features_file, "rb") as reader:
                pred_features = pickle.load(reader)
        else:
            pred_features = self._convert_examples_to_features(
                examples=pred_examples,
                is_train=False
            )
            logger.info("  Saving train features into file %s", pred_features_file)
            with open(pred_features_file, "wb") as writer:
                pickle.dump(pred_features, writer)

        return pred_features

    def prepare_train_dataloader(self,  train_features, train_batch_size, local_rank, union):
        all_unique_ids = torch.tensor([f.unique_id for f in train_features], dtype=torch.long)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_labels = torch.tensor([f.label for f in train_features], dtype=torch.long)

        train_data = TensorDataset(all_unique_ids, all_input_ids, all_input_mask, all_segment_ids, all_labels)
        if local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=train_batch_size, drop_last=False)
        return train_dataloader

    def prepare_pred_dataloader(self, predict_file, predict_batch_size, k=1, cache_dir=None):
        pred_examples = self.read_QA_examples(
            predict_file, is_train=False, k=k)
        # eval_examples=eval_examples[:100]

        logger.info("***** Running predictions *****")
        logger.info("  Num predict examples = %d", len(pred_examples))
        logger.info("  Predict batch size = %d", predict_batch_size)

        cache_file = os.path.join(cache_dir, 'pred_features.pkl')
        if os.path.exists(cache_file):
            t0 = time.time()
            with open(cache_file, 'rb') as fp:
                pred_features = pickle.loads(fp.read())
            t1 = time.time()
            print('cache: predict_features --> {} loaded, cost time: {}s'.format(cache_file, t1-t0))
        else:
            pred_features = self._convert_examples_to_features(
                examples=pred_examples,
                is_train=False
            )
            with open(cache_file, 'wb') as fp:
                fp.write(pickle.dumps(pred_features))

        logger.info("  Num batch predict features = %d", len(pred_features))

        all_unique_id = torch.tensor([f.unique_id[0] for f in pred_features], dtype=torch.long)
        all_input_ids = torch.tensor([f.input_ids[0] for f in pred_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask[0] for f in pred_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids[0] for f in pred_features], dtype=torch.long)

        pred_data = TensorDataset(all_unique_id, all_input_ids, all_input_mask, all_segment_ids)
        # Run prediction for full data
        pred_sampler = SequentialSampler(pred_data)
        pred_dataloader = DataLoader(pred_data, sampler=pred_sampler, batch_size=predict_batch_size)

        return pred_dataloader, pred_features



