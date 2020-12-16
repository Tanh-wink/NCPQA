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
    def __init__(self, qid=None, qusetion=None, docids=None, contexts=None, answer=None, answer_span=None):

        self.qid = qid
        self.qusetion = qusetion
        self.docids = docids
        self.contexts = contexts
        self.answer = answer
        self.answer_span = answer_span

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

    def __init__(self, unique_id, qid, context_index, token_to_orig_map, doc_span,
                 tokens, input_ids, input_mask, segment_ids, docid, answer, start_idx, end_idx, label):
        self.unique_id = unique_id,
        self.qid = qid
        self.context_index = context_index
        self.tokens = tokens,
        self.token_to_orig_map = token_to_orig_map,
        self.doc_span = doc_span,
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.docid = docid
        self.answer = answer
        self.start_idx = start_idx
        self.end_idx = end_idx
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

    def read_QA_examples(self, data_file, is_train=True):
        with open(data_file, 'rb') as fin:
            data_set = pickle.load(fin)
        if is_train:
            data_set = self._train_data_preprocess(data_set)
        examples = []
        with tqdm(total=len(data_set), desc="reading examples:") as pbar:
            for index, data in enumerate(data_set):
                qid = data['qid']
                qusetion = data['question']
                contexts = []
                docids = []
                for docid in data['contexts']:
                    context = self.id2doc[docid]
                    docids.append(docid)
                    contexts.append(context)
                if is_train:
                    examples.append(QAExample(
                        qid=qid,
                        qusetion=qusetion,
                        docids=docids,
                        contexts=contexts,
                        answer=data['answer'],
                        answer_span=data['answer_span']
                    ))
                else:
                    examples.append(QAExample(
                        qid=qid,
                        qusetion=qusetion,
                        docids=docids,
                        contexts=contexts
                    ))
                pbar.update(1)
        return examples

    def read_QA_examples_sub_doc(self, data_file):
        with open(data_file, 'rb') as fin:
            data_set = pickle.load(fin)
        examples = []
        with tqdm(total=len(data_set), desc="reading examples:") as pbar:
            for index, data in enumerate(data_set):
                qid = data['qid']
                qusetion = data['question']
                contexts = []
                docids = []
                for sub_doc_id in data['sub_docs_id']:
                    context = self.sub_docs_dict[sub_doc_id]
                    docids.append(sub_doc_id)
                    contexts.append(context)

                examples.append(QAExample(
                    qid=qid,
                    qusetion=qusetion,
                    docids=docids,
                    contexts=contexts
                ))
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

    def _convert_examples_to_features(self, examples, sentence_stride, is_train=True, addBad=None):
        """Loads a data file into a list of `InputBatch`s."""
        unique_id = 1000000000
        features = []
        total_count = 0
        bad_count = 0
        with tqdm(total=len(examples), desc="convert examples to features:") as pbar:
            for example_id, example in enumerate(examples):
                qid = example.qid
                qusetion = example.qusetion
                docids = example.docids
                contexts = example.contexts
                answer = None
                answer_span = None
                label = None
                for context_index, (docid, context) in enumerate(zip(docids, contexts)):
                    if is_train:
                        answer = example.answer

                    qusetion_tokens = self.tokenizer.tokenize(qusetion) if len(qusetion) > 0 else []
                    if len(qusetion_tokens) > self.max_query_length: # cut at tail
                        qusetion_tokens = qusetion_tokens[0:self.max_query_length]
                    max_context_length = self.max_seq_length - self.max_query_length - 3
                    doc_spans = self._cut_doc(context, max_context_length, sentence_stride)

                    for doc_span in doc_spans:
                        orig_to_token_index = []
                        token_to_orig_index = []
                        if is_train:
                            start, end = self._check_answer(doc_span, answer)
                        doc_tokens = []
                        for (i, word) in enumerate(doc_span):
                            orig_to_token_index.append(len(doc_tokens))
                            sub_tokens = self.tokenizer.tokenize(word)
                            for sub_token in sub_tokens:
                                token_to_orig_index.append(i)
                                doc_tokens.append(sub_token)

                        token_start_position = None
                        token_end_position = None
                        if len(doc_tokens) > max_context_length:
                            doc_tokens = doc_tokens[:max_context_length]
                        if is_train:
                            if start != -1 and end != -1:
                                token_start_position = orig_to_token_index[start]
                                token_end_position = orig_to_token_index[end]
                                if token_end_position > len(doc_tokens) - 1:
                                    token_end_position = len(doc_tokens) - 1
                                label = 1
                            else:
                                label = 0
                                token_start_position = 0
                                token_end_position = 0
                        tokens = ["[CLS]"] + qusetion_tokens + ["[SEP]"]
                        segment_ids = [0] * len(tokens)

                        token_to_orig_map = {}

                        for i in range(len(doc_tokens)):
                            token_to_orig_map[len(tokens)] = token_to_orig_index[i]

                            tokens.append(doc_tokens[i])
                            segment_ids.append(1)
                        tokens.append("[SEP]")
                        segment_ids.append(1)

                        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                        input_mask = [1] * len(tokens)

                        padding = [0] * (self.max_seq_length - len(input_ids))
                        input_ids += padding
                        input_mask += padding
                        segment_ids += padding

                        assert len(input_ids) == self.max_seq_length
                        assert len(input_mask) == self.max_seq_length
                        assert len(segment_ids) == self.max_seq_length

                        start_position = None
                        end_position = None
                        if is_train:
                            # For training, if our document chunk does not contain an annotation

                            doc_offset = len(qusetion_tokens) + 2
                            start_position = token_start_position + doc_offset
                            end_position = token_end_position + doc_offset


                        features.append(InputFeatures(
                            unique_id=unique_id,
                            qid=qid,
                            context_index=context_index,
                            tokens=tokens,
                            doc_span=doc_span,
                            token_to_orig_map=token_to_orig_map,
                            input_ids=input_ids,
                            input_mask=input_mask,
                            segment_ids=segment_ids,
                            docid=docid,
                            answer=answer,
                            start_idx=start_position,
                            end_idx=end_position,
                            label=label
                        ))
                        unique_id += 1
                pbar.update(1)
        print('total_count {}, bad_count {}'.format(total_count, bad_count))
        return features


    def get_train_features(self, train_examples, train_features_file, sentence_stride, addBad=None):

        if os.path.exists(train_features_file):
            with open(train_features_file, "rb") as reader:
                train_features = pickle.load(reader)
        else:
            train_features = self._convert_examples_to_features(
                examples=train_examples,
                sentence_stride=sentence_stride,
                is_train=True,
                addBad=addBad,
            )
            logger.info("  Saving train features into file %s", train_features_file)
            with open(train_features_file, "wb") as writer:
                pickle.dump(train_features, writer)
        logger.info("Num train features:{}".format(len(train_features)))
        return train_features

    def get_valid_features(self, valid_examples, valid_features_file, sentence_stride, addBad=None):

        if os.path.exists(valid_features_file):
            with open(valid_features_file, "rb") as reader:
                valid_features = pickle.load(reader)
        else:
            valid_features = self._convert_examples_to_features(
                examples=valid_examples,
                sentence_stride=sentence_stride,
                is_train=True,
                addBad=addBad,
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
                sentence_stride=sentence_stride,
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
        all_start_idxs = torch.tensor([f.start_idx for f in train_features], dtype=torch.long)
        all_end_idxs = torch.tensor([f.end_idx for f in train_features], dtype=torch.long)
        if union:
            all_labels = torch.tensor([f.label for f in train_features], dtype=torch.long)
            train_data = TensorDataset(all_unique_ids, all_input_ids, all_input_mask, all_segment_ids, all_start_idxs, all_end_idxs, all_labels)
        else:
            train_data = TensorDataset(all_unique_ids, all_input_ids, all_input_mask, all_segment_ids, all_start_idxs,
                                       all_end_idxs)
        if local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=train_batch_size, drop_last=False)
        return train_dataloader

    def prepare_pred_dataloader(self, predict_file, predict_batch_size, doc_stride, cache_dir=None):
        pred_examples = self.read_QA_examples(
            predict_file, is_train=False)
        # eval_examples=eval_examples[:100]

        logger.info("***** Running predictions *****")
        logger.info("  Num predict examples = %d", len(pred_examples))
        logger.info("  Predict batch size = %d", predict_batch_size)

        example_n = len(pred_examples)
        indices = np.arange(example_n)
        for batch_start in np.arange(0, example_n, 400):
            cache_file = os.path.join(cache_dir, 'pred_features_{}.pkl'.format(batch_start))
            if os.path.exists(cache_file):
                t0 = time.time()
                with open(cache_file, 'rb') as fp:
                    pred_features = pickle.loads(fp.read())
                t1 = time.time()
                batch_indices = indices[batch_start: batch_start + 400]
                example_batch = []
                for batch_indice in batch_indices:
                    example_batch.append(pred_examples[batch_indice])
                print('cache: predict_features --> {} loaded, cost time: {}s'.format(cache_file, t1-t0))
            else:
                batch_indices = indices[batch_start: batch_start + 400]
                example_batch = []
                for batch_indice in batch_indices:
                    example_batch.append(pred_examples[batch_indice])
                pred_features = self._convert_examples_to_features(
                    examples=example_batch,
                    doc_stride=doc_stride,
                    is_train=False
                )
                with open(cache_file, 'wb') as fp:
                    fp.write(pickle.dumps(pred_features))


            logger.info("  Num batch predict features = %d", len(pred_features))

            all_unique_id = torch.tensor([f.unique_id for f in pred_features], dtype=torch.long)
            all_input_ids = torch.tensor([f.input_ids for f in pred_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in pred_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in pred_features], dtype=torch.long)

            pred_data = TensorDataset(all_unique_id, all_input_ids, all_input_mask, all_segment_ids)
            # Run prediction for full data
            pred_sampler = SequentialSampler(pred_data)
            pred_dataloader = DataLoader(pred_data, sampler=pred_sampler, batch_size=predict_batch_size)


            yield pred_dataloader, pred_features, example_batch

    def _cut_doc(self, context, doc_len, sentence_stride):
        sentences = context.strip().split(" ")

        doc_spans = []

        doc_span = ""
        while sentences:
            sentence = sentences.pop(0)
            if len(sentence) <= doc_len:
                if len(doc_span) + len(sentence) <= doc_len:
                    doc_span += sentence
                else:
                    doc_spans.append(doc_span)
                    sentences.insert(0, sentence)
                    doc_span = ""
            else:
                if doc_span != "":
                    doc_spans.append(doc_span)
                    doc_span = ""
                else:
                    sentence = sentence[:doc_len]
                    doc_spans.append(sentence)
        return doc_spans
        # start_offset = 0
        # while start_offset < len(sentences_tokens):
        #     length = len(sentences_tokens) - start_offset
        #     if length > doc_len:
        #         length = doc_len
        #     doc_spans.append(_DocSpan(start=start_offset, length=length))
        #     if start_offset + length == len(context_tokens):
        #         break
        #     start_offset += min(length, sentence_stride)



    def _improve_answer_span(self, doc_tokens, input_start, input_end, orig_answer_text):
        """Returns tokenized answer spans that better match the annotated answer."""

        tok_answer_text = " ".join(self.tokenizer.tokenize(orig_answer_text))

        for new_start in range(input_start, input_end + 1):
            for new_end in range(input_end, new_start - 1, -1):
                text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
                if text_span == tok_answer_text:
                    return (new_start, new_end)

        return (input_start, input_end)

    def _check_is_max_context(self, doc_spans, cur_span_index, position):
        """Check if this is the 'max context' doc span for the token."""


        best_score = None
        best_span_index = None
        for (span_index, doc_span) in enumerate(doc_spans):
            end = doc_span.start + doc_span.length - 1
            if position < doc_span.start:
                continue
            if position > end:
                continue
            num_left_context = position - doc_span.start
            num_right_context = end - position
            score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
            if best_score is None or score > best_score:
                best_score = score
                best_span_index = span_index

        return cur_span_index == best_span_index

    def _check_answer(self, doc_span, answer):

        start = doc_span.find(answer)
        if start != -1:
            end = start + len(answer) - 1
            if doc_span[start:end + 1] == answer:
                return start, end
            else:
                return -1, -1
        else:
            return -1, -1


