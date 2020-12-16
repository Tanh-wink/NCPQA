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
from gensim.summarization.bm25 import BM25
import pynlpir
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

    def __init__(self, unique_id, qid, context_index, doc_span_index, token_to_orig_map, token_is_max_context,
                 tokens, input_ids, input_mask, segment_ids, docid, answer, start_idx, end_idx):
        self.unique_id = unique_id,
        self.qid = qid
        self.context_index = context_index
        self.doc_span_index = doc_span_index,
        self.tokens = tokens,
        self.token_to_orig_map = token_to_orig_map,
        self.token_is_max_context = token_is_max_context,
        self.input_ids=input_ids
        self.input_mask=input_mask
        self.segment_ids=segment_ids
        self.docid=docid
        self.answer=answer
        self.start_idx=start_idx
        self.end_idx=end_idx



class Data_processor_bm25(object):
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
                    contexts = []
                    docids = []
                    for docid in data['contexts']:
                        context = self.id2doc[docid]
                        docids.append(docid)
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

    def _convert_examples_to_features(self, examples, doc_stride, is_train=True, topRate=0.5):
        """Loads a data file into a list of `InputBatch`s."""
        unique_id = 1000000000
        features = []
        pynlpir.open()
        with tqdm(total=len(examples), desc="convert examples to features add bm25:") as pbar:
            for example_id, example in enumerate(examples):
                qid = example.qid
                qusetion = example.qusetion
                docids = example.docids
                contexts = example.contexts
                answer = None
                answer_span = None

                features_temp = []
                sub_doc = []
                for context_index, (docid, context) in enumerate(zip(docids, contexts)):
                    if is_train:
                        answer_span = example.answer_span

                    qusetion_tokens = self.tokenizer.tokenize(qusetion) if len(qusetion) > 0 else []
                    if len(qusetion_tokens) > self.max_query_length: # cut at tail
                        qusetion_tokens = qusetion_tokens[0:self.max_query_length]

                    token_to_orig_index = []
                    orig_to_token_index = []
                    context_tokens = []
                    for (i, word) in enumerate(context):
                        orig_to_token_index.append(len(context_tokens))
                        sub_tokens = self.tokenizer.tokenize(word)
                        for sub_token in sub_tokens:
                            token_to_orig_index.append(i)
                            context_tokens.append(sub_token)

                    token_start_position = None
                    token_end_position = None
                    if is_train:
                        token_start_position = -1
                        token_end_position = -1
                    if is_train:
                        token_start_position = orig_to_token_index[answer_span[0]]
                        if answer_span[1] < len(context) - 1:
                            token_end_position = orig_to_token_index[answer_span[1] + 1] - 1
                        else:
                            token_end_position = len(context_tokens) - 1
                        (token_start_position, token_end_position) = self._improve_answer_span(
                            context_tokens, token_start_position, token_end_position, example.answer)

                    # We can have documents that are longer than the maximum sequence length.
                    # To deal with this we do a sliding window approach, where we take chunks
                    # of the up to our max length with a stride of `doc_stride`.
                    max_context_length = self.max_seq_length - self.max_query_length - 3
                    _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
                        "DocSpan", ["start", "length"])
                    doc_spans = []
                    start_offset = 0
                    while start_offset < len(context_tokens):
                        length = len(context_tokens) - start_offset
                        if length > max_context_length:
                            length = max_context_length
                        doc_spans.append(_DocSpan(start=start_offset, length=length))
                        if start_offset + length == len(context_tokens):
                            break
                        start_offset += min(length, doc_stride)


                    for (doc_span_index, doc_span) in enumerate(doc_spans):
                        token_to_orig_map = {}
                        token_is_max_context = {}
                        tokens = ["[CLS]"] + qusetion_tokens + ["[SEP]"]
                        segment_ids = [0] * len(tokens)
                        for i in range(doc_span.length):
                            split_token_index = doc_span.start + i
                            token_to_orig_map[len(tokens)] = token_to_orig_index[split_token_index]

                            is_max_context = self._check_is_max_context(doc_spans, doc_span_index,
                                                                   split_token_index)
                            token_is_max_context[len(tokens)] = is_max_context
                            tokens.append(context_tokens[split_token_index])
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
                            # we throw it out, since there is nothing to predict.
                            doc_start = doc_span.start
                            doc_end = doc_span.start + doc_span.length - 1
                            out_of_span = False
                            if not (token_start_position >= doc_start and
                                    token_end_position <= doc_end):
                                out_of_span = True
                            if not out_of_span:
                                doc_offset = len(qusetion_tokens) + 2
                                start_position = token_start_position - doc_start + doc_offset
                                end_position = token_end_position - doc_start + doc_offset
                            else:
                                continue

                        features_temp.append(InputFeatures(
                            unique_id=unique_id,
                            qid=qid,
                            context_index=context_index,
                            doc_span_index=doc_span_index,
                            tokens=tokens,
                            token_to_orig_map=token_to_orig_map,
                            token_is_max_context=token_is_max_context,
                            input_ids=input_ids,
                            input_mask=input_mask,
                            segment_ids=segment_ids,
                            docid=docid,
                            answer=answer,
                            start_idx=start_position,
                            end_idx=end_position
                        ))
                        if not is_train:
                            sub_doc.append(pynlpir.segment(''.join(tokens), pos_tagging=False))
                        unique_id += 1
                try:
                    if not is_train:
                        bm25_model = BM25(sub_doc)
                        bm25_score = bm25_model.get_scores(pynlpir.segment(qusetion, False))
                        rankindex = np.argsort(-np.array(bm25_score))
                        features.extend([features_temp[i] for i in rankindex[:int(len(rankindex)*topRate)]])
                    else:
                        features.extend(features_temp)
                except:
                    print('end')
                pbar.update(1)
        pynlpir.close()
        return features

    def get_train_features(self, train_examples, train_features_file, doc_stride):

        if os.path.exists(train_features_file):
            with open(train_features_file, "rb") as reader:
                train_features = pickle.load(reader)
        else:
            train_features = self._convert_examples_to_features(
                examples=train_examples,
                doc_stride=doc_stride,
                is_train=True
            )
            logger.info("  Saving train features into file %s", train_features_file)
            with open(train_features_file, "wb") as writer:
                pickle.dump(train_features, writer)
        logger.info("Num train features:{}".format(len(train_features)))
        return train_features

    def get_valid_features(self, valid_examples, valid_features_file, doc_stride):

        if os.path.exists(valid_features_file):
            with open(valid_features_file, "rb") as reader:
                valid_features = pickle.load(reader)
        else:
            valid_features = self._convert_examples_to_features(
                examples=valid_examples,
                doc_stride=doc_stride,
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
                doc_stride=doc_stride,
                is_train=False
            )
            logger.info("  Saving train features into file %s", pred_features_file)
            with open(pred_features_file, "wb") as writer:
                pickle.dump(pred_features, writer)

        return pred_features

    def prepare_train_dataloader(self,  train_features, train_batch_size, local_rank):
        all_unique_ids = torch.tensor([f.unique_id for f in train_features], dtype=torch.long)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_start_idxs = torch.tensor([f.start_idx for f in train_features], dtype=torch.long)
        all_end_idxs = torch.tensor([f.end_idx for f in train_features], dtype=torch.long)

        train_data = TensorDataset(all_unique_ids, all_input_ids, all_input_mask, all_segment_ids, all_start_idxs, all_end_idxs)
        if local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=train_batch_size, drop_last=False)
        return train_dataloader

    def prepare_pred_dataloader(self, predict_file, predict_batch_size, doc_stride, cache_dir=None, topRate=0.5):
        pred_examples = self.read_QA_examples(
            predict_file, is_train=False)
        # eval_examples=eval_examples[:100]

        logger.info("***** Running predictions *****")
        logger.info("  Num predict examples = %d", len(pred_examples))
        logger.info("  Predict batch size = %d", predict_batch_size)

        example_n = len(pred_examples)
        indices = np.arange(example_n)
        for batch_start in np.arange(0, example_n, 200):

            cache_file = os.path.join(cache_dir, 'pred_features_{}.pkl'.format(batch_start))

            if os.path.exists(cache_file):
                t0 = time.time()
                with open(cache_file, 'rb') as fp:
                    pred_features = pickle.loads(fp.read())
                t1 = time.time()
                batch_indices = indices[batch_start: batch_start + 200]
                example_batch = []
                for batch_indice in batch_indices:
                    example_batch.append(pred_examples[batch_indice])
                print('cache: predict_features --> {} loaded, cost time: {}s'.format(cache_file, t1 - t0))
            else:
                batch_indices = indices[batch_start: batch_start + 200]
                example_batch = []
                for batch_indice in batch_indices:
                    example_batch.append(pred_examples[batch_indice])
                pred_features = self._convert_examples_to_features(
                    examples=example_batch,
                    doc_stride=doc_stride,
                    is_train=False,
                    topRate=topRate,
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

    def _improve_answer_span(self, doc_tokens, input_start, input_end, orig_answer_text):
        """Returns tokenized answer spans that better match the annotated answer."""

        # The SQuAD annotations are character based. We first project them to
        # whitespace-tokenized words. But then after WordPiece tokenization, we can
        # often find a "better match". For example:
        #
        #   Question: What year was John Smith born?
        #   Context: The leader was John Smith (1895-1943).
        #   Answer: 1895
        #
        # The original whitespace-tokenized answer will be "(1895-1943).". However
        # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
        # the exact answer, 1895.
        #
        # However, this is not always possible. Consider the following:
        #
        #   Question: What country is the top exporter of electornics?
        #   Context: The Japanese electronics industry is the lagest in the world.
        #   Answer: Japan
        #
        # In this case, the annotator chose "Japan" as a character sub-span of
        # the word "Japanese". Since our WordPiece tokenizer does not split
        # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
        # in SQuAD, but does happen.
        tok_answer_text = " ".join(self.tokenizer.tokenize(orig_answer_text))

        for new_start in range(input_start, input_end + 1):
            for new_end in range(input_end, new_start - 1, -1):
                text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
                if text_span == tok_answer_text:
                    return (new_start, new_end)

        return (input_start, input_end)

    def _check_is_max_context(self, doc_spans, cur_span_index, position):
        """Check if this is the 'max context' doc span for the token."""

        # Because of the sliding window approach taken to scoring documents, a single
        # token can appear in multiple documents. E.g.
        #  Doc: the man went to the store and bought a gallon of milk
        #  Span A: the man went to the
        #  Span B: to the store and bought
        #  Span C: and bought a gallon of
        #  ...
        #
        # Now the word 'bought' will have two scores from spans B and C. We only
        # want to consider the score with "maximum context", which we define as
        # the *minimum* of its left and right context (the *sum* of left and
        # right context will always be the same, of course).
        #
        # In the example the maximum context for 'bought' would be span C since
        # it has 1 left context and 3 right context, while span B has 4 left context
        # and 0 right context.
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


