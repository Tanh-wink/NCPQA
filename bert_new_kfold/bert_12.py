# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run BERT on SQuAD."""

# from __future__ import absolute_import
# from __future__ import division
from __future__ import print_function

import logging
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2,1,0'

import random
import pickle
import json
import math
import six
import collections
from tqdm import tqdm
# from rouge import Rouge
from NCPQA.utils.rouge import Rouge

import sys
sys.path.append('../')

import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import gc

from bert_optimization import BertAdam

from torch.utils.tensorboard import SummaryWriter

# from pytorchtools import EarlyStopping
from bert_12_config import getArgs
from prepare.prepare_fold_data import Data_processor, InputFeatures
from prepare.prepare_multi_answer import Data_processor_multi
import time
from sklearn.model_selection import KFold

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AlbertModel,
    AlbertPreTrainedModel,
    AlbertConfig,
    AlbertForQuestionAnswering,
    AlbertTokenizer,
    BertConfig,
    BertForQuestionAnswering,
    BertTokenizer,
    DistilBertConfig,
    DistilBertForQuestionAnswering,
    DistilBertTokenizer,
    XLMConfig,
    XLMForQuestionAnswering,
    XLMTokenizer,
    XLNetConfig,
    XLNetForQuestionAnswering,
    XLNetTokenizer,
    RobertaConfig,
)
MODEL_CLASSES = {
    "bert": (BertConfig, BertForQuestionAnswering, BertTokenizer),
    "xlnet": (XLNetConfig, XLNetForQuestionAnswering, XLNetTokenizer),
    "xlm": (XLMConfig, XLMForQuestionAnswering, XLMTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForQuestionAnswering, DistilBertTokenizer),
    "albert": (AlbertConfig, AlbertForQuestionAnswering, AlbertTokenizer),
    "roberta": (RobertaConfig, BertForQuestionAnswering, BertTokenizer)
}


class AlbertForQuestionAnswer(AlbertPreTrainedModel):

    def __init__(self, config):
        super(AlbertForQuestionAnswer, self).__init__(config)
        self.num_labels = config.num_labels

        self.albert = AlbertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None,
                start_positions=None, end_positions=None, labels=None):

        outputs = self.albert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        sequence_output = outputs[0]
        pooled_output = outputs[1]

        # classification
        pooled_output = self.dropout(pooled_output)
        classifier_logits = self.classifier(pooled_output)

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        outputs = (start_logits, end_logits, classifier_logits) + outputs[2:]
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)

            if labels is not None:
                loss_fct = CrossEntropyLoss(ignore_index=-1)
                class_loss = loss_fct(classifier_logits, labels)
                total_loss = 0.4 * start_loss + 0.4 * end_loss + 0.1 * class_loss
            else:
                total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

start_time = time.strftime("%Y-%m-%d@%H_%M_%S", time.localtime())

# get args
args = getArgs()

model_save_path = os.path.join(args.output_dir, args.model_type+'_'+args.model_info+'_{}fold'.format(args.kfold))

if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

def save_config_file():
    with open('./bert_12_config.py', 'r', encoding='utf-8') as fp1:
        with open(model_save_path+'/bert_12_config.py', 'w', encoding='utf-8') as fp2:
            fp2.write(fp1.read())

def load_state_dict(model, init_checkpoint):
    logger.info('load bert weight from {}'.format(init_checkpoint))
    state_dict = torch.load(init_checkpoint, map_location='cpu')
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    # new_state_dict=state_dict.copy()
    # for kye ,value in state_dict.items():
    #     new_state_dict[kye.replace("bert","c_bert")]=value
    # state_dict=new_state_dict
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})

        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            # logger.info("name {} chile {}".format(name,child))
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix='' if hasattr(model, 'bert') else 'bert.')
    logger.info("missing keys:{}".format(missing_keys))
    logger.info('unexpected keys:{}'.format(unexpected_keys))
    logger.info('error msgs:{}'.format(error_msgs))

    return model


def prepare_summary_writer(args, mode='train', fold=None):
    if fold is None:
        summary_dir = os.path.join('/'.join(args.output_dir.split('/')[:-1]), 'tensorboard', args.model_type+'_'+args.model_info,
                                   'log_{}'.format(start_time), mode)
    else:
        summary_dir = os.path.join('/'.join(args.output_dir.split('/')[:-1]), 'tensorboard',
                                   args.model_type + '_' + args.model_info + '_{}fold'.format(fold[0]),
                                   'log_{}'.format(start_time), mode+'_fold{}'.format(fold[1]))
    writer = SummaryWriter(summary_dir)
    return writer

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x

def prepare_optimizer(args, model, num_train_steps):
    '''
    prepare optimizer
    :param args:
    :param model:
    :param num_train_steps:
    :return:
    '''
    param_optimizer = list(model.named_parameters())

    # hack to remove pooler, which is not used
    # thus it produce None grad that break apex
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    t_total = num_train_steps
    if args.local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size()
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=t_total)
    return optimizer, t_total


def gpu_profile(args):
    '''
    set gpu
    :param args:
    :return:
    '''
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    return n_gpu, device


def seed_set(seed, n_gpu):
    '''
    set random seed of cpu and gpu
    :param seed:
    :param n_gpu:
    :return:
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

def prepare_model(args, model, device, n_gpu):
    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    return model


def main():

    # gpu profile
    n_gpu, device = gpu_profile(args)
    # set random seed
    seed_set(args.seed, n_gpu)

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    if not args.do_train and not args.do_predict:
        raise ValueError("At least one of `do_train` or `do_predict` must be True.")

    if args.do_train:
        if not args.train_file:
            raise ValueError(
                "If `do_train` is True, then `train_file` must be specified.")
    if args.do_predict:
        if not args.predict_file:
            raise ValueError(
                "If `do_predict` is True, then `predict_file` must be specified.")

    if os.path.exists(args.output_dir)==False:
        # raise ValueError("Output directory () already exists and is not empty.")
        os.makedirs(args.output_dir, exist_ok=True)

    if args.do_train:
        feature_dir = '../data/' + args.model_type + "_title"
        if os.path.exists(feature_dir) == False:
            # raise ValueError("Output directory () already exists and is not empty.")
            os.makedirs(feature_dir, exist_ok=True)

        args.model_type = args.model_type.lower()
        config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
        config = config_class.from_pretrained(
            args.model_name_or_path
        )
        vocab_file = os.path.join(args.model_name_or_path, 'vocab.txt')
        logger.info("loading the vocab file from {}".format(vocab_file))
        tokenizer = BertTokenizer(vocab_file=vocab_file, do_lower_case=args.do_lower_case)

        data_processor = Data_processor(tokenizer, args.policies_file, args.max_seq_length, args.max_query_length)

        kf = KFold(n_splits=args.kfold, shuffle=False)
        args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)
        train_examples_file = "../data/train_examples.pkl"
        total_examples = data_processor.get_train_examples(args.train_file, train_examples_file)
        total_index = [i for i in range(len(total_examples))]
        trained = []
        for cur_k, (train_index, valid_index) in enumerate(kf.split(total_index)):
            if cur_k in trained:
                continue
            # set random seed
            seed_set(args.seed+cur_k*10, n_gpu)

            logger.info("***** Running fold {} training *****".format(cur_k + 1))
            cur_k_feature_dir = os.path.join(feature_dir, '{}fold'.format(args.kfold), 'fold_{}'.format(cur_k + 1))
            if not os.path.exists(cur_k_feature_dir):
                os.makedirs(cur_k_feature_dir)

            train_examples = ''
            train_writer = prepare_summary_writer(args, mode='train', fold=(args.kfold, cur_k + 1))
            train_features_file = os.path.join(cur_k_feature_dir,
                                               'train_features_{0}_{1}.pkl'.format(str(args.max_seq_length),
                                                                                   str(args.doc_stride)))

            if os.path.exists(train_features_file):
                train_features = data_processor.get_train_features(train_examples, train_features_file,
                                                                   args.doc_stride)
            else:
                train_examples = [total_examples[index] for index in train_index]
                train_features = data_processor.get_train_features(train_examples, train_features_file,
                                                                   args.doc_stride)
                logger.info("  Num train_features = {} in fold {}".format(len(train_features), cur_k + 1))

            train_dataloader = data_processor.prepare_train_dataloader(train_features,
                                                                       args.train_batch_size, args.local_rank)
            num_train_steps = int(
                len(
                    train_features) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)
            logger.info("  Batch size = %d", args.train_batch_size)
            logger.info("  Num steps = %d", num_train_steps)
            del train_examples
            del train_features

            valid_examples = None
            valid_features = None
            valid_dataloader = None
            if args.do_valid:
                valid_examples = [total_examples[index] for index in valid_index]
                valid_writer = prepare_summary_writer(args, mode='valid', fold=(args.kfold, cur_k + 1))
                valid_features_file = os.path.join(cur_k_feature_dir,
                                                   'valid_features_{0}_{1}.pkl'.format(str(args.max_seq_length),
                                                                                       str(args.doc_stride)))
                if os.path.exists(valid_features_file):
                    valid_features = data_processor.get_valid_features(valid_examples, valid_features_file,
                                                                       args.doc_stride)
                else:
                    valid_examples = [total_examples[index] for index in valid_index]
                    valid_features = data_processor.get_valid_features(valid_examples, valid_features_file,
                                                                       args.doc_stride)
                    logger.info("  Num valid_features = {} in fold {}".format(len(valid_features), cur_k + 1))


                valid_dataloader = data_processor.prepare_train_dataloader(valid_features,
                                                                           args.train_batch_size*2, args.local_rank)

            cur_k_model_save_path = os.path.join(model_save_path, 'fold_{}'.format(cur_k + 1))
            if not os.path.exists(cur_k_model_save_path):
                os.makedirs(cur_k_model_save_path)

            if args.our_pretrain_model != '':
                model = model_class.from_pretrained(args.our_pretrain_model, config=config)
            else:
                model = model_class.from_pretrained(args.model_name_or_path, config=config)
            model = train(args, model, train_dataloader, device, num_train_steps,
                          valid_examples=valid_examples, valid_features=valid_features,
                          valid_dataloader=valid_dataloader, n_gpu=n_gpu, train_writer=train_writer,
                          valid_writer=valid_writer, tokenizer=tokenizer, model_save_path=cur_k_model_save_path)

            torch.cuda.empty_cache()
            gc.collect()
            del model

    if args.do_predict:

        #trained_model_file = os.path.join(args.output_dir, args.model_type + '_' + args.model_info, "2020-03-22@11_57_57")
        # trained_model_file = os.path.join(args.output_dir + '/' + args.model_type)
        args.model_type = args.model_type.lower()
        config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

        config = config_class.from_pretrained(
            args.model_name_or_path
        )
        vocab_file = os.path.join(args.model_name_or_path, 'vocab.txt')
        logger.info("loading the vocab file from {}".format(vocab_file))
        tokenizer = BertTokenizer(vocab_file=vocab_file, do_lower_case=args.do_lower_case)
        data_processor = Data_processor(tokenizer, args.policies_file, args.max_seq_length, args.max_query_length)

        model_path = os.path.join(args.output_dir,
                                  args.model_type + '_' + args.model_info + '_{}fold'.format(args.kfold))


        predict_file_name = args.predict_file.split('/')[-1].split('.')[0]
        cache_dir = os.path.join('/'.join(args.output_dir.split('/')[:-1]),
                                 'predict_cache_{}'.format(predict_file_name))
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        all_results = {}
        for pred_dataloader, pred_features, pred_examples in data_processor.prepare_pred_dataloader(
                args.predict_file, args.predict_batch_size, args.doc_stride, cache_dir=cache_dir):
                if args.do_vote:
                    results = predict_vote(args, model_class, model_path, config, pred_examples, pred_features
                                           ,pred_dataloader, tokenizer, device, n_gpu)
                    all_results.update(results)
                else:
                    results = predict(args, model_class, model_path, config, pred_dataloader, device,  n_gpu)
    
                    output_prediction_file = os.path.join('/'.join(args.output_dir.split('/')[:-1]), "predictions.json")
                    # output_null_log_odds_file = os.path.join(args.output_dir, "null_odds.json")
                    results, _ = write_predictions(args, pred_examples, pred_features, results, n_best_size=10,
                                      output_prediction_file=output_prediction_file, tokenizer=tokenizer)

                    all_results.update(results)

        submit_file_name = '_'.join([args.model_type, args.model_info, predict_file_name])
        # generate submit file
        submit_file = os.path.join('/'.join(args.output_dir.split('/')[:-1]), submit_file_name+'.csv')
        gen_submit_csv(all_results, submit_file)


def train(args, model, train_dataloader, device, num_train_steps,
          valid_examples=None, valid_features=None, valid_dataloader=None, n_gpu=1, train_writer=None,
          valid_writer=None, tokenizer=None, model_save_path=None):

    model = prepare_model(args, model, device, n_gpu)
    global_step = 0
    # prepare optimizer
    optimizer, t_total = prepare_optimizer(args, model, num_train_steps)

    best_Rouge_L = 0.1
    # start training model
    for epoch in range(int(args.num_train_epochs)):
        logger.info("start running {} epoch...".format(str(epoch+1)))
        model.train()
        model.zero_grad()
        with tqdm(total=len(train_dataloader)) as pbar:
            for step, batch in enumerate(train_dataloader):
                if n_gpu == 1:
                    batch = tuple(t.to(device) for t in batch)  # multi-gpu does scattering it-self

                unique_ids, input_ids, input_mask, segment_ids, start_positions, end_positions = batch

                outputs = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids,
                                    start_positions=start_positions, end_positions=end_positions)
                loss = outputs[0]
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # modify learning rate with special warm up BERT uses
                    lr_this_step = args.learning_rate * warmup_linear(global_step / t_total, args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
                pbar.set_description_str('loss:{:.6f}'.format(loss))
                if train_writer:
                    train_writer.add_scalar('loss', loss, global_step)
                    train_writer.add_scalar('lr', lr_this_step, global_step)
                # print loss every 100 steps
                if (step + 1) % args.valid_step == 0:
                    if epoch == 0:
                        continue
                    if args.do_valid:

                        Rouge_L, valid_loss = valid(args, model, valid_examples, valid_features, valid_dataloader,
                                                    device, valid_writer=valid_writer, global_step=global_step, tokenizer=tokenizer)
                        logger.info(
                            "epoch:{}, step:{}, loss:{:.5f}, Rouge_L:{:.5f}".format(epoch + 1, global_step,
                                                                               loss.cpu().item(), Rouge_L))
                        if Rouge_L > best_Rouge_L:
                            best_Rouge_L = Rouge_L
                            model_to_save = model.module if hasattr(model, "module") else model
                            model_to_save.save_pretrained(model_save_path)
                            save_config_file()
                            logger.info(
                                "saved the best model with {:.5f} Rouge_L at the {} step".format(Rouge_L, global_step))
                        model.train()
                    else:
                        logger.info("epoch:{},step:{}, loss:{:.5f}".format(epoch+1, global_step, loss.cpu().item()))

                pbar.update(1)


    return model



def predict(args, model_class, model_path, config, pred_dataloader, device, n_gpu):
    all_results = []
    starts = []
    ends = []
    bath_unique_ids = []
    for k in range(args.kfold):
        cur_k_model_path = os.path.join(model_path, 'fold_{}'.format(k + 1))
        model = model_class.from_pretrained(cur_k_model_path, config=config)
        model = prepare_model(args, model, device, n_gpu)
        model.eval()

        logger.info("Start {} fold predicting ".format(k+1))
        for bath_id, (unique_ids, input_ids, input_mask, segment_ids) in enumerate(tqdm(pred_dataloader)):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids)
                start_logits, end_logits = outputs[0], outputs[1]
            start_logits = start_logits.detach().cpu().numpy()
            end_logits = end_logits.detach().cpu().numpy()
            unique_ids = unique_ids.numpy()
            if k == 0:
                starts.append(start_logits)
                ends.append(end_logits)
                bath_unique_ids.append(unique_ids)
            else:
                starts[bath_id] += start_logits
                ends[bath_id] += end_logits
                # for i, unique_id in enumerate(unique_ids):
                #     result = {
                #         'unique_id': unique_id[0],
                #         'start_logits': start_logits[i],
                #         'end_logits': end_logits[i]
                #     }
                #     all_results.append(result)
        del model

    
    for unique_ids, start_logits, end_logits in zip(bath_unique_ids, starts, ends):
        unique_ids = unique_ids.tolist()
        start_logits = start_logits.tolist()
        end_logits = end_logits.tolist()
        for i, unique_id in enumerate(unique_ids):
            result = {
                'unique_id': unique_id[0],
                'start_logits': start_logits[i],
                'end_logits': end_logits[i]
            }
            all_results.append(result)

    return all_results


# def predict_all(args, model_class, config, pred_dataloader, device, n_gpu):
#     all_results = {}
#     starts = []
#     ends = []
#     model_id = 0
#     for model_type, model_name_or_path in zip(args.predict_model_type, args.pred_model_name_or_path):
#
#         config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type.lower]
#
#         config = config_class.from_pretrained(
#             model_name_or_path
#         )
#         vocab_file = os.path.join(model_name_or_path, 'vocab.txt')
#         logger.info("loading the vocab file from {}".format(vocab_file))
#         tokenizer = BertTokenizer(vocab_file=vocab_file, do_lower_case=args.do_lower_case)
#         data_processor = Data_processor(tokenizer, args.policies_file, args.max_seq_length, args.max_query_length)
#
#         model_path = os.path.join(args.output_dir,
#                                   args.model_type + '_' + args.model_info + '_{}fold'.format(args.kfold))
#
#         predict_file_name = args.predict_file.split('/')[-1].split('.')[0]
#         cache_dir = os.path.join('/'.join(args.output_dir.split('/')[:-1]),
#                                  'predict_cache_{}'.format(predict_file_name))
#         if not os.path.exists(cache_dir):
#             os.makedirs(cache_dir)
#
#
#         bath_unique_ids = []
#         for pred_dataloader, pred_features, pred_examples in data_processor.prepare_pred_dataloader(
#                 args.predict_file, args.predict_batch_size, args.doc_stride, cache_dir=cache_dir):
#             for k in range(args.kfold):
#                 cur_k_model_path = os.path.join(model_path, 'fold_{}'.format(k + 1))
#                 model = model_class.from_pretrained(cur_k_model_path, config=config)
#                 model = prepare_model(args, model, device, n_gpu)
#                 model.eval()
#
#                 logger.info("Start {} fold predicting ".format(k + 1))
#                 for bath_id, (unique_ids, input_ids, input_mask, segment_ids) in enumerate(tqdm(pred_dataloader)):
#                     input_ids = input_ids.to(device)
#                     input_mask = input_mask.to(device)
#                     segment_ids = segment_ids.to(device)
#                     with torch.no_grad():
#                         outputs = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids)
#                         start_logits, end_logits = outputs[0], outputs[1]
#                     start_logits = start_logits.detach().cpu().numpy()
#                     end_logits = end_logits.detach().cpu().numpy()
#                     unique_ids = unique_ids.numpy()
#                     if model_id == 0:
#                         starts.append(start_logits)
#                         ends.append(end_logits)
#                         bath_unique_ids.append(unique_ids)
#                     else:
#                         starts[bath_id] += start_logits
#                         ends[bath_id] += end_logits
#                 model_id += 1
#                         # for i, unique_id in enumerate(unique_ids):
#                         #     result = {
#                         #         'unique_id': unique_id[0],
#                         #         'start_logits': start_logits[i],
#                         #         'end_logits': end_logits[i]
#                         #     }
#                         #     all_results.append(result)
#                 del model
#
#     all_results = []
#
#
#
#     for unique_ids, start_logits, end_logits in zip(bath_unique_ids, starts, ends):
#         unique_ids = unique_ids.tolist()
#         start_logits = start_logits.tolist()
#         end_logits = end_logits.tolist()
#         for i, unique_id in enumerate(unique_ids):
#             result = {
#                 'unique_id': unique_id[0],
#                 'start_logits': start_logits[i],
#                 'end_logits': end_logits[i]
#             }
#             all_results.append(result)
#
#     return all_results

def vote_predictions(all_results):
    all_predict = {}
    example_qid_to_results = collections.defaultdict(list)
    for k, results in enumerate(all_results):
        for qid, result in results.items():
            example_qid_to_results[qid].append(result)

    for qid, results in example_qid_to_results.items():
        asnwer_count = {}
        for answer_docid, answer, span_score in results:
            if answer in asnwer_count.keys():
                asnwer_count[answer][0] += 1
                if span_score > asnwer_count[answer][1]:
                    asnwer_count[answer][1] = span_score
            else:
                asnwer_count[answer] = [1, span_score, answer_docid]
        if len(asnwer_count) < len(all_results):
            result = sorted(asnwer_count.items(), key=lambda x: x[1][0])
        else:
            result = sorted(asnwer_count.items(), key=lambda x: x[1][1])
            # (answer_docid, answer, span_score)
        all_predict[qid] = [result[0][1][2], result[0][0]]
    return all_predict


def predict_vote(args, model_class, model_path, config, pred_examples, pred_features,
                 pred_dataloader, tokenizer, device, n_gpu):
    all_results = []

    for k in range(args.kfold):
        cur_k_model_path = os.path.join(model_path, 'fold_{}'.format(k + 1))
        model = model_class.from_pretrained(cur_k_model_path, config=config)
        model = prepare_model(args, model, device, n_gpu)
        model.eval()

        cur_k_results = []
        logger.info("Start {} fold predicting ".format(k+1))
        for bath_id, (unique_ids, input_ids, input_mask, segment_ids) in enumerate(tqdm(pred_dataloader)):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids)
                start_logits, end_logits = outputs[0], outputs[1]
            start_logits = start_logits.detach().cpu().numpy().tolist()
            end_logits = end_logits.detach().cpu().numpy().tolist()
            unique_ids = unique_ids.numpy().tolist()
            for i, unique_id in enumerate(unique_ids):
                result = {
                    'unique_id': unique_id[0],
                    'start_logits': start_logits[i],
                    'end_logits': end_logits[i]
                }
                cur_k_results.append(result)
        output_prediction_file = os.path.join('/'.join(args.output_dir.split('/')[:-1]), "predictions.json")
        cur_k_results, _ = write_predictions(args, pred_examples, pred_features, cur_k_results, n_best_size=10,
                                    output_prediction_file=output_prediction_file, tokenizer=tokenizer)
        all_results.append(cur_k_results)
    all_results = vote_predictions(all_results)

    return all_results

def save_pred_result(output_dir, results):
    '''
    save prediction result into output_dir, and the file name is submit.csv
    :param output_dir:
    :param results: a list of [[text_id, pred_result], [], ... ]
    :return:
    '''
    output_prediction_file = os.path.join(output_dir, "submit.csv")
    with open(output_prediction_file, "w") as f:
        f.write("id,label" + "\n")
        for each in results:
            f.write(each[0] + ',' + str(each[1]) + "\n")
    logger.info('saved the predicted results in the {}'.format(output_prediction_file))


def valid(args, model, valid_examples, valid_features, valid_dataloader, device, valid_writer=None,
          global_step=None, tokenizer=None):
    '''
    evaluate f1 score of dev set
    :param model:
    :param vaild_dataloader:
    :param device:
    :return:
    '''
    model.eval()
    all_results = []
    valid_loss = []
    for unique_ids, input_ids, input_mask, segment_ids, start_idxs, end_idxs in valid_dataloader:

        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids,
                            start_positions=start_idxs, end_positions=end_idxs)
            loss, start_logits, end_logits = outputs[0], outputs[1], outputs[2]
        start_logits = start_logits.detach().cpu().numpy().tolist()
        end_logits = end_logits.detach().cpu().numpy().tolist()
        unique_ids = unique_ids.numpy().tolist()
        valid_loss.append(loss.mean().cpu().numpy())
        for unique_id, start_logit, end_logit in zip(unique_ids, start_logits, end_logits):
            result = {
                'unique_id': unique_id[0],
                'start_logits': start_logit,
                'end_logits': end_logit
            }
            all_results.append(result)
    results = write_predictions(args, valid_examples, valid_features, all_results, n_best_size=10, tokenizer=tokenizer)
    Rouge_L = cal_Rouge(results, valid_examples)
    valid_loss = np.mean(valid_loss)
    if valid_writer:
        valid_writer.add_scalar('Rouge_L', Rouge_L, global_step)
        valid_writer.add_scalar('loss', valid_loss, global_step)
    return Rouge_L, valid_loss


def cal_Rouge(results, examples):
    rouge = Rouge(beta=1.0)

    for example in examples:
        pred_answer = results[example.qid][1]
        if len(pred_answer) == 1:
            pred_answer.replace('.', '。')
        rouge.add_inst(pred_answer, example.answer)

    scores = rouge.score()

    return scores


def write_predictions(args, all_examples, all_features, all_results, n_best_size=1, output_prediction_file=None,
                      tokenizer=None):

    example_qid_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_qid_to_features[feature.qid].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result["unique_id"]] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit", "feature_null_score"])

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()
    empty_num = 0
    no_answers = []

    for (example_index, example) in enumerate(all_examples):

        features = example_qid_to_features[example.qid]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        min_null_feature_index = 0  # the paragraph slice with min mull score
        null_start_logit = 0  # the start logit at the slice with min null score
        null_end_logit = 0  # the end logit at the slice with min null score
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id[0]]
            start_indexes = _get_best_indexes(result["start_logits"], n_best_size)
            end_indexes = _get_best_indexes(result["end_logits"], n_best_size)
            # if we could have irrelevant answers, get the min score of irrelevant
            feature_null_score = result["start_logits"][0] + result["end_logits"][0]
            if args.version_2_with_negative:
                # feature_null_score = result["start_logits"][0] + result["end_logits"][0]
                if feature_null_score < score_null:
                    score_null = feature_null_score
                    min_null_feature_index = feature_index
                    null_start_logit = result["start_logits"][0]
                    null_end_logit = result["end_logits"][0]
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens[0]):
                        continue
                    if end_index >= len(feature.tokens[0]):
                        continue
                    if start_index not in feature.token_to_orig_map[0]:
                        continue
                    if end_index not in feature.token_to_orig_map[0]:
                        continue
                    if not feature.token_is_max_context[0].get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > args.max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            feature_null_score=feature_null_score,
                            start_logit=result["start_logits"][start_index],
                            end_logit=result["end_logits"][end_index]))

        if args.version_2_with_negative:
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_null_feature_index,
                    start_index=0,
                    end_index=0,
                    feature_null_score=score_null,
                    start_logit=null_start_logit,
                    end_logit=null_end_logit))

        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit - x.feature_null_score),
            reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "answer_docid", "start_logit", "end_logit", "span_score"])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= args.n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index > 0:  # this is a non-null prediction
                tok_tokens = feature.tokens[0][pred.start_index:(pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[0][pred.start_index]
                orig_doc_end = feature.token_to_orig_map[0][pred.end_index]
                orig_text = example.contexts[feature.context_index][orig_doc_start:(orig_doc_end + 1)]
                answer_docid = example.docids[feature.context_index]
                tok_text = " ".join(tok_tokens)

                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())

                final_text = get_final_text(tok_text, orig_text, args.do_lower_case, args.verbose_logging,
                                            tokenizer=tokenizer)
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:

                final_text = ""
                answer_docid = '0000000000000000000'
                seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    answer_docid=answer_docid,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit,
                    span_score=pred.start_logit + pred.end_logit - pred.feature_null_score
                )
            )

        # if we didn't inlude the empty option in the n-best, inlcude it
        if args.version_2_with_negative:
            if "" not in seen_predictions:
                nbest.append(
                    _NbestPrediction(
                        text="", answer_docid="00000000000", start_logit=null_start_logit,
                        end_logit=null_end_logit, span_score=-1000000.0))

            # In very rare edge cases we could only have single null prediction.
            # So we just create a nonce prediction in this case to avoid failure.
            if len(nbest) == 1:
                nbest.insert(0, _NbestPrediction(text="empty", answer_docid="00000000000", start_logit=0.0,
                                                 end_logit=0.0, span_score=-10000000.0))

        if not nbest:
            nbest.append(
                _NbestPrediction(text="empty", answer_docid="00000000000", start_logit=0.0, end_logit=0.0, span_score=-10000000.0))

        assert len(nbest) >= 1

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text != '':
                    best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["answer_docid"] = entry.answer_docid
            output["question"] = example.qusetion
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            output["span_score"] = entry.span_score
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        if not args.version_2_with_negative:
            all_predictions[example.qid] = [nbest_json[0]["answer_docid"], nbest_json[0]["text"], nbest_json[0]["span_score"]]
        else:
            # predict "" if the null score - the score of best non-null > threshold
            score_diff = score_null - best_non_null_entry.start_logit - (
                best_non_null_entry.end_logit)
            scores_diff_json[example.qid] = score_diff
            if score_diff > args.null_score_diff_threshold:
                all_predictions[example.qid] = ["null", ""]
                empty_num += 1
                no_answers.append(example.qid)
            else:
                all_predictions[example.qid] = [best_non_null_entry.answer_docid, best_non_null_entry.text]
        all_nbest_json[example.qid] = nbest_json
    if not args.do_valid:
        logger.info("all results of prediction :{}".format(len(all_predictions)))
        logger.info("There were {} empty results".format(empty_num))

    # save result of prediction
    if output_prediction_file != None:
        with open(output_prediction_file, "wb") as writer:
            pickle.dump(all_predictions, writer)

    #
    # if args.version_2_with_negative:
    #   with open(output_null_log_odds_file, "wb") as writer:
    #     pickle.dump(scores_diff_json, writer)
    if not args.do_valid:
        return all_predictions, all_nbest_json
    else:
        return all_predictions


def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging, tokenizer=None):
  """Project the tokenized prediction back to the original text."""

  # When we created the data, we kept track of the alignment between original
  # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
  # now `orig_text` contains the span of our original text corresponding to the
  # span that we predicted.
  #
  # However, `orig_text` may contain extra characters that we don't want in
  # our prediction.
  #
  # For example, let's say:
  #   pred_text = steve smith
  #   orig_text = Steve Smith's
  #
  # We don't want to return `orig_text` because it contains the extra "'s".
  #
  # We don't want to return `pred_text` because it's already been normalized
  # (the SQuAD eval script also does punctuation stripping/lower casing but
  # our tokenizer does additional normalization like stripping accent
  # characters).
  #
  # What we really want to return is "Steve Smith".
  #
  # Therefore, we have to apply a semi-complicated alignment heruistic between
  # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
  # can fail in certain cases in which case we just return `orig_text`.

  def _strip_spaces(text):
    ns_chars = []
    ns_to_s_map = collections.OrderedDict()
    for (i, c) in enumerate(text):
      if c == " ":
        continue
      ns_to_s_map[len(ns_chars)] = i
      ns_chars.append(c)
    ns_text = "".join(ns_chars)
    return (ns_text, ns_to_s_map)

  # We first tokenize `orig_text`, strip whitespace from the result
  # and `pred_text`, and check if they are the same length. If they are
  # NOT the same length, the heuristic has failed. If they are the same
  # length, we assume the characters are one-to-one aligned.

  #tokenizer = BasicTokenizer(do_lower_case=do_lower_case) # xc_erroe: why use BasicTokenizer
  tok_text = " ".join(tokenizer.tokenize(orig_text))

  start_position = tok_text.find(pred_text)
  if start_position == -1:
    if verbose_logging:
      logger.info(
          "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
    return orig_text
  end_position = start_position + len(pred_text) - 1

  (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
  (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

  if len(orig_ns_text) != len(tok_ns_text):
    if verbose_logging:
      logger.info("Length not equal after stripping spaces: '%s' vs '%s'",
                      orig_ns_text, tok_ns_text)
    return orig_text

  # We then project the characters in `pred_text` back to `orig_text` using
  # the character-to-character alignment.
  tok_s_to_ns_map = {}
  for (i, tok_index) in six.iteritems(tok_ns_to_s_map):
    tok_s_to_ns_map[tok_index] = i

  orig_start_position = None
  if start_position in tok_s_to_ns_map:
    ns_start_position = tok_s_to_ns_map[start_position]
    if ns_start_position in orig_ns_to_s_map:
      orig_start_position = orig_ns_to_s_map[ns_start_position]

  if orig_start_position is None:
    if verbose_logging:
      logger.info("Couldn't map start position")
    return orig_text

  orig_end_position = None
  if end_position in tok_s_to_ns_map:
    ns_end_position = tok_s_to_ns_map[end_position]
    if ns_end_position in orig_ns_to_s_map:
      orig_end_position = orig_ns_to_s_map[ns_end_position]

  if orig_end_position is None:
    if verbose_logging:
      logger.info("Couldn't map end position")
    return orig_text

  output_text = orig_text[orig_start_position:(orig_end_position + 1)]
  return output_text


def _get_best_indexes(logits, n_best_size):
  """Get the n-best logits from a list."""
  index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

  best_indexes = []
  for i in range(len(index_and_score)):
    if i >= n_best_size:
      break
    best_indexes.append(index_and_score[i][0])
  return best_indexes

def _compute_softmax(scores):
  """Compute softmax probability over raw logits."""
  if not scores:
    return []

  max_score = None
  for score in scores:
    if max_score is None or score > max_score:
      max_score = score

  exp_scores = []
  total_sum = 0.0
  for score in scores:
    x = math.exp(score - max_score)
    exp_scores.append(x)
    total_sum += x

  probs = []
  for score in exp_scores:
    probs.append(score / total_sum)
  return probs


def gen_submit_csv(all_predictions, submit_file):
    with open(submit_file, 'w', encoding='utf-8') as fout:
        fout.write('\ufeffid\tdocid\tanswer\n')

        for id, answer in all_predictions.items():
            s = id +'\t'+ answer[0] + '\t' + answer[1] + '\n'
            fout.write(s)
    logger.info("submit csv saved in {}".format(submit_file))

if __name__ == "__main__":
    main()


