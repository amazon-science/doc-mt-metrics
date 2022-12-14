import torch
from tqdm import tqdm
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, MBart50Tokenizer
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


class MBARTPrism:
    def __init__(self, src_lang, tgt_lang, checkpoint='facebook/mbart-large-cc25', device='None'):
        # Set up model
        langs = ["ar_AR", "cs_CZ", "de_DE", "en_XX", "es_XX", "et_EE", "fi_FI", "fr_XX", "gu_IN", "hi_IN", "it_IT",
                 "ja_XX", "kk_KZ", "ko_KR", "lt_LT", "lv_LV", "my_MM", "ne_NP", "nl_XX", "ro_RO", "ru_RU", "si_LK",
                 "tr_TR", "vi_VN", "zh_CN"]
        # , "pl_PL", "ta_IN"]
        src_lang = [l for l in langs if src_lang in l][0]
        tgt_lang = [l for l in langs if tgt_lang in l][0]

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = MBart50Tokenizer.from_pretrained(checkpoint, src_lang=src_lang, tgt_lang=tgt_lang)
        self.model = MBartForConditionalGeneration.from_pretrained(checkpoint)
        self.model.eval()
        self.model.to(device)

        # Set up loss
        self.loss_fct = nn.NLLLoss(reduction='none', ignore_index=self.model.config.pad_token_id)
        self.lsm = nn.LogSoftmax(dim=1)

    def score(self, cand, ref, batch_size, doc, segment_scores=True):
        """ Score a batch of examples """

        if len(cand) != len(ref):
            raise Exception(f'Length of cand ({len(cand)}) does not match length of ref ({len(ref)})')

        sent_scores = [[], []]

        with torch.no_grad():
            for sent_idx, (srcs, tgts) in enumerate([(ref, cand), (cand, ref)]):
                for i in tqdm(range(0, len(srcs), batch_size)):
                    src_list = srcs[i: i + batch_size]
                    tgt_list = tgts[i: i + batch_size]
                    with torch.no_grad():
                        encoded_src = self.tokenizer(
                            src_list,
                            truncation=True,
                            padding=True,
                            return_tensors='pt',
                            max_length=self.tokenizer.model_max_length
                        )
                        with self.tokenizer.as_target_tokenizer():
                            encoded_tgt = self.tokenizer(
                                tgt_list,
                                truncation=True,
                                padding=True,
                                return_tensors='pt',
                                max_length=self.tokenizer.model_max_length
                            )
                            tgt_len = [len(self.tokenizer(sent.split("</s>")[-1]).input_ids) for sent in tgt_list]
                            if doc:
                                start_toks = [len(self.tokenizer(sent).input_ids) - tgt_len[i] for i, sent in
                                              enumerate(tgt_list)]
                            else:
                                start_toks = [0] * len(tgt_list)

                        src_tokens = encoded_src['input_ids'].to(self.device)
                        src_mask = encoded_src['attention_mask'].to(self.device)

                        tgt_tokens = encoded_tgt['input_ids'].to(self.device)
                        # tgt_mask = encoded_tgt['attention_mask']
                        # tgt_len = tgt_mask.sum(dim=1).to(self.device)

                        output = self.model(
                            input_ids=src_tokens,
                            attention_mask=src_mask,
                            labels=tgt_tokens
                        )
                        logits = output.logits.view(-1, self.model.config.vocab_size)
                        loss = self.loss_fct(self.lsm(logits), tgt_tokens.view(-1))
                        loss = loss.view(tgt_tokens.shape[0], -1)
                        ppl = []
                        for i, s in enumerate(loss):
                            ppl.append(s[start_toks[i]:start_toks[i] + tgt_len[i] - 1].sum() / (tgt_len[i] - 1))
                        # loss = loss.sum(dim=1) / tgt_len
                        curr_score_list = [-x.item() for x in ppl]
                        sent_scores[sent_idx] += curr_score_list

        segm_scores = np.mean(sent_scores, axis=0)
        sys_score = np.mean(segm_scores) if not segment_scores else segm_scores

        return sys_score


    def score_src(self, src, tgt, batch_size, doc, segment_scores=True):
        """ Score a batch of examples """

        if len(src) != len(tgt):
            raise Exception(f'Length of cand ({len(src)}) does not match length of ref ({len(tgt)})')

        sent_scores = []

        with torch.no_grad():
            for i in tqdm(range(0, len(src), batch_size)):
                src_list = src[i: i + batch_size]
                tgt_list = tgt[i: i + batch_size]
                with torch.no_grad():
                    encoded_src = self.tokenizer(
                        src_list,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )
                    with self.tokenizer.as_target_tokenizer():
                        encoded_tgt = self.tokenizer(
                            tgt_list,
                            truncation=True,
                            padding=True,
                            return_tensors='pt'
                        )
                        tgt_len = [len(self.tokenizer(sent.split("</s>")[-1]).input_ids) for sent in tgt_list]
                        if doc:
                            start_toks = [len(self.tokenizer(sent).input_ids) - tgt_len[i] for i, sent in
                                          enumerate(tgt_list)]
                        else:
                            start_toks = [0] * len(tgt_list)

                    src_tokens = encoded_src['input_ids'].to(self.device)
                    src_mask = encoded_src['attention_mask'].to(self.device)

                    tgt_tokens = encoded_tgt['input_ids'].to(self.device)
                    # tgt_mask = encoded_tgt['attention_mask']
                    # tgt_len = tgt_mask.sum(dim=1).to(self.device)

                    output = self.model(
                        input_ids=src_tokens,
                        attention_mask=src_mask,
                        labels=tgt_tokens
                    )
                    logits = output.logits.view(-1, self.model.config.vocab_size)
                    loss = self.loss_fct(self.lsm(logits), tgt_tokens.view(-1))
                    loss = loss.view(tgt_tokens.shape[0], -1)
                    ppl = []
                    for i, s in enumerate(loss):
                        ppl.append(s[start_toks[i]:start_toks[i] + tgt_len[i] - 1].sum() / (tgt_len[i] - 1))
                    # loss = loss.sum(dim=1) / tgt_len
                    curr_score_list = [-x.item() for x in ppl]
                    sent_scores += curr_score_list

        sys_score = np.mean(sent_scores) if not segment_scores else sent_scores

        return sys_score
