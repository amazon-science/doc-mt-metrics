# Original Copyright (c) 2019 Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q. Weinberger, and Yoav Artzi.
# Modifications Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.

import argparse
import json
import os
import numpy as np
from bert_score import BERTScorer


def add_context(org_txt: List[str], context: List[str], docs: List[int], sep_token: str = "</s>",
                ws: int = 2) -> List[int]:
    """Function that adds the previous sentences as context to the current sentence, respecting document boundaries
    :param org_txt: the original text
    :param context: the text from which the context will be taken (same as org_txt for source/reference)
    :param docs: the size of each document in the text
    :param sep_token: the separator token of the tokenizer for the specific model
    :param ws: the window size, maximum of the previous sentences to be considered as context
    :return: the original text augmented with context
    """
    k = 0
    l = 0
    augm_txt = []
    for i, line in enumerate(org_txt):
        if k < docs[l]:
            context = context[i - min(k, ws):i]
            augm_txt.append(" {} ".format(sep_token).join(context + [org_txt[i]]))
        else:
            k = -1
            l += 1
        k += 1
    return augm_txt


def main(args):

    scorer = BERTScorer(lang=lp.split("-")[-1], rescale_with_baseline=False)
    
    scores_path = args.dir + "scores/{}bertscore_scores.json".format("doc-" if args.context else "")


    scores = {}

    cand = open(args.hyp, "r").read().splitlines()
    ref = open(args.target, "r").read().splitlines()

    if args.doc:
        doc_lens = open(args.doc_lens, "r").read().splitlines()

        cand = add_context(org_txt=cand, context=ref, docs=doc_lens, sep_token=model.encoder.tokenizer.sep_token)
        ref = add_context(org_txt=ref, context=ref, docs=doc_lens, sep_token=model.encoder.tokenizer.sep_token)


    P, R, F1 = scorer.score(cand, ref, context=args.doc)
    seg_score = F1.cpu().numpy()
    scores[args.level] = [float(sys_score) if not args.seg else [float(x) for x in seg_score]]

    with open(scores_path, 'w') as fp:
        json.dump(scores, fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Score BERTScore.')
    parser.add_argument('--doc', action="store_true", help='document- or sentence-level bertscore')
    parser.add_argument('--dir', default="", type=str, help='directory where the scores will be saved')
    parser.add_argument('--hyp', required=True, type=str, help='the translated text')
    parser.add_argument('--target', required=False, type=str, help='the target text')
    parser.add_argument('--doc_lens', required=False, type=str, help='the lengths of each document in the data')
    parser.add_argument('--level', required=False, default="sys", choices=["seg", "sys"],
                        help='whether segment-level or system-level scores will be computed')

    args = parser.parse_args()

    main(args)
