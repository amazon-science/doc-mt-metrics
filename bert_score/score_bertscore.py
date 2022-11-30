# Original Copyright (c) 2019 Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q. Weinberger, and Yoav Artzi.
# Modifications Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.

import argparse
import json
import os
import numpy as np
from bert_score import BERTScorer
from add_context import add_context


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
    scores[args.level] = [np.mean(seg_score) if not args.seg else [float(x) for x in seg_score]]

    with open(scores_path, 'w') as fp:
        json.dump(scores, fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Score BERTScore.')
    parser.add_argument('--lang', default="en", help='evaluation language')
    parser.add_argument('--doc', action="store_true", help='document- or sentence-level bertscore')
    parser.add_argument('--dir', default="", type=str, help='directory where the scores will be saved')
    parser.add_argument('--hyp', required=True, type=str, help='the translated text')
    parser.add_argument('--target', required=False, type=str, help='the target text')
    parser.add_argument('--doc_lens', required=False, type=str, help='the lengths of each document in the data')
    parser.add_argument('--level', required=False, default="sys", choices=["seg", "sys"],
                        help='whether segment-level or system-level scores will be computed')

    args = parser.parse_args()

    main(args)
