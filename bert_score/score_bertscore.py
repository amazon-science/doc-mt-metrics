# Original Copyright (c) 2019 Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q. Weinberger, and Yoav Artzi.
# Modifications Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.

import argparse
import json
import os
import numpy as np
from bert_score import BERTScorer
from add_context import add_context

pair2ref = {("wmt21.tedtalks", "en-de"): "refA", ("wmt21.tedtalks", "zh-en"): "refB",
               ("wmt21.tedtalks", "en-ru"): "refA",
               ("wmt21.news", "en-de"): "refC", ("wmt21.news", "zh-en"): "refB",
               ("wmt21.news", "en-ru"): "refA"}


# the code assumes the following directory structure:
# hypotheses files: dir/hyps/*
# reference(s) file: dir/ref (see pair2ref)

def main(args):

    scorer = BERTScorer(lang=args.lang, rescale_with_baseline=False)


    scores_path = args.dir + "{}bertscore_scores.json".format("doc-" if args.doc else "")

    scores = {level: {} for level in [args.level]}

    ref = open(args.dir + "/ref", "r").read().splitlines()

    for sysname in os.listdir(args.dir + "/hyps/"):
        cand = open(args.dir + "/hyps/" + sysname, "r").read().splitlines()

        if args.doc:
            doc_ids = open(args.dir + "/docids", "r").read().splitlines()

            cand = add_context(orig_txt=cand, context=ref, doc_ids=doc_ids, sep_token=scorer._tokenizer.sep_token)
            ref = add_context(orig_txt=ref, context=ref, doc_ids=doc_ids, sep_token=scorer._tokenizer.sep_token)

        P, R, F1 = scorer.score(cand, ref, context=args.doc)
        seg_score = F1.cpu().numpy()
        scores[args.level][sysname] = [np.mean(seg_score) if not args.seg else [float(x) for x in seg_score]]

    with open(scores_path, 'w') as fp:
        json.dump(scores, fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Score BERTScore models.')
    parser.add_argument('--lang', default="en", help='evaluation language')
    parser.add_argument('--doc', action="store_true", help='document- or sentence-level comet')
    parser.add_argument('--dir', default="", type=str, help='directory where the files are located')
    parser.add_argument('--level', default="sys", choices=["seg", "sys"],
                        help='whether segment-level or system-level scores will be computed')

    args = parser.parse_args()

    main(args)
