# Original Copyright (c) 2019 Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q. Weinberger, and Yoav Artzi.
# Modifications Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.

import argparse
import json
import os
import numpy as np
from bert_score import BERTScorer

pair2ref = {("wmt21.tedtalks", "en-de"): "refA", ("wmt21.tedtalks", "zh-en"): "refB",
            ("wmt21.tedtalks", "en-ru"): "refA",
            ("wmt21.news", "en-de"): "refC", ("wmt21.news", "zh-en"): "refB",
            ("wmt21.news", "en-ru"): "refA", ("wmt21.news", "de-en"): "refA",
            ("wmt21.news", "cs-en"): "refA", ("wmt21.news", "ja-en"): "refA",
            ("wmt21.news", "ru-en"): "refA", ("wmt21.news", "ha-en"): "refA",
            ("wmt21.news", "is-en"): "refA", ("wmt21.news", "en-zh"): "refA",
            ("wmt21.news", "en-ja"): "refA", ("wmt21.news", "en-cs"): "refA",
            ("wmt21.news", "de-fr"): "refA", ("wmt21.news", "fr-de"): "refA"}


def add_context(txt1, docs, txt2=None, sep_token="</s>"):
    """
    Function that reformats the input by adding the two previous sentences as context
    :param txt1: the original text
    :param docs: a list that contains the size of each document in the dataset
    :param txt2: an additional textfile, used to add reference context to the hypothesis
    :param sep_token: the separator token for each model
    """
    k = 0
    l = 0
    new_txt = []
    for i, line in enumerate(txt1):
        k += 1
        if k <= docs[l]:
            if k == 1:
                new_txt.append(txt1[i])
            elif k == 2:
                context = " {} ".format(sep_token).join(txt2[i - 1:i]) if txt2 is not None else " {} ".format(
                    sep_token).join(txt1[i - 1:i])
                new_txt.append(context + " {} ".format(sep_token) + txt1[i])
            else:
                context = " {} ".format(sep_token).join(txt2[i - 2:i]) if txt2 is not None else " {} ".format(
                    sep_token).join(txt1[i - 2:i])
                new_txt.append(context + " {} ".format(sep_token) + txt1[i])
        else:
            k = 1
            l += 1
            new_txt.append(txt1[i])
    return new_txt


def main(args):

    ref_name = pair2ref[(args.testset, args.lp)]

    print("Campaign: {}\n Language Pair: {}\n".format(args.testset, args.lp))

    scorer = BERTScorer(lang=lp.split("-")[-1], rescale_with_baseline=False)

    file_path = args.dir + "outs/{}/{}/".format(args.testset, args.lp)
    scores_path = args.dir + "scores/{}_{}_{}{}bertscore_scores.json".format(args.testset, args.lp,
                                                                                "seg-" if args.seg else "",
                                                                                "ctx" if args.context else "")

    print(scores_path)

    scores = {level: {} for level in ['sys']}

    ref = open(file_path + ref_name, "r").read().splitlines()
    if args.context:
        sep_token = scorer._tokenizer.sep_token
        with open(args.dir + "docs/{}/{}/ids".format(args.testset, args.lp), "r") as fp:
            docs = json.load(fp)
            ref = add_context(ref, docs, sep_token)
    for sysname in os.listdir(file_path):
        print("evaluating {}".format(sysname))
        cand = open(file_path + sysname, "r").read().splitlines()
        if args.context:
            org_ref = open(file_path + ref_name, "r").read().splitlines()
            cand = add_context(cand, org_ref, docs, sep_token)

        P, R, F1 = scorer.score(cand, ref, context=args.context)
        seg_score = F1.cpu().numpy()
        scores['sys'][sysname] = [float(np.mean(seg_score)) if not args.seg else [float(x) for x in seg_score]]
        print('System-level metric for {} is : {}'.format(sysname, np.mean(seg_score)))

        with open(scores_path, 'w') as fp:
            json.dump(scores, fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Score BERTScore.')
    parser.add_argument('--context', required=True, type=bool, help='document- or sentence-level comet')
    parser.add_argument('--testset', required=False, default="wmt21.news", choices=['wmt21.news', 'wmt21.tedtalks'],
                        help='name of wmt campaign')
    parser.add_argument('--lp', required=True, type=str, help='language pair')
    parser.add_argument('--dir', required=True, type=str, help='directory where the scores will be saved')
    parser.add_argument('--seg', required=False, default=False,
                    help='if segment-level or system-level scores will be stored')

    args = parser.parse_args()

    main(args)
