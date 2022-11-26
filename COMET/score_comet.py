# Original Copyright (C) 2020 Unbabel
# Modifications Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.

import argparse
import os
import json
from comet import download_model, load_from_checkpoint

pair2ref = {("wmt21.tedtalks", "en-de"): "refA", ("wmt21.tedtalks", "zh-en"): "refB",
               ("wmt21.tedtalks", "en-ru"): "refA",
               ("wmt21.news", "en-de"): "refC", ("wmt21.news", "zh-en"): "refB",
               ("wmt21.news", "en-ru"): "refA", ("wmt21.news", "de-en"): "refA",
               ("wmt21.news", "cs-en"): "refA", ("wmt21.news", "ja-en"): "refA",
               ("wmt21.news", "ru-en"): "refA", ("wmt21.news", "ha-en"): "refA",
               ("wmt21.news", "is-en"): "refA", ("wmt21.news", "en-zh"): "refA",
               ("wmt21.news", "en-ja"): "refA", ("wmt21.news", "en-cs"): "refA",
               ("wmt21.news", "de-fr"): "refA", ("wmt21.news", "fr-de"): "refA"}


def add_context(txt1, docs, txt2=None):
    """
    Function that reformats the input by adding the two previous sentences as context
    :param txt1: the original text
    :param docs: a list that contains the size of each document in the dataset
    :param txt2: an additional textfile, used to add reference context to the hypothesis
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
                context = " </s> ".join(txt2[i - 1:i]) if txt2 is not None else " </s> ".join(txt1[i - 1:i])
                new_txt.append(context + " </s> " + txt1[i])
            else:
                context = " </s> ".join(txt2[i - 2:i]) if txt2 is not None else " </s> ".join(txt1[i - 2:i])
                new_txt.append(context + " </s> " + txt1[i])
        else:
            k = 1
            l += 1
            new_txt.append(txt1[i])
    return new_txt


def main(args):

    ref_name = pair2ref[(args.testset, args.lp)]

    print("Campaign: {}\n Language Pair: {}\n".format(args.testset, args.lp))

    model_name = "wmt21-comet-mqm" if args.use_ref else "wmt21-comet-qe-mqm"
    model_path = download_model(model_name)
    model = load_from_checkpoint(model_path)
    if args.context:
        model.set_document_level()

    file_path = args.dir + "outs/{}/{}/".format(args.testset, args.lp)
    scores_path = args.dir + "scores/{}_{}_{}{}comet21{}_scores.json".format(args.testset, args.lp,
                                                                                "seg-" if args.seg else "",
                                                                                "ctx" if args.context else "",
                                                                                "-qe" if not args.use_ref else "")

    print(scores_path)

    scores = {level: {} for level in ['sys']}

    src = open(args.dir + "ins/{}/{}/src".format(args.testset, args.lp), "r").read().splitlines()
    ref = open(file_path + ref_name, "r").read().splitlines()
    if args.context:
        with open(args.dir + "docs/{}/{}/ids".format(args.testset, args.lp), "r") as fp:
            docs = json.load(fp)
        # add contexts to reference and source texts
        ref = add_context(ref, docs)
        src = add_context(src, docs)
    for sysname in os.listdir(file_path):
        print("evaluating {}".format(sysname))
        cand = open(file_path + sysname, "r").read().splitlines()
        if args.context:
            # add reference/hypothesis context to the hypothesis for comet/comet-qe
            if not args.use_ref:
                cand = add_context(cand, docs)
            else:
                org_ref = open(file_path + ref_name, "r").read().splitlines()
                cand = add_context(cand, docs, org_ref)
        if args.use_ref:
            data = [{"src": x, "mt": y, "ref": z} for x, y, z in zip(src, cand, ref)]
        else:
            data = [{"src": x, "mt": y} for x, y in zip(src, cand)]
        seg_score, sys_score = model.predict(data, batch_size=32, gpus=1)
        scores['sys'][sysname] = [float(sys_score) if not args.seg else [float(x) for x in seg_score]]
        print('System-level metric for {} is : {}'.format(sysname, sys_score))

    with open(scores_path, 'w') as fp:
        json.dump(scores, fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Score COMET models.')
    parser.add_argument('--context', required=True, type=bool, help='document- or sentence-level comet')
    parser.add_argument('--testset', required=False, default="wmt21.news", choices=['wmt21.news', 'wmt21.tedtalks'],
                        help='name of wmt campaign')
    parser.add_argument('--lp', required=True, type=str, help='language pair')
    parser.add_argument('--dir', required=True, type=str, help='directory where the scores will be saved')
    parser.add_argument('--use_ref', required=False, default=True,
                        help='whether evaluation is reference-based or reference-free')
    parser.add_argument('--seg', required=False, default=False,
                    help='if segment-level or system-level scores will be stored')

    args = parser.parse_args()

    main(args)
