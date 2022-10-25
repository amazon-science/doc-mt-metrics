# Original Copyright 2021 Google LLC
# Modifications Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved. 

from mt_metrics_eval import data
import json
import numpy as np
import argparse


def compute_corr(testset, lp, system, dir, level='sys', human=False, type="mqm"):
    filename = "{}/scores/{}_{}_{}_scores.json".format(dir, testset, lp, system)
    inds = "{}/index/{}/{}/inds".format(dir, testset, lp)

    with open(filename) as json_file:
        scores = json.load(json_file)

    scores["seg"] = {k: v[0] for k, v in scores["sys"].items()}
    if "wmt21" in testset and lp in ["en-de", "zh-en", "en-ru"]:
        if "tedtalks" in testset and lp == "en-ru":
            scores["sys"] = {k: [np.mean(v[0])] for k, v in scores["sys"].items()}
        else:
            with open(inds) as f:
                ind_list = np.array([x == "True" for x in f.read().splitlines()])
                scores["sys"] = {k: [np.mean(np.array(v[0])[ind_list])] for k, v in scores["sys"].items()}

    evs = data.EvalSet(testset, lp)

    include_outliers = False
    ref = {evs.std_ref}
    close_refs = {'refB'} if testset == "wtm21.news" and lp == "en-de" else set()

    # Official WMT correlations.
    gold_scores = evs.Scores(level, type)
    sys_names = set(gold_scores) - ref - close_refs
    if not human:
        sys_names -= evs.human_sys_names
    if not include_outliers:
        sys_names -= evs.outlier_sys_names
    corr = evs.Correlation(gold_scores, scores[level], sys_names)

    return corr

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate correlation of metrics with human judgements.')
    parser.add_argument('--testset', required=False, default="wmt21.news", help='name of wmt campaign')
    parser.add_argument('--lp', required=False, default="en-de", help='language pair')
    parser.add_argument('--system', required=False, default="seg-ctx-comet21", help='metric name')
    parser.add_argument('--dir', required=False, default="/home/ubuntu/Projects/myPrism/",
                        help='directory where the scores are saved')
    parser.add_argument('--level', required=False, default="sys", help='level where the correlation will be computed')
    parser.add_argument('--human', action='store_true', help='include human translations')
    parser.add_argument('--type', required=False, default="mqm", help='type of human judgements')

    args = parser.parse_args()

    corr = compute_corr(**vars(args))

    if args.level == "sys":
        print(f'{args.level}: Pearson={corr.Pearson()[0]:f}')
    elif args.level == "seg":
        print(f'{args.level}: Kendall-like={corr.Kendall()[0]:f}')
    else:
        print("Invalid level input")

