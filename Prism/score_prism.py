# Original Copyright (c) Brian Thompson. Licensed under the MIT License.
# Modifications Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.

import json
import os
from prism.prism import Prism

source = True

for testset, lp, ref_name, seg in [
    # ("wmt21.tedtalks", "en-de", "refA", True), ("wmt21.tedtalks", "zh-en", "refB", True),
    # ("wmt21.tedtalks", "en-ru", "refA", False),
    # ("wmt21.news", "en-de", "refC", True), ("wmt21.news", "zh-en", "refB", True),
    # ("wmt21.news", "en-ru", "refA", True), ("wmt21.news", "de-en", "refA", False),
    # ("wmt21.news", "cs-en", "refA", False), ("wmt21.news", "ja-en", "refA", False),
    # ("wmt21.news", "ru-en", "refA", False),
    # # ("wmt21.news", "ha-en", "refA", False), ("wmt21.news", "is-en", "refA", False),
    # ("wmt21.news", "en-zh", "refA", False), ("wmt21.news", "en-ja", "refA", False),
    # ("wmt21.news", "en-cs", "refA", False),
    ("wmt21.news", "de-fr", "refA", False), ("wmt21.news", "fr-de", "refA", False)
]:

    prism_dir = "./"
    model_dir = "prism/m39v1/"
    file_path = prism_dir + "outs/{}/{}/".format(testset, lp)
    scores_path = prism_dir + "scores/{}_{}_{}{}prism_scores.json".format(testset, lp, "seg-" if seg else "",
                                                                          "src" if source else "")

    print(scores_path)

    prism = Prism(model_dir=model_dir, lang=lp.split("-")[-1])

    print('Prism identifier:', prism.identifier())

    scores = {level: {} for level in ['sys']}

    ref = open(file_path + ref_name, "r").readlines()
    if source:
        src = open(prism_dir + "ins/{}/{}/src".format(testset, lp), "r").read().splitlines()
    for sysname in os.listdir(file_path):
        # if sysname != ref_name:
        cand = open(file_path + sysname, "r").readlines()
        if source:
            score = prism.score(cand=cand, src=src, segment_scores=seg)
        else:
            score = prism.score(cand=cand, ref=ref, segment_scores=seg)
        scores['sys'][sysname] = [float(score) if not seg else [float(x) for x in score]]
        print('System-level metric for {} is : {}'.format(sysname, score))

    with open(scores_path, 'w') as fp:
        json.dump(scores, fp)
