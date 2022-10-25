# Original Copyright (c) Brian Thompson. Licensed under the MIT License.
# Modifications Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.

import json
import os
from prism.ctx_prism import Prism

context = True
source = True

def add_prompt(txt1, docs):
    k = 0
    l = 0
    new_txt = []
    for i, line in enumerate(txt1):
        k += 1
        if k <= docs[l]:
            if k == 1:
                new_txt.append("")
            elif k == 2:
                new_txt.append(" ".join(txt1[i - 1:i]))
            else:
                new_txt.append(" ".join(txt1[i - 2:i]))
        else:
            k = 1
            l += 1
            new_txt.append("")
    return new_txt

def add_context(txt, docs):
    k = 0
    l = 0
    new_txt = []
    for i, line in enumerate(txt):
        k += 1
        if k <= docs[l]:
            if k == 1:
                new_txt.append(txt[i])
            elif k == 2:
                new_txt.append(" ".join(txt[i - 1:i + 1]))
            else:
                new_txt.append(" ".join(txt[i - 2:i + 1]))
        else:
            k = 1
            l += 1
            new_txt.append(txt[i])
    return new_txt


for testset, lp, ref_name, seg in [
    # ("wmt21.tedtalks", "en-de", "refA", True), ("wmt21.tedtalks", "zh-en", "refB", True),
    # ("wmt21.tedtalks", "en-ru", "refA", False),
    ("wmt21.news", "en-de", "refC", True), ("wmt21.news", "zh-en", "refB", True),
    ("wmt21.news", "en-ru", "refA", True), ("wmt21.news", "de-en", "refA", False),
    ("wmt21.news", "cs-en", "refA", False), ("wmt21.news", "ja-en", "refA", False),
    ("wmt21.news", "ru-en", "refA", False),
    # ("wmt21.news", "ha-en", "refA", False), ("wmt21.news", "is-en", "refA", False),
    ("wmt21.news", "en-zh", "refA", False), ("wmt21.news", "en-ja", "refA", False),
    ("wmt21.news", "en-cs", "refA", False), ("wmt21.news", "de-fr", "refA", False),
    ("wmt21.news", "fr-de", "refA", False)
                              ]:

    prism_dir = "/home/ubuntu/Projects/myPrism/"
    model_dir = "/home/ubuntu/Projects/myPrism/prism/m39v1/"
    file_path = prism_dir + "outs/{}/{}/".format(testset, lp)
    scores_path = prism_dir + "scores/{}_{}_{}{}srcprism_scores.json".format(testset, lp, "seg-" if seg else "",
                                                                          "ctx" if context else "")

    print(scores_path)

    prism = Prism(model_dir=model_dir, lang=lp.split("-")[-1])

    print('Prism identifier:', prism.identifier())

    scores = {level: {} for level in ['sys']}

    ref = open(file_path + ref_name, "r").readlines()
    if source:
        src = open(prism_dir + "ins/{}/{}/src".format(testset, lp), "r").read().splitlines()
    if context:
        with open(prism_dir + "docs/{}/{}/ids".format(testset, lp), "r") as fp:
            docs = json.load(fp)
        prompt = add_prompt(ref, docs)
        if source:
            src = add_context(src, docs)
    else:
        prompt = None
    for sysname in os.listdir(file_path):
        print("evaluating {}".format(sysname))
        cand = open(file_path + sysname, "r").read().splitlines()
        # if context:
        #     cand = add_context(cand, docs)
        if source:
            score = prism.score(cand, src=src, prompt=prompt, segment_scores=seg)
        else:
            score = prism.score(cand, ref=ref, prompt=prompt, segment_scores=seg)
        scores['sys'][sysname] = [float(score) if not seg else [float(x) for x in score]]
        print('System-level metric for {} is : {}'.format(sysname, score))

    with open(scores_path, 'w') as fp:
        json.dump(scores, fp)
