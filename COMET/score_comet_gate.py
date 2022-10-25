# Original Copyright (C) 2020 Unbabel
# Modifications Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.

import argparse
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import json
from comet import download_model, load_from_checkpoint

use_ref = False

def create_context(txt, docs):
    k = 0
    l = 0
    new_txt = []
    for i, line in enumerate(txt):
        k += 1
        if k <= docs[l]:
            if k == 1:
                new_txt.append(" </s> ")
            elif k == 2:
                new_txt.append(" </s> ".join(txt[i - 1:i]))
            else:
                new_txt.append(" </s> ".join(txt[i - 2:i]))
        else:
            k = 1
            l += 1
            new_txt.append(" </s> ")
    return new_txt


for testset, lp, ref_name, seg in [
                                   ("wmt21.tedtalks", "en-de", "refA", True),
                                    ("wmt21.tedtalks", "zh-en", "refB", True),
                                   ("wmt21.tedtalks", "en-ru", "refA", False),
                                    ("wmt21.news", "en-de", "refC", True), ("wmt21.news", "zh-en", "refB", True),
                                    ("wmt21.news", "en-ru", "refA", True),
                                    # ("wmt21.news", "de-en", "refA", False), ("wmt21.news", "cs-en", "refA", False),
                                    # ("wmt21.news", "ja-en", "refA", False), ("wmt21.news", "ru-en", "refA", False),
                                    # ("wmt21.news", "ha-en", "refA", False), ("wmt21.news", "is-en", "refA", False),
                                    # ("wmt21.news", "en-zh", "refA", False), ("wmt21.news", "en-ja", "refA", False),
                                    # ("wmt21.news", "en-cs", "refA", False),  ("wmt21.news", "de-fr", "refA", False),
                                    #   ("wmt21.news", "fr-de", "refA", False)
                                   ]:

    print("Campaign: {}\n Language Pair: {}\n".format(testset, lp))

    prism_dir = "/home/ubuntu/Projects/myPrism/"

    model_name = "wmt21-comet-mqm" if use_ref else "wmt21-comet-qe-mqm"
    model_path = download_model(model_name)
    # model_path = "/home/ubuntu/Projects/COMET/lightning_logs/mycomet_fp16_v3/checkpoints/epoch=1-step=84162.ckpt"
    model = load_from_checkpoint(model_path)

    file_path = prism_dir + "outs/{}/{}/".format(testset, lp)
    scores_path = prism_dir + "scores/{}_{}_{}comet21{}_scores.json".format(testset, lp, "seg-" if seg else "",
                                                                                "-qe" if not use_ref else "")

    print(scores_path)

    scores = {level: {} for level in ['sys']}

    src = open(prism_dir + "ins/{}/{}/src".format(testset, lp), "r").read().splitlines()
    ref = open(file_path + ref_name, "r").read().splitlines()
    with open(prism_dir + "docs/{}/{}/ids".format(testset, lp), "r") as fp:
        docs = json.load(fp)
    for sysname in os.listdir(file_path):
        print("evaluating {}".format(sysname))
        cand = open(file_path + sysname, "r").read().splitlines()
        ctx = create_context(src, docs)
        data = [{"src": x, "ctx": y, "mt": z} for x, y, z in zip(src, ctx, cand)]
        seg_score, sys_score = model.predict(data, batch_size=32, gpus=1)
        scores['sys'][sysname] = [float(sys_score) if not seg else [float(x) for x in seg_score]]
        print('System-level metric for {} is : {}'.format(sysname, sys_score))

    with open(scores_path, 'w') as fp:
        json.dump(scores, fp)
