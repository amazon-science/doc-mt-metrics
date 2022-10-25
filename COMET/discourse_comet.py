# Original Copyright (C) 2020 Unbabel
# Modifications Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
from comet import download_model, load_from_checkpoint

context = False


def add_contexts(txt, contxt):
    new_txt = [" </s> ".join([contxt_line, txt_line]) for contxt_line, txt_line in zip(contxt, txt)]
    return new_txt


lp = "en-fr"

cp_dir = "/home/ubuntu/Projects/discourse-mt-test-sets/test-sets/"

test = "lexical_choice"

model_path = download_model("wmt21-comet-qe-mqm")
model = load_from_checkpoint(model_path)
if context:
    model.hparams.pool = "part_avg"

scores_path = cp_dir + "scores/{}_{}comet21-qe_scores.txt".format(test, "ctxpart" if context else "")

print(scores_path)

scores = {level: {} for level in ['sys']}

src = open(cp_dir + "{}.current.{}".format(test, lp.split("-")[0]), "r").read().splitlines()
ref = open(cp_dir + "{}.current.{}".format(test, lp.split("-")[-1]), "r").read().splitlines()
if context:
    ctx = open(cp_dir + "{}.prev.{}".format(test, lp.split("-")[0]), "r").read().splitlines()
    ref_ctx = open(cp_dir + "{}.prev.{}".format(test, lp.split("-")[-1]), "r").read().splitlines()
    src = add_contexts(src, ctx)
    ref = add_contexts(ref, ref_ctx)
data = [{"src": x, "mt": y} for x, y in zip(src, ref)]
seg_score, _ = model.predict(data, batch_size=32, gpus=1)

with open(scores_path, 'w') as fp:
    for line in seg_score:
        fp.write(str(line))
        fp.write('\n')
