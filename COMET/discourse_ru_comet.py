# Original Copyright (C) 2020 Unbabel
# Modifications Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
from comet import download_model, load_from_checkpoint

context = True

lp = "en-ru"

cp_dir = "/home/ubuntu/Projects/good-translation-wrong-in-context/consistency_testsets/scoring_data/"

test = "deixis"

model_path = download_model("wmt21-comet-qe-mqm")
model = load_from_checkpoint(model_path)
if context:
    model.hparams.pool = "part_avg"

scores_path = cp_dir + "scores/{}_{}comet21-qe_scores.txt".format(test, "ctxpart" if context else "")

print(scores_path)

scores = {level: {} for level in ['sys']}

src = open(cp_dir + "{}{}.src".format(test, "_test" if test in ["deixis", "lex_cohesion"] else ""), "r").read().splitlines()
ref = open(cp_dir + "{}{}.dst".format(test, "_test" if test in ["deixis", "lex_cohesion"] else ""), "r").read().splitlines()
if context:
    src = [sent.replace("_eos", "</s>") for sent in src]
    ref = [sent.replace("_eos", "</s>") for sent in ref]
else:
    src = [sent.split("_eos")[-1] for sent in src]
    ref = [sent.split("_eos")[-1] for sent in ref]
data = [{"src": x, "mt": y} for x, y in zip(src, ref)]
seg_score, _ = model.predict(data, batch_size=32, gpus=1)

with open(scores_path, 'w') as fp:
    for line in seg_score:
        fp.write(str(-1*line))
        fp.write('\n')
