# Original Copyright (C) 2020 Unbabel
# Modifications Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
from comet import download_model, load_from_checkpoint

context = True


def add_contexts(txt, contxt):
    new_contxt = []
    for i, line in enumerate(contxt):
        if i > 0:
            if i % 2 == 0:
                continue
            else:
                new_contxt.append(" </s> ".join([contxt[i - 1], line]))
    new_txt = [" </s> ".join([contxt_line, txt_line]) for contxt_line, txt_line in zip(new_contxt, txt)]
    return new_txt


lp = "en-fr"

cp_dir = "/home/ubuntu/Projects/Large-contrastive-pronoun-testset-EN-FR/OpenSubs/"

model_path = download_model("wmt21-comet-qe-mqm")
model = load_from_checkpoint(model_path)
if context:
    model.hparams.pool = "part_avg"

scores_path = cp_dir + "scores/contrapro_{}comet21-qe_scores.txt".format("ctxpart" if context else "")

print(scores_path)

scores = {level: {} for level in ['sys']}

src = open(cp_dir + "contrapro.{}.{}".format("text" if "de" in lp else "current",
                                             lp.split("-")[0] if "de" in lp else "src"), "r").read().splitlines()
ref = open(cp_dir + "contrapro.{}.{}".format("text" if "de" in lp else "current",
                                             lp.split("-")[-1] if "de" in lp else "trg"), "r").read().splitlines()
if context:
    ctx = open(cp_dir + "contrapro.context.{}".format(lp.split("-")[0] if "de" in lp else "src"),
               "r").read().splitlines()
    ref_ctx = open(cp_dir + "contrapro.context.{}".format(lp.split("-")[-1] if "de" in lp else "trg"),
                   "r").read().splitlines()
    src = add_contexts(src, ctx)
    ref = add_contexts(ref, ref_ctx)
data = [{"src": x, "mt": y} for x, y in zip(src, ref)]
seg_score, _ = model.predict(data, batch_size=32, gpus=1)

with open(scores_path, 'w') as fp:
    for line in seg_score:
        fp.write(str(line))
        fp.write('\n')
