# Original Copyright 2021 Google LLC
# Modifications Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved. 

import numpy as np
import scipy.stats
from mt_metrics_eval import meta_info
from mt_metrics_eval import data
from mt_metrics_eval import stats
import os
import json

src = True

file_path = "/home/user/Projects/outs/"
docs_path = "/home/user/Projects/docs/"
src_file_path = "/home/user/Projects/ins/"

all_evs = {}  # name/lp -> evs
testset = 'wmt20'
lp = "zh-en"
file_path = file_path + "{}/{}/".format(testset, lp)
evs = data.EvalSet(testset, lp, True)
name = f'{testset}/{lp}'

nsegs = len(evs.src)
nsys = len(evs.sys_names)
nmetrics = len(evs.metric_basenames)
gold = evs.StdHumanScoreName('sys')
nrefs = len(evs.ref_names)
std_ref = evs.std_ref

os.makedirs(file_path, exist_ok=True)
print(f'{name:<20} {nsegs:5d} {nsys:3d} {nmetrics:7d} '
      f'{gold:5} {nrefs:4d} {std_ref}')

doc_dir = docs_path + "/{}/{}/".format(testset, lp)
os.makedirs(doc_dir, exist_ok=True)
docs = [int(np.diff(x)[0]) for x in list(evs.docs.values())]
with open(doc_dir + "ids", "w") as fp:
    json.dump(docs, fp)

if src:
    src_dir = src_file_path + "/{}/{}/".format(testset, lp)
    os.makedirs(src_dir, exist_ok=True)
    with open(src_dir + "src".format(testset, lp), 'w') as outfile:
        for row in evs.src:
            outfile.write(str(row) + '\n')

for name, out_sys in evs.sys_outputs.items():
    with open(f"{file_path}{name}", 'w') as outfile:
        for row in out_sys:
            outfile.write(str(row) + '\n')

print("The end")
