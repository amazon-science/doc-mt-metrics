# Original Copyright 2021 Google LLC
# Modifications Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved. 

import json
import os

import numpy as np
import scipy.stats
from mt_metrics_eval import meta_info
from mt_metrics_eval import data
from mt_metrics_eval import stats

all_evs = {}  # name/lp -> evs
testset = 'wmt21.tedtalks'
lp = "zh-en"
level = 'sys'

evs = data.EvalSet(testset, lp, True)
name = f'{testset}/{lp}'

nsegs = len(evs.src)
nsys = len(evs.sys_names)
nmetrics = len(evs.metric_basenames)
gold = evs.StdHumanScoreName(level)
nrefs = len(evs.ref_names)
std_ref = evs.std_ref

print(f'{name:<20} {nsegs:5d} {nsys:3d} {nmetrics:7d} '
      f'{gold:5} {nrefs:4d} {std_ref}')

# ind_list = [x is not None for x in evs._scores["seg"]["mqm"][std_ref]]
# file_path = "/home/user/Projects/index/{}/{}/".format(testset, lp)
# os.makedirs(file_path ,exist_ok=True)
# with open(file_path + "inds", "w") as f:
#     for row in ind_list:
#         f.write(str(row)+"\n")

# file_path = "/home/user/Projects/mqm_scores/"
# os.makedirs(file_path, exist_ok=True)
# with open(file_path + "{}_{}_mqm_scores.json".format(testset, lp), "w") as f:
#     json.dump(evs._scores["seg"]["mqm"], f)

corrs = data.GetCorrelations(
    evs=evs,
    level=level,
    main_refs={"refA"},
    # which references will be exluded when we include HT in DA
    close_refs=set(),
    include_human=False,
    include_outliers=False,
    gold_name=evs.StdHumanScoreName(level),
    # gold_name= "mqm",
    primary_metrics=False
)

# Compute and print Pearson correlations.
if level == "sys":
    pearsons = {m: corr.Pearson()[0] for m, corr in corrs.items()}
    pearsons = dict(sorted(pearsons.items(), key=lambda x: -x[1]))
    print('System-level Pearson correlations for {} {}:'.format(testset, lp))
    for m in pearsons:
        print(f'{m:<21} {pearsons[m]: f}')
    print()
elif level == "seg":
    kendalls = {m: corr.Kendall()[0] for m, corr in corrs.items()}
    kendalls = dict(sorted(kendalls.items(), key=lambda x: -x[1]))
    print('Segment-level Kendall correlations for {} {}:'.format(testset, lp))
    for m in kendalls:
        print(f'{m:<21} {kendalls[m]: f}')
    print()
else:
    print("The end")
