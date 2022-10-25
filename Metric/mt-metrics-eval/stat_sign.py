# Original Copyright 2021 Google LLC
# Modifications Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved. 

import scipy

from metric_corr import compute_corr
from mt_metrics_eval import meta_info
from mt_metrics_eval import data
from mt_metrics_eval import stats

systems = ["seg-comet21", "seg-ctxpartrefcomet21"]

for testset in ["wmt21.news", "wmt21.tedtalks"]:
    for lp in ["en-de", "zh-en", "en-ru"]:

        if testset == "wmt21.tedtalks" and lp == "en-ru":
            systems = [sys.replace("seg-", "") for sys in systems]

        corr1 = compute_corr(testset=testset, lp=lp, system=systems[0], dir="/home/user/Projects")
        corr2 = compute_corr(testset=testset, lp=lp, system=systems[1], dir="/home/user/Projects")
        pearson = corr1.GenCorrFunction(scipy.stats.pearsonr, averaged=False)

        pear_1 = corr1.Pearson()
        pear_2 = corr2.Pearson()

        print("{} - {} ".format(testset, lp))
        print("s1 = {} : {}".format(systems[0], pear_1[0]))
        print("s2 = {} : {}".format(systems[1], pear_2[0]))

        def _SigTest(corr1, corr2, v1, v2, corr_fcn):
            better = v2[0] >= v1[0]
            if not better:
                corr2, corr1 = corr1, corr2
            w = stats.WilliamsSigDiff(corr1, corr2, corr_fcn)
            p = stats.PermutationSigDiff(corr1, corr2, corr_fcn, 1000)
            return better, w, p


        pear_b, pear_w, pear_p = _SigTest(corr1, corr2, pear_1, pear_2, pearson)


        def _Summary(better, sig_williams, sig_perm):
            s = '2>1,' if better else '1>2,'
            s += f'pWilliams={sig_williams[0]:0.5f},pPERM={sig_perm:0.5f}'
            return s


        print("Pearson {}".format(_Summary(pear_b, pear_w, pear_p)))
