# Original Copyright (C) 2020 Unbabel
# Modifications Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.

import argparse
import json
from add_context import add_context
from comet import download_model, load_from_checkpoint
from mt_metrics_eval.data import EvalSet
import numpy as np


def main(args):
    evs = EvalSet(args.campaign, args.lp)
    model_name = "wmt21-comet-mqm" if args.use_ref else "wmt21-comet-qe-mqm"
    model_path = download_model(model_name)
    model = load_from_checkpoint(model_path)
    if args.doc:
        model.set_document_level()

    scores = {level: {} for level in [args.level]}

    src = evs.src
    ref = evs.all_refs[evs.std_ref]

    for cand, sysname in evs.sys_outputs.items():

        if args.doc:
            doc_ids = [""] * len(src)
            # get the document ids in a suitable format
            for doc_name, doc_bounds in evs.docs.items():
                doc_start, doc_end = doc_bounds
                doc_ids[doc_start:doc_end] = (doc_end - doc_start) * [doc_name]

            # add contexts to reference and source texts
            src = add_context(orig_txt=src, context=src, doc_ids=doc_ids, sep_token=model.encoder.tokenizer.sep_token)

            if args.use_ref:
                cand = add_context(orig_txt=cand, context=ref, doc_ids=doc_ids,
                                   sep_token=model.encoder.tokenizer.sep_token)
                ref = add_context(orig_txt=ref, context=ref, doc_ids=doc_ids,
                                  sep_token=model.encoder.tokenizer.sep_token)
            else:
                cand = add_context(orig_txt=cand, context=cand, doc_ids=doc_ids,
                                   sep_token=model.encoder.tokenizer.sep_token)

        if args.use_ref:
            data = [{"src": x, "mt": y, "ref": z} for x, y, z in zip(src, cand, ref)]
        else:
            data = [{"src": x, "mt": y} for x, y in zip(src, cand)]

        seg_score, _ = model.predict(data, batch_size=32, gpus=1)

        # keep scores only for the segments that have mqm annotations
        mqm_annot = [True] * len(seg_score)
        if 'no-mqm' in evs.domains:
            for seg_start, seg_end in evs.domains['no_mqm']:
                mqm_annot[seg_start:seg_end] = (seg_end - seg_start) * [False]
        seg_score = [float(x) if mqm_annot[i] else None for i, x in enumerate(seg_score)]
        scores[args.level][sysname] = [
            np.mean(np.array(seg_score)[mqm_annot]) if not args.level == 'sys' else seg_score]

    if args.save:
        scores_file = "{}comet21{}_scores.json".format("doc-" if args.doc else "", "-qe" if not args.use_ref else "")
        with open(scores_file, 'w') as fp:
            json.dump(scores, fp)

    gold_scores = evs.Scores(args.level, evs.StdHumanScoreName(args.level))
    sys_names = set(gold_scores) - evs.human_sys_names
    corr = evs.Correlation(gold_scores, scores[args.level], sys_names)
    print(f'{args.level}: Pearson={corr.Pearson()[0]:f}, '
          f'Kendall-like={corr.KendallLike()[0]:f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reproduce Doc-COMET(-QE) scores from the paper.')
    parser.add_argument('--campaign', choices=['wmt21.news', 'wmt21.tedtalks'], default='wmt21.news',
                        help='the wmt campaign to test on')
    parser.add_argument('--lp', choices=['en-de', 'zh-en', 'en-ru'], default='en-de',
                        help='the language pair')
    parser.add_argument('--doc', action="store_true", help='document- or sentence-level comet')
    parser.add_argument('--use_ref', required=False, default=True,
                        help='whether evaluation is reference-based or reference-free')
    parser.add_argument('--level', required=False, default="sys", choices=["seg", "sys"],
                        help='whether segment-level or system-level scores will be computed')
    parser.add_argument('--save', action="store_true", help='save scores in a file')

    args = parser.parse_args()

    main(args)
