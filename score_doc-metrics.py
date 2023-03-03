# Original Copyright (C) 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
import argparse
from bert_score.bert_score.scorer import BERTScorer
from COMET.add_context import add_context
from COMET.comet import download_model, load_from_checkpoint
import json
from mt_metrics_eval.data import EvalSet
import numpy as np
from Prism.prism import MBARTPrism
import torch


def main(args):
    evs = EvalSet(args.campaign, args.lp)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if args.model == "comet":
        model_name = "wmt21-comet-mqm" if not args.qe else "wmt21-comet-qe-mqm"
        model_path = download_model(model_name)
        model = load_from_checkpoint(model_path)
        sep_token = model.encoder.tokenizer.sep_token
        if args.doc:
            model.set_document_level()
    elif args.model == "bertscore":
        scorer = BERTScorer(lang=args.lp.split("-")[-1], rescale_with_baseline=False)
        sep_token = scorer._tokenizer.sep_token
    elif args.model == "prism":
        model_path = "facebook/mbart-large-50"
        model = MBARTPrism(checkpoint=model_path, src_lang=args.lp.split("-")[-1], tgt_lang=args.lp.split("-")[-1],
                           device=device)
        sep_token = model.tokenizer.sep_token

    scores = {level: {} for level in [args.level]}

    src = evs.src
    orig_ref = evs.all_refs[evs.std_ref]

    # keep scores only for the segments that have mqm annotations
    mqm_annot = [True] * len(src)
    if 'no-mqm' in evs.domains:
        for seg_start, seg_end in evs.domains['no-mqm']:
            mqm_annot[seg_start:seg_end] = (seg_end - seg_start) * [False]

    if args.doc:
        doc_ids = [""] * len(src)
        # get the document ids in a suitable format
        for doc_name, doc_bounds in evs.docs.items():
            doc_start, doc_end = doc_bounds
            doc_ids[doc_start:doc_end] = (doc_end - doc_start) * [doc_name]

        # add contexts to source and reference texts
        src = add_context(orig_txt=src, context=src, doc_ids=doc_ids, sep_token=sep_token)
        if not args.qe:
            ref = add_context(orig_txt=orig_ref, context=orig_ref, doc_ids=doc_ids, sep_token=sep_token)
    else:
        if not args.qe:
            ref = orig_ref

    for sysname, cand in evs.sys_outputs.items():

        print(f'----Evaluating {sysname}----')

        if args.doc:
            # add contexts to hypotheses text
            if args.qe:
                cand = add_context(orig_txt=cand, context=cand, doc_ids=doc_ids, sep_token=sep_token)
            else:
                cand = add_context(orig_txt=cand, context=orig_ref, doc_ids=doc_ids, sep_token=sep_token)

        if args.model == "comet":
            if args.qe:
                data = [{"src": x, "mt": y} for x, y in zip(src, cand)]
            else:
                data = [{"src": x, "mt": y, "ref": z} for x, y, z in zip(src, cand, ref)]
            seg_score, _ = model.predict(data, batch_size=32, gpus=1)
        elif args.model == "bertscore":
            P, R, F1 = scorer.score(cand, ref, doc=args.doc)
            seg_score = F1.cpu().numpy()
        elif args.model == "prism":
            seg_score = model.score(cand=cand, ref=ref, doc=args.doc, batch_size=2, segment_scores=True)

        seg_score = [float(x) if mqm_annot[i] else None for i, x in enumerate(seg_score)]
        scores[args.level][sysname] = [
            np.mean(np.array(seg_score)[mqm_annot]) if args.level == 'sys' else seg_score]

    if args.save:
        scores_file = "{}{}{}_scores.json".format("doc-" if args.doc else "", args.model, "-qe" if args.qe else "")
        with open(scores_file, 'w') as fp:
            json.dump(scores, fp)

    gold_scores = evs.Scores(args.level, evs.StdHumanScoreName(args.level))
    sys_names = set(gold_scores) - evs.human_sys_names
    corr = evs.Correlation(gold_scores, scores[args.level], sys_names)

    if args.level == "sys":
        print(f'system: Pearson={corr.Pearson()[0]:f}')
    else:
        print(f'segment: Kendall-like={corr.KendallLike()[0]:f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reproduce document-level metric scores from the paper.')
    parser.add_argument('--campaign', choices=['wmt21.news', 'wmt21.tedtalks'], default='wmt21.news',
                        type=str, help='the wmt campaign to test on')
    parser.add_argument('--lp', choices=['en-de', 'zh-en', 'en-ru'], default='en-de',
                        type=str, help='the language pair')
    parser.add_argument('--model', choices=['comet', 'bertscore', 'prism'], default='comet',
                        type=str, help='the model/metric to be tested')
    parser.add_argument('--doc', action="store_true", help='document- or sentence-level comet')
    parser.add_argument('--qe', action="store_true",
                        help='(only for comet) whether evaluation is reference-based or reference-free (qe)')
    parser.add_argument('--level', required=False, default="sys", choices=["seg", "sys"],
                        help='whether segment-level or system-level scores will be computed')
    parser.add_argument('--save', action="store_true", help='save scores in a file')

    args = parser.parse_args()

    main(args)
