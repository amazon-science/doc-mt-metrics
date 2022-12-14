import argparse
import json
import os
from prism import MBARTPrism

device = "cuda:7"

pair2ref = {("wmt21.tedtalks", "en-de"): "refA", ("wmt21.tedtalks", "zh-en"): "refB",
            ("wmt21.tedtalks", "en-ru"): "refA",
            ("wmt21.news", "en-de"): "refC", ("wmt21.news", "zh-en"): "refB",
            ("wmt21.news", "en-ru"): "refA", ("wmt21.news", "de-en"): "refA",
            ("wmt21.news", "cs-en"): "refA", ("wmt21.news", "ja-en"): "refA",
            ("wmt21.news", "ru-en"): "refA", ("wmt21.news", "ha-en"): "refA",
            ("wmt21.news", "is-en"): "refA", ("wmt21.news", "en-zh"): "refA",
            ("wmt21.news", "en-ja"): "refA", ("wmt21.news", "en-cs"): "refA",
            ("wmt21.news", "de-fr"): "refA", ("wmt21.news", "fr-de"): "refA"}


def main(args):
    testset = args.testset
    lp = args.lp
    ref_name = pair2ref[(testset, lp)]

    print("Campaign: {}\n Language Pair: {}\n".format(testset, lp))

    model_path = "facebook/mbart-large-50"
    file_path = args.dir + "outs/{}/{}/".format(testset, lp)
    scores_path = args.dir + "scores/{}_{}_{}{}mbart50prism_scores.json".format(testset, lp, "seg-" if args.seg else "",
                                                                                "ctx" if args.context else "", )

    print(scores_path)

    prism = MBARTPrism(checkpoint=model_path, src_lang=lp.split("-")[-1] if not args.use_ref else lp.split("-")[0],
                       tgt_lang=lp.split("-")[-1], device=device)

    scores = {level: {} for level in ['sys']}

    if args.use_ref:
        src = open(args.dir + "ins/{}/{}/src".format(testset, lp), "r").read().splitlines()
    ref = open(file_path + ref_name, "r").read().splitlines()
    if args.context:
        # add context to reference (and source) text
        with open(args.dir + "docs/{}/{}/ids".format(testset, lp), "r") as fp:
            docs = json.load(fp)
        ref = add_context(ref, docs)
        if args.use_ref:
            src = add_context(src, docs)
    for sysname in os.listdir(file_path):
        print("evaluating {}".format(sysname))
        cand = open(file_path + sysname, "r").read().splitlines()
        if args.context:
            # add conntext to the hypothesis
            org_ref = open(file_path + ref_name, "r").read().splitlines()
            cand = add_context(cand, docs, org_ref)
        if args.use_ref:
            score = prism.score_src(src=src, tgt=cand, context=args.context, batch_size=2,
                                    segment_scores=args.seg)
        else:
            score = prism.score(cand=cand, ref=ref, context=args.context, batch_size=2,
                                segment_scores=args.seg)
        scores['sys'][sysname] = [float(score) if not args.seg else [float(x) for x in score]]
        print('System-level metric for {} is : {}'.format(sysname, score))

    with open(scores_path, 'w') as fp:
        json.dump(scores, fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Score Prism (mBART50) models.')
    parser.add_argument('--testset', required=False, default="wmt21.news", choices=['wmt21.news', 'wmt21.tedtalks'],
                        help='name of wmt campaign')
    parser.add_argument('--lp', required=False, default="en-de", help='language pair')
    parser.add_argument('--dir', required=False, default="/home/ubuntu/Projects/myPrism/",
                        help='directory where the scores will be saved')
    parser.add_argument('--seg', required=False, default=False,
                        help='if segment-level or system-level scores will be stored')
    parser.add_argument('--use_ref', required=False, default=True,
                        help='whether evaluation is reference-based or reference-free')
    parser.add_argument('--context', required=False, default=True, help='document- or sentence-level comet')

    args = parser.parse_args()

    main(args)
