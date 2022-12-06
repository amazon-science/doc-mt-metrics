# Original Copyright (C) 2020 Unbabel
# Modifications Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.

import argparse
import json
from add_context import add_context
from comet import download_model, load_from_checkpoint


pair2ref = {("wmt21.tedtalks", "en-de"): "refA", ("wmt21.tedtalks", "zh-en"): "refB",
               ("wmt21.tedtalks", "en-ru"): "refA",
               ("wmt21.news", "en-de"): "refC", ("wmt21.news", "zh-en"): "refB",
               ("wmt21.news", "en-ru"): "refA"}

# the code assumes the following directory structure:
# src file: dir/src
# hypotheses files: dir/hyps/*
# reference(s) file: dir/ref (see pair2ref)

def main(args):
    model_name = "wmt21-comet-mqm" if args.use_ref else "wmt21-comet-qe-mqm"
    model_path = download_model(model_name)
    model = load_from_checkpoint(model_path)
    if args.doc:
        model.set_document_level()

    scores_path = args.dir + "{}comet21{}_scores.json".format("doc-" if args.context else "",
                                                                     "-qe" if not args.use_ref else "")

    scores = {level: {} for level in [args.level]}

    src = open(args.dir + "/src", "r").read().splitlines()
    if args.use_ref:
        ref = open(args.dir + "/ref", "r").read().splitlines()

    for sysname in os.listdir(args.dir + "/hyps/"):
        cand = open(args.dir + "/hyps/" + sysname, "r").read().splitlines()

        if args.doc:
            doc_ids = open(args.dir + "/docids", "r").read().splitlines()

            # add contexts to reference and source texts
            src = add_context(orig_txt=src, context=src, doc_ids=doc_ids, sep_token=model.encoder.tokenizer.sep_token)

            if args.use_ref:
                cand = add_context(orig_txt=cand, context=ref, doc_ids=doc_ids, sep_token=model.encoder.tokenizer.sep_token)
                ref = add_context(orig_txt=ref, context=ref, doc_ids=doc_ids, sep_token=model.encoder.tokenizer.sep_token)
            else:
                cand = add_context(orig_txt=cand, context=cand, doc_ids=doc_ids, sep_token=model.encoder.tokenizer.sep_token)

        if args.use_ref:
            data = [{"src": x, "mt": y, "ref": z} for x, y, z in zip(src, cand, ref)]
        else:
            data = [{"src": x, "mt": y} for x, y in zip(src, cand)]

        seg_score, sys_score = model.predict(data, batch_size=32, gpus=1)
        scores[args.level][sysname] = [float(sys_score) if not args.seg else [float(x) for x in seg_score]]

    with open(scores_path, 'w') as fp:
        json.dump(scores, fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Score COMET models.')
    parser.add_argument('--doc', action="store_true", help='document- or sentence-level comet')
    parser.add_argument('--dir', default="", type=str, help='directory where the files are located')
    parser.add_argument('--use_ref', required=False, default=True,
                        help='whether evaluation is reference-based or reference-free')
    parser.add_argument('--level', required=False, default="sys", choices=["seg", "sys"],
                        help='whether segment-level or system-level scores will be computed')

    args = parser.parse_args()

    main(args)
