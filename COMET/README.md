# Doc-COMET(-QE)

This README describes how to use **Doc-COMET** an extension of the original COMET metric that can be used for document-level evaluation. This can also be applied to the referenceless version of COMET, i.e. COMET-QE (QE-as-a-metric), resulting in the corresponding **Doc-COMET-QE** metric.

## Installation

This codebase is built upon the original [COMET code](https://github.com/Unbabel/COMET). For a detailed documentation of the COMET metric, including usage examples and instructions see the [Full Documentation](https://unbabel.github.io/COMET/html/index.html).

To run Doc-COMET you will need to develop locally:
```bash
git clone https://github.com/amazon-science/doc-mt-metrics.git
cd doc-mt-metrics/COMET
pip install -r requirements.txt
pip install -e .
```

## Scoring MT outputs:

### Command Line usage:

To score using the original, sentence-level COMET/COMET-QE models:
```bash
comet-score -s src.de -t hyp1.en -r ref.en --model wmt21-comet-mqm
comet-score -s src.de -t hyp1.en --model wmt21-comet-qe-mqm
```

To evaluate at the document level we need to know the number of documents per file. This can either be a list of strings where each string is a document name, i.e. `[doc1, doc1, doc1, doc2, doc2, doc3]`, or just a list of indices, i.e. `[1, 1, 1, 2, 3, 3]`. 

For WMT test sets this can be obtained via [sacreBLEU](https://github.com/mjpost/sacrebleu):
```bash
sacrebleu -t wmt21 -l en-de --echo docid > docid.en-de
```

Next, we have to add context to each of the source, hypothesis and target files:
```bash
python add_context.py --f1 src.de --doc_ids docid.en-de
python add_context.py --f1 ref.en --doc_ids docid.en-de
python add_context.py --f1 hyp1.en --f2 ref.en --doc_ids docid.en-de
```
> the window size can be changed by setting the `--ws` flag (default=2). 
> If you don't want to overwrite the original files set the `--name` flag accordingly.

(!) Note that we use the reference context for the hypothesis in the paper.

Finally, add the `--doc` flag to the `comet-score` command:
```bash
comet-score -s src.de -t hyp1.en -r ref.en --doc --model wmt21-comet-mqm
comet-score -s src.de -t hyp1.en --doc --model wmt21-comet-qe-mqm
```
> you can set `--gpus 0` to test on CPU.

In the paper we use `wmt21-comet-mqm` and `wmt21-comet-qe-mqm` models. To select a different model from the [available COMET models/metrics](https://unbabel.github.io/COMET/html/models.html) set the `--model` flag accordingly. 

### Python usage:

In order to use Doc-COMET(-QE) with python simply add `model.set_document_level()` after loading the model.

```python
from comet import download_model, load_from_checkpoint
from add_context import add_context

with open("bert_score/example/docids.txt") as f:
    doc_ids = [line.strip() for line in f]

model_path = download_model("wmt21-comet-mqm")
model = load_from_checkpoint(model_path)

# this command allows us to encode context for document-level evaluation
model.set_document_level()

# add contexts to reference, source and hypothesis texts
src = add_context(org_txt=src, context=src, docs=doc_ids, sep_token=model.encoder.tokenizer.sep_token)
cand = add_context(org_txt=cand, context=ref, docs=doc_ids, sep_token=model.encoder.tokenizer.sep_token)
ref = add_context(org_txt=ref, context=ref, docs=doc_ids, sep_token=model.encoder.tokenizer.sep_token)

data = [{"src": x, "mt": y, "ref": z} for x, y, z in zip(src, cand, ref)]

seg_scores, sys_score = model.predict(data, batch_size=8, gpus=1)
```

## Related Publications

- [Searching for Cometinho: The Little Metric That Could -- EAMT22 Best paper award](https://aclanthology.org/2022.eamt-1.9/)

- [Are References Really Needed? Unbabel-IST 2021 Submission for the Metrics Shared Task](http://statmt.org/wmt21/pdf/2021.wmt-1.111.pdf)

- [Uncertainty-Aware Machine Translation Evaluation](https://aclanthology.org/2021.findings-emnlp.330/) 

- [COMET - Deploying a New State-of-the-art MT Evaluation Metric in Production](https://www.aclweb.org/anthology/2020.amta-user.4)

- [Unbabel's Participation in the WMT20 Metrics Shared Task](https://aclanthology.org/2020.wmt-1.101/)

- [COMET: A Neural Framework for MT Evaluation](https://www.aclweb.org/anthology/2020.emnlp-main.213)
