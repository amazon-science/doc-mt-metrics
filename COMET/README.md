# Doc-COMET(-QE)

This README describes how to use **Doc-COMET** an extension of the original COMET metric that can be used for document-level evaluation. This can also be applied to the referenceless version of COMET, i.e. COMET-QE (QE-as-a-metric), resulting in the corresponding **Doc-COMET-QE** metric.

## Installation

This codebase is built upon the original [COMET code](https://github.com/Unbabel/COMET). For a detailed documentation of the COMET metric, including usage examples and instructions see the [Full Documentation](https://unbabel.github.io/COMET/html/index.html).

To run Doc-COMET you will need to develop locally:
```bash
git clone https://github.com/amazon-science/doc-mt-metrics.git
cd doc-mt-metrics/COMET
conda create -n doc-metrics-env python=3.9 
conda activate doc-metrics-env
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### Get some files to score
```bash
sacrebleu -t wmt21 -l en-de --echo src | head -n 20 > src.en
sacrebleu -t wmt21 -l en-de --echo ref | head -n 20 > ref.de
sacrebleu -t wmt21 -l en-de --echo ref | head -n 20 > hyp.de  # put your system output here
```

To evaluate at the document level we need to know where the document boundaries are in the test set, so that we only use valid context. This is passed in as a file where each line contains a document ID.

For WMT test sets this can be obtained via [sacreBLEU](https://github.com/mjpost/sacrebleu):
```bash
sacrebleu -t wmt21 -l en-de --echo docid | head -n 20 > docids.ende
```

### Command Line usage

Comet and comet-qe are run just as before, except we, add the `--doc` flag to the `comet-score` command:
```bash
comet-score -s src.en -t hyp.de -r ref.de --doc docids.ende --model wmt21-comet-mqm
comet-score -s src.en -t hyp.de --doc docids.ende --model wmt21-comet-qe-mqm
```
> Note: you can set `--gpus 0` to run on CPU.

In the paper we use `wmt21-comet-mqm` and `wmt21-comet-qe-mqm` models. To select a different model from the [available COMET models/metrics](https://unbabel.github.io/COMET/html/models.html) set the `--model` flag accordingly. 

### Python usage:

In order to use Doc-COMET(-QE) with python simply add `model.set_document_level()` after loading the model.

```python
from comet import download_model, load_from_checkpoint
from add_context import add_context

# load data files
doc_ids = [x.strip() for x in open('docids.ende', 'rt').readlines()]
src = [x.strip() for x in open('src.en', 'rt').readlines()]
hyp = [x.strip() for x in open('hyp.de', 'rt').readlines()]
ref = [x.strip() for x in open('ref.de', 'rt').readlines()]

# load comet model
model_path = download_model("wmt21-comet-mqm")
model = load_from_checkpoint(model_path)

# enable document-level evaluation
model.set_document_level()

# add contexts to reference, source and hypothesis texts
src = add_context(orig_txt=src, context=src, doc_ids=doc_ids, sep_token=model.encoder.tokenizer.sep_token)
hyp = add_context(orig_txt=hyp, context=ref, doc_ids=doc_ids, sep_token=model.encoder.tokenizer.sep_token)
ref = add_context(orig_txt=ref, context=ref, doc_ids=doc_ids, sep_token=model.encoder.tokenizer.sep_token)

data = [{"src": x, "mt": y, "ref": z} for x, y, z in zip(src, hyp, ref)]

seg_scores, sys_score = model.predict(data, batch_size=8, gpus=1)
```

## Reproduce
Follow the [instructions](reproduce/) to reproduce the Doc-COMET(-QE) results from the paper.

## Paper

If you use the code in your work, please cite [Embarrassingly Easy Document-Level MT Metrics: How to Convert Any Pretrained Metric Into a Document-Level Metric](https://statmt.org/wmt22/pdf/2022.wmt-1.6.pdf):

```
@inproceedings{easy_doc_mt
    title = {Embarrassingly Easy Document-Level MT Metrics: How to Convert Any Pretrained Metric Into a Document-Level Metric},
    author = {Vernikos, Giorgos and Thompson, Brian and Mathur, Prashant and Federico, Marcello},
    booktitle = "Proceedings of the Seventh Conference on Machine Translation",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://statmt.org/wmt22/pdf/2022.wmt-1.6.pdf",
}
```
