# Doc-BERTScore

This README describes hot to use **Doc-BERTScore** an extension of the BERTScore metric that can be used for document-level evaluation.  

## Installation

This codebase is built upon the original [BERTScore code](https://github.com/Tiiiger/bert_score). For a detailed presnetation of the BERTScore metric, including usage examples and instructions see the original documentation.

To run Doc-BERTScore you will need to develop locally:
```bash
git clone https://github.com/amazon-science/doc-mt-metrics.git
cd BERTScore
pip install .
```

### Get some files to score
```bash
sacrebleu -t wmt21 -l en-de --echo ref | head -n 20 > ref.de
sacrebleu -t wmt21 -l en-de --echo ref | head -n 20 > hyp.de  # put your system output here
```
To evaluate at the document level we need to know where the document boundaries are in the test set, so that we only use valid context. This is passed in as a file where each line contains a document ID.

For WMT test sets this can be obtained via [sacreBLEU](https://github.com/mjpost/sacrebleu):
```bash
sacrebleu -t wmt21 -l en-de --echo docid | head -n 20 > docids.ende
```
Next, we have to add context to each of the source, hypothesis and target files:
```bash
python add_context.py --f1 ref.en --doc_ids docids.ende
python add_context.py --f1 hyp1.en --f2 ref.en --doc_ids docids.ende
```
> the window size can be changed by setting the `--ws` flag (default=2). 
> If you don't want to overwrite the original files set the `--name` flag accordingly.

(!) Note that we use the reference context for the hypothesis in the paper.

### Command Line usage:

To score using the document-level BERTScore simply add the `--doc` flag:
```bash
bert-score -r ref.en -c hyp1.en --lang en --doc
```

In the paper we use`roberta-large` for X->En pairs and `bert-base-multilingual-cased` for En->X pairs (default at the time) but you can select another model with the `-m MODEL_TYPE` flag. See the [spreadsheet](https://docs.google.com/spreadsheets/d/1RKOVpselB98Nnh_EOC4A2BYn8_201tmPODpNWu4w7xI/edit?usp=sharing) provided by the authors of BERTScore for a full list of supported models.

### Python usage (Object-oriented API):

The [BERTScore](https://github.com/Tiiiger/bert_score) framework provides two APIs in order to use the BERTScore metric with python: an object-oriented one that caches the model and is recommended for multiple evaluations and a functional one that can be used for single evaluation. For more details see the [demo](https://github.com/Tiiiger/bert_score/blob/master/example/Demo.ipynb) provided by the authors.

In order to use Doc-BERTScore simple simply add `doc=True` when calling the `score` function:

```python
from bert_score import BERTScorer

with open("hyp.de") as f:
    cands = [line.strip() for line in f]

with open("ref.de") as f:
    refs = [line.strip() for line in f]
    
with open("docids.ende") as f:
    doc_ids = [line.strip() for line in f]

scorer = BERTScorer(lang="en", rescale_with_baseline=True)

# add contexts to reference and hypothesis texts
cands = add_context(orig_txt=cand, context=ref, doc_ids=doc_ids, sep_token=scorer._tokenizer.sep_token)
refs = add_context(orig_txt=ref, context=ref, doc_ids=doc_ids, sep_token=scorer._tokenizer.sep_token)

# set doc=True to evaluate at the document level
P, R, F1 = scorer.score(cands, refs, doc=True)
```
### Python usage (Function API):

In order to use Doc-BERTScore simple simply add `doc=True` when calling the `score` function:

```python
from bert_score import score

with open("hyp.de") as f:
    cands = [line.strip() for line in f]

with open("ref.de") as f:
    refs = [line.strip() for line in f]

with open("docids.ende") as f:
    doc_ids = [line.strip() for line in f]
    
# add contexts to reference and hypothesis texts
cands = add_context(orig_txt=cand, context=ref, doc_ids=doc_ids, sep_token="</s>")
refs = add_context(orig_txt=ref, context=ref, doc_ids=doc_ids, sep_token="</s>")

# set doc=True to evaluate at the document level
P, R, F1 = score(cands, refs, lang="en", verbose=True, doc=True)
```

To use another model set the flag `model_type=MODEL_TYPE` when calling `score` function.

## Reproduce
Follow the [instructions](reproduce/) to reproduce the Doc-BERTScore results from the paper.

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
