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
## Scoring MT outputs:

### Command Line usage:

To score using the original, sentence-level BERTScore model:
```bash
bert-score -r ref.en -c hyp1.en --lang en
```
To evaluate at the document level we need to know the number of documents per file. This can either be a list of strings where each string is a document name, i.e. `[doc1, doc1, doc1, doc2, doc2, doc3]`, or just a list of indices, i.e. `[1, 1, 1, 2, 3, 3]`. 

For WMT test sets this can be obtained via [sacreBLEU](https://github.com/mjpost/sacrebleu):
```bash
sacrebleu -t wmt21 -l en-de --echo docid > docid.en-de
```

Next, we have to add context to each of the source, hypothesis and target files:
```bash
python add_context.py --f1 ref.en --doc_ids docid.en-de
python add_context.py --f1 hyp1.en --f2 ref.en --doc_ids docid.en-de
```
> the window size can be changed by setting the `--ws` flag (default=2). 
> If you don't want to overwrite the original files set the `--name` flag accordingly.

(!) Note that we use the reference context for the hypothesis in the paper.

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

with open("bert_score/example/hyps.txt") as f:
    cands = [line.strip() for line in f]

with open("bert_score/example/refs.txt") as f:
    refs = [line.strip() for line in f]
    
with open("bert_score/example/docids.txt") as f:
    doc_ids = [line.strip() for line in f]
    
# add contexts to reference and hypothesis texts
cands = add_context(org_txt=cand, context=ref, docs=doc_ids)
refs = add_context(org_txt=ref, context=ref, docs=doc_ids)

scorer = BERTScorer(lang="en", rescale_with_baseline=True)

# set doc=True to evaluate at the document level
P, R, F1 = scorer.score(cands, refs, doc=True)
```
### Python usage (Function API):

In order to use Doc-BERTScore simple simply add `doc=True` when calling the `score` function:

```python
from bert_score import score

with open("bert_score/example/hyps.txt") as f:
    cands = [line.strip() for line in f]

with open("bert_score/example/refs.txt") as f:
    refs = [line.strip() for line in f]

with open("bert_score/example/docids.txt") as f:
    doc_ids = [line.strip() for line in f]
    
# add contexts to reference and hypothesis texts
cands = add_context(org_txt=cand, context=ref, docs=doc_ids)
refs = add_context(org_txt=ref, context=ref, docs=doc_ids)

# set doc=True to evaluate at the document level
P, R, F1 = score(cands, refs, lang="en", verbose=True, doc=True)
```

To use another model set the flag `model_type=MODEL_TYPE` when calling `score` function.


## Related Publications

- [BERTScore: Evaluating Text Generation with BERT](https://openreview.net/forum?id=SkeHuCVFDr)
