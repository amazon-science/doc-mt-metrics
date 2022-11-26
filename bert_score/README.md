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
bert-score -r example/refs.txt -c example/hyps.txt --lang en
```

To score using the document-level BERTScore simply add the `--doc` flag:
```bash
bert-score -r example/refs.txt -c example/hyps.txt --lang en --doc
```

In the paper we use `wmt21-comet-mqm` and `wmt21-comet-qe-mqm` models but you can select another model/metric with the `--model` flag.


### Python usage (Object-oriented API):

The [BERTScore](https://github.com/Tiiiger/bert_score) framework provides two APIs in order to use the BERTScore metric with python: an object-oriented one that caches the model and is recommended for multiple evaluations and a functional one that can be used for single evaluation. For more details see the [demo](https://github.com/Tiiiger/bert_score/blob/master/example/Demo.ipynb) provided by the authors.

In order to use Doc-BERTScore simple simply add `doc=True` when calling the `score` function:

```python
from bert_score import BERTScorer

with open("bert_score/example/hyps.txt") as f:
    cands = [line.strip() for line in f]

with open("bert_score/example/refs.txt") as f:
    refs = [line.strip() for line in f]

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

# set doc=True to evaluate at the document level
P, R, F1 = score(cands, refs, lang="en", verbose=True, doc=True)
```


## Related Publications

- [BERTScore: Evaluating Text Generation with BERT](https://openreview.net/forum?id=SkeHuCVFDr)
