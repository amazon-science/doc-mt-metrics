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

To score using the document-level COMET/COMET-QE simply add the `--doc` flag:
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

model_path = download_model("wmt21-comet-mqm")
model = load_from_checkpoint(model_path)

# this command allows us to encode context for document-level evaluation
model.set_document_level()

data = [
    {
        "src": "Dem Feuer konnte Einhalt geboten werden",
        "mt": "The fire could be stopped",
        "ref": "They were able to control the fire."
    },
    {
        "src": "Schulen und Kindergärten wurden eröffnet.",
        "mt": "Schools and kindergartens were open",
        "ref": "Schools and kindergartens opened"
    }
]
seg_scores, sys_score = model.predict(data, batch_size=8, gpus=1)
```

## Related Publications

- [Searching for Cometinho: The Little Metric That Could -- EAMT22 Best paper award](https://aclanthology.org/2022.eamt-1.9/)

- [Are References Really Needed? Unbabel-IST 2021 Submission for the Metrics Shared Task](http://statmt.org/wmt21/pdf/2021.wmt-1.111.pdf)

- [Uncertainty-Aware Machine Translation Evaluation](https://aclanthology.org/2021.findings-emnlp.330/) 

- [COMET - Deploying a New State-of-the-art MT Evaluation Metric in Production](https://www.aclweb.org/anthology/2020.amta-user.4)

- [Unbabel's Participation in the WMT20 Metrics Shared Task](https://aclanthology.org/2020.wmt-1.101/)

- [COMET: A Neural Framework for MT Evaluation](https://www.aclweb.org/anthology/2020.emnlp-main.213)
