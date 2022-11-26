# Doc-COMET

This README describes how to use **Doc-COMET** an extension of the original COMET metric that can be used for document-level evaluation.

## Installation

This codebase is built upon the original [COMET code](https://github.com/Unbabel/COMET). For a detailed documentation of the COMET metric, including usage examples and instructions see the [Full Documentation](https://unbabel.github.io/COMET/html/index.html).

To run Doc-COMET you will need to develop locally:
```bash
git clone https://github.com/GeorgeVern/doc-mt-metrics.git
cd COMET
pip install -r requirements.txt
pip install -e .
```

## Scoring MT outputs:

### Command Line usage:

To score using the original, sentence-level COMET:
```bash
comet-score -s src.de -t hyp1.en -r ref.en
```

To score using the document-level COMET:
```bash
comet-score -s src.de -t hyp1.en -r ref.en -doc
```
> you can set `--gpus 0` to test on CPU.

### Python usage:

```python
from comet import download_model, load_from_checkpoint

model_path = download_model("wmt21-comet-mqm")
model = load_from_checkpoint(model_path)

# this command allows us to encode context while scoring the desired hypothesis
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
If you use COMET please cite our work! Also, don't forget to say which model you used to evaluate your systems.

- [Searching for Cometinho: The Little Metric That Could -- EAMT22 Best paper award](https://aclanthology.org/2022.eamt-1.9/)

- [Are References Really Needed? Unbabel-IST 2021 Submission for the Metrics Shared Task](http://statmt.org/wmt21/pdf/2021.wmt-1.111.pdf)

- [Uncertainty-Aware Machine Translation Evaluation](https://aclanthology.org/2021.findings-emnlp.330/) 

- [COMET - Deploying a New State-of-the-art MT Evaluation Metric in Production](https://www.aclweb.org/anthology/2020.amta-user.4)

- [Unbabel's Participation in the WMT20 Metrics Shared Task](https://aclanthology.org/2020.wmt-1.101/)

- [COMET: A Neural Framework for MT Evaluation](https://www.aclweb.org/anthology/2020.emnlp-main.213)



