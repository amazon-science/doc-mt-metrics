# Embarrassingly Easy Document-Level MT Metrics

## Overview

In this work we extend state-of-the-art Machine Translation metrics, namely [Prism](https://github.com/thompsonb/prism), [COMET](https://github.com/Unbabel/COMET),  [COMET-QE](https://github.com/Unbabel/COMET) and [BERTScore](https://github.com/Tiiiger/bert_score) to the document level. Our approach is _embarassingly simple_: instead of encoding only the hypothesis and reference we also encode the previous reference sentences as context. We still compute the metric score at the sentence level but also attend to previous context.

![image](media/bertscore.png)


The extended metrics outperform their sentence-level counterparts in about 85% of the tested conditions ([WMT 2021 Metrics Shared Task](https://wmt-metrics-task.github.io/) ) and dramatically improve the ability of the corresponding model to handle discourse phenomena.

## Usage

The current repository contains code that extends the original MT metrics to document level by providing the option to encode additional context. The code is presented as an extension of the corresponding original codebase. For information on how to use each metric see the corresponding README:
* [COMET/COMET-QE](COMET/README.md) 
* [BERTScore](bert_score/README.md)  
* [Prism](Prism//README.md)

It is recommended to create an environment for this project 
```bash
conda create -n doc-metrics-env python=3.9
conda activate doc-metrics-env
```

## Reproducibility

In order to reproduce the results of the paper, regarding the correlation with human annotations of document or sentence-level metrics on the test sets from the [WMT Metrics Shared Task](https://wmt-metrics-task.github.io/) first install the required packages for [BERTScore](/bert_score) and [COMET](/COMET) models. Next, install the [MT Metrics Eval](https://github.com/google-research/mt-metrics-eval) toolkit
and download the database.
```bash
git clone https://github.com/google-research/mt-metrics-eval.git
cd mt-metrics-eval
pip install .
alias mtme='python3 -m mt_metrics_eval.mtme'
mtme --download  # Puts ~1G of data into $HOME/.mt-metrics-eval.
```
Then use the `score_doc-metrics.py` script to obtain the scores for the model, domain and language pair of your choice from the WMT21 test sets. 
For example, to obtain system-level scores of Doc-COMET for the en-de language pair in the news domain, run:
```bash
python score_doc-metrics.py --campaign wmt21.news --model comet --lp en-de --level sys --doc
````
## Acknowledgments

We would like to thank the community for releasing their code! This repository contains code from [COMET](https://github.com/Unbabel/COMET), [BERTScore](https://github.com/Tiiiger/bert_score) and [Prism](https://github.com/thompsonb/prism) repositories.


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

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.

