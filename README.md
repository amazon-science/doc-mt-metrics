## Document Level MT Metrics

If you use the code in your work, please cite: 

[Embarrassingly Easy Document-Level MT Metrics: How to Convert Any Pretrained Metric Into a Document-Level Metric](https://statmt.org/wmt22/pdf/2022.wmt-1.6.pdf). Giorgos Vernikos, Brian Thompson, Prashant Mathur, and Marcello Federico. WMT 2022

## Overview

In this work we extend state-of-the-art Machine Translation metrics, namely [Prism](https://github.com/thompsonb/prism), [COMET](https://github.com/Unbabel/COMET),  [COMET-QE](https://github.com/Unbabel/COMET) and [BERTScore](https://github.com/Tiiiger/bert_score) to the document level. Our approach is _embarassingly simple_: instead of encoding only the hypothesis and reference we also encode the previous reference sentences as context. We still compute the metric score at the sentence level but also attend to previous context.

![image](media/bertscore.png)


The extended metrics outperform their sentence-level counterparts in about 85% of the tested conditions ([WMT 2021 Metrics Shared Task](https://wmt-metrics-task.github.io/) ) and dramatically improve the ability of the corresponding model to handle discourse phenomena.

## Usage

The current repository contains code that extends the original MT metrics to document level by providing the option to encode additional context. The code is presented as an extension of the corresponding original codebase. For information on how to use each metric see the corresponding README:
* [COMET/COMET-QE](COMET/README.md) 
* [BERTScore](bert_score/README.md)  
* [Prism](prism//README.md)

It is recommended to create an environment for this project 
```bash
conda create -n doc-metrics-env python=3.8 anaconda
conda activate doc-metrics-env
```

## Reproducibility

In order to reproduce the reuslts of the paper, regarding the correlation with human annotations of our (or any other) metrics on the standard test sets from the [WMT Metrics Shared Task](https://wmt-metrics-task.github.io/) we _strongly recommend_ using the [MT Metrics Eval](https://github.com/google-research/mt-metrics-eval) toolkit.

## Acknowledgments

We would like to thank the community for releasing their code! This repository contains code from [COMET](https://github.com/Unbabel/COMET), [BERTScore](https://github.com/Tiiiger/bert_score) and [Prism](https://github.com/thompsonb/prism) repositories.

Bibtex Entry

```
@misc{https://doi.org/10.48550/arxiv.2209.13654,
  doi = {10.48550/ARXIV.2209.13654},
  url = {https://arxiv.org/abs/2209.13654},
  author = {Vernikos, Giorgos and Thompson, Brian and Mathur, Prashant and Federico, Marcello},
  keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Embarrassingly Easy Document-Level MT Metrics: How to Convert Any Pretrained Metric Into a Document-Level Metric},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.

