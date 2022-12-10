To reproduce the results of the paper on the [WTM21 Metrics task](https://www.statmt.org/wmt21/metrics-task.html) data first install the [MT Metrics Eval](https://github.com/google-research/mt-metrics-eval) toolkit
and download the database.
```bash
git clone https://github.com/google-research/mt-metrics-eval.git
cd mt-metrics-eval
pip install .
alias mtme='python3 -m mt_metrics_eval.mtme'
mtme --download  # Puts ~1G of data into $HOME/.mt-metrics-eval.
```
To obtain system-level scores of Doc-COMET(-QE) for the WMT21 testet run:
```bash
python score_bertscore.py --campaign wmt21.news --lp en-de --doc --level sys
````
