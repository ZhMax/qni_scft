This directory contains code for computing sensitivity metrics for the columns of weight matrices in a model.


To compute sensitivity metrics which are proposed in our work:
```bash
bash sensitivity_metrics/salientcolumns/scripts/sensitive_metric.sh
```

To compute sensitivity metrics which are used in [SmoothQuant](https://github.com/mit-han-lab/smoothquant):
```bash
bash sensitivity_metrics/smoothquant/scripts/generate_act_scales.sh
```

Before running the scripts, [download](https://huggingface.co/datasets/mit-han-lab/pile-val-backup/resolve/main/val.jsonl.zst) the validation dataset of the Pile.