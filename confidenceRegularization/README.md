# Confidence regularization

Training of DistilBert with confidence regularization method

Code is based on https://github.com/UKPLab/acl2020-confidence-regularization

Install with:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Proceed with training of parent and distillation:
```
# traininng parent, with all routes set
python train_parent.py

# training distilled model, uses parent outputs
python train_distillbert.py
```
