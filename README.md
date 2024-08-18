# rsna2024
## Run in kaggle notebook
```python:something.ipynb
!python -m run.train --multirun directory=kaggle env=kaggle exp_name=run0818 split=fold_0,fold_1,fold_2,fold_3,fold_4 trainer.epochs=1 batch_size=8 optimizer.lr=0.00008 model.name='swin_large_patch4_window12_384.ms_in22k' dataset.image_size=384
```
```
!python -m run.inference directory=kaggle env=kaggle model.params.pretrained=false model.name='swin_large_patch4_window12_384.ms_in22k'
```

## Run in your environment
```python
$python -m run.train --multirun directory=local env=local exp_name=run0818 split=fold_0,fold_1,fold_2,fold_3,fold_4 trainer.epochs=1 batch_size=8 optimizer.lr=0.00008 model.name='swin_large_patch4_window12_384.ms_in22k' dataset.image_size=384
```
```python
$python -m run.inference directory=local env=local model.params.pretrained=false model.name='swin_large_patch4_window12_384.ms_in22k'
```
