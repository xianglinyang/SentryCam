# dataset prepare
```console
git clone timm
bash run.sh
```

# tdvi
```console
git clone Genvis

mkdir /data/path/config
python config.py --path /data/path/config/xxx.yaml
# modify config parameters
# modify model parameters with model parameters (vit required)

# cold start in sequence
bash run.sh
# skip for the rest
bash run.sh
```
