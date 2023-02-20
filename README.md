Best to run in a clean conda env

```bash
# get analysis code and install as package
git clone git@github.com:andrzejnovak/boostedhiggs.git
cd boostedhiggs
pip install -e . 
cd ..

# get runner code
git clone git@github.com:andrzejnovak/nanocc.git
cd nanocc

# initiate proxy e.g
# voms-proxy-init --voms cms:/cms/dcms --valid 168:00  --vomses ~/.grid-security/vomses/
```


### Test run
Use `--executor iterative` for single process to debug, `--executor futures` for local multiprocessing.

```bash
python runner.py --id test17 --json metadata/v2x17.json --year 2017 --limit 1 --chunk 5000 --max 2 --executor futures -j 5 
```

### Test scale out
Scale-out will be dependent on the cluster setup. If the cluster is sufficiently permissive the below might run right away, otherwise some editing of `runner.py` and `HighThroughputExecutor` when using `--executor parsl` will be necessary. Analogously for `--executor dask`

```bash
python runner.py --id test17 --json metadata/v2x17.json --year 2017 --limit 1 --chunk 5000 --max 2 --executor parsl
```

### Full scale
Removing test limiters...
```bash
python runner.py --id test17 --json metadata/v2x17.json --year 2017 --executor parsl
```