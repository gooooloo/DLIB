# feudal_networks

My daily used FuN(FeUdal Networks for Hierarchical Learning, https://arxiv.org/abs/1703.01161) implementaion. Based on Implementation from https://github.com/dmakian/feudal_networks.  I make some tiny changes to fit my working preferences.


## environment
To run this algo, I use tmux. I use python3.


## coding
- Typically I don't change `a3c.py` and `worker.py`. 
- If I want to change the model, I change `model.py`. 
- If I want to handle the env stuff, I change `envs.py`. 
- If I want to handle the hyper-parameters, I also change `envs.py`. 
- I hard-code almost every argument in `envs.py`, so that I can easily start the program in CLI without too much typing.


## running
To start training with default num-workers(=4), just run below in CLI:
```
./do.fun.sh
```

To start training with 32 workers:
```
./do.fun.sh -w 32
```

To stop training:
```
./no.fun.sh
```


## log archiving
```
python ./ziplog.py

```


## Versioning when tuning
I use `VSTR` in `envs.py` to identify my hyper-parameter and algo version info. I use `git` so I can easily check out
detail of every version. I use `ziplog.py` to archive tuning result.

