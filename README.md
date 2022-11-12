# Finding Private Bugs: Debugging Implementations of Differentially Private Stochastic Gradient Descent

This repository is an implementation of the paper [Finding Private Bugs: Debugging Implementations of Differentially Private Stochastic Gradient Descent](https://openreview.net/forum?id=gKKUZ4fTEqh). 
In this paper, we proposed an easy method to detect common implementation errors in DP-SGD for practitioners.

### Dependency
Our code is implemented and tested on PyTorch. Following packages are used:
```
hypothesis
numpy
opacus==0.15.0
pandas
pynvml
pytest
scikit_image
scipy
torch==1.10.0
torchvision==0.11.1
```

###  Check if there is clipping
```
# for no clipping 
python exp_no_clipping.py --optim sgd --no-clipping 1 --batch-size 1

# for per-example clipping or mini-batch clipping (with batch-size = 1, they are the same)
python exp_no_clipping.py --optim sgd --no-clipping 0 --batch-size 1

```

### Check if clipping is correct 
``` 
# for mini-batch clipping 
python exp_batch_clip.py --correct-clipping 0 --batch-clipping 1

# for per-example clipping 
python exp_batch_clip.py --correct-clipping 1 --batch-clipping 0
```

### Check if the noise is calibrated 
```
# for uncalibrated noise 
python exp_wrong_noise.py --correct-clipping 1 --wrong-noise-calibration 1 --noise-multiplier 0.001 --optim sgd

# for calibrated noise
python exp_wrong_noise.py --correct-clipping 1 --wrong-noise-calibration 0 --noise-multiplier 0.001 --optim sgd

```

### Questions or suggestions
If you have any questions or suggestions, feel free to raise an issue or send me an email at emmy.fang@mail.utoronto.ca


### Citing this work
If you use this repository for academic research, you are highly encouraged (though not required) to cite our paper:
```
@inproceedings{
anonymous2023finding,
title={Finding Private Bugs: Debugging Implementations of Differentially Private Stochastic Gradient Descent },
author={Anonymous},
booktitle={Submitted to The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=gKKUZ4fTEqh},
note={under review}
}
```


