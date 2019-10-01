# DeepArchitect: Automatically Designing and Training Deep Architectures

**IMPORTANT:** This repo is not under active development.
It contains a prototype for the ideas described in this [paper](https://arxiv.org/abs/1704.08792).
See our [NeurIPS 2019 paper](https://arxiv.org/abs/1909.13404) for the latest developments.
The code and documentation for the latest framework can be found [here](https://github.com/negrinho/deep_architect).

This repository contains a Python implementation of the DeepArchitect framework described in
[our paper](https://arxiv.org/abs/1704.08792).
To get familiar with the framework, we recommend starting with
[this tutorial](https://github.com/negrinho/deep_architect_legacy/blob/master/tutorial.ipynb).

A tar file with the logs of the experiments in the paper is available [here](http://www.cs.cmu.edu/~negrinho/assets/papers/deep_architect/logs.tar.gz). You can download it, unzip it in the top folder of the repo, and generate the plots of the paper using `plots.py`. The logs are composed of text and pickle files. It may be informative to inspect them. The experiments reported in the paper can be reproduced using `experiments.py`.

Contributors:
[Renato Negrinho](http://www.cs.cmu.edu/~negrinho/),
[Geoff Gordon](http://www.cs.cmu.edu/~ggordon/),
[Matt Gormley](http://www.cs.cmu.edu/~mgormley/),
[Christoph Dann](http://cdann.net/),
[Matt Barnes](http://www.cs.cmu.edu/~mbarnes1/).


## References

```
@article{negrinho2017deeparchitect,
  title={Deeparchitect: Automatically designing and training deep architectures},
  author={Negrinho, Renato and Gordon, Geoff},
  journal={arXiv preprint arXiv:1704.08792},
  year={2017}
}

@article{negrinho2019towards,
  title={Towards modular and programmable architecture search},
  author={Negrinho, Renato and Patil, Darshan and Le, Nghia and Ferreira, Daniel and Gormley, Matthew and Gordon, Geoffrey},
  journal={Neural Information Processing Systems},
  year={2019}
}
```