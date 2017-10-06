Stack LSTM
==========

This is a Python/[DyNet](https://github.com/clab/dynet) implementation of the
neural stack LSTM model presented in the following paper:

Grefenstette, Edward, Karl Moritz Hermann, Mustafa Suleyman, and Phil Blunsom.
"Learning to transduce with unbounded memory." In _Advances in Neural
Information Processing Systems_, pp. 1828-1836. 2015.

The purpose of this repository is to serve as a reference implementation for
those who might be interested in reproducing the results presented in this
paper, as fine-tuning the model to converge on algorithmic behavior can prove
to be quite challenging. This repository includes a script for training a stack
LSTM on the sequence reversal transduction task.

Setup
-----

This code runs in Python 3 and requires a relatively recent (as of October 2017)
version of DyNet. As DyNet is under active development, it is recommended to
use the script `./setup` to install the exact version of DyNet tested with this
code into an isolated [virtualenv](https://virtualenv.pypa.io/en/stable/).

```sh
./setup
```

Instructions for installing the `virtualenv` command can be found
[here](https://virtualenv.pypa.io/en/stable/installation/).

In order to run a script inside the virtualenv, first source its `activate`
script, then run the python script as usual.

```sh
. virtualenv/bin/activate
python train.py
```

Running
-------

The script `train.py` can be used to train and test a stack LSTM on the
sequence reversal transduction task as described in the paper. Note that it is
rather typical for this architecture to get permanently stuck in a local
minimum during training, so you may need to try more than once to get a
successful run. Using the default values for `train.py`, you should be able to
successfully train a model about 80% of the time.

Note that the default size of the model is quite large, and the script is best
run on GPU in this case. If you want to test out a smaller version of the
model first, try running `train.py` with smaller values for number of hidden
units, embedding size, etc. You can control the dimensions of the model trained,
number of training samples, learning rate, and more by passing command line
arguments to `train.py`. You can also save the parameters of the trained model
to an output file. Run `python train.py --help` for more details.

For example:

```sh
python train.py \
    --hidden-units 20 \
    --source-alphabet-size 2 \
    --embedding-size 5 \
    --stack-embedding-size 3 \
    --training-length-range 10,10 \
    --test-length-range 10,10 \
    --test-data-size 10 \
    --output parameters.txt
```

The script `test.py` can be used to run just the test phase on a pre-trained
model. Note that the model dimensions passed to `test.py` need to match those
used with `train.py`.

As noted by the authors, success during training is extremely sensitive to
parameter initialization. In my experiments, I have found that initializing the
parameters for the push and pop signal layers close to 0 has a marked
improvement on success rate, so they are initialized with an Xavier
initialization gain of 0.5 in this implementation, as opposed to the
recommended value of 4. Success is also very sensitive to the choice of
optimizer (RMSProp in this case) and initial learning rate. The set of default
values for `train.py` is one that happens to work well. If you deviate from
these default values, you will need to fiddle a lot with the various arguments
to `train.py` to get an experimental setup that succeeds.

The source code for the model is found in `stack_lstm.py`.

Author
------

[Brian DuSell](http://bdusell.com/)
