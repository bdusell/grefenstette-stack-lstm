import argparse

import dynet

from stack_lstm import StackLSTMBuilder
from train import test, parse_range

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('input',
        help='Input file containing saved parameters.')
    parser.add_argument('--hidden-units', type=int, default=256,
        help='Number of hidden units used in the LSTM controller.')
    parser.add_argument('--source-alphabet-size', type=int, default=128,
        help='Number of symbols to use in the source sequence.')
    parser.add_argument('--embedding-size', type=int, default=64,
        help='Input embedding size.')
    parser.add_argument('--stack-embedding-size', type=int, default=256,
        help='Size of vector values stored on the neural stack.')
    parser.add_argument('--test-length-range', type=parse_range, default=(65, 128),
        help='Range of lengths of source sequences during testing.',
        metavar='MIN,MAX')
    parser.add_argument('--test-data-size', type=int, default=1000,
        help='Number of samples used in the test data.')
    args = parser.parse_args()

    params = dynet.ParameterCollection()
    builder = StackLSTMBuilder(
        params,
        source_alphabet_size=args.source_alphabet_size,
        embedding_size=args.embedding_size,
        stack_embedding_size=args.stack_embedding_size,
        hidden_units=args.hidden_units)
    params.populate(args.input)
    test(args, builder)

if __name__ == '__main__':
    main()
