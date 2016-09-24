#!/usr/bin/env python2
import argparse
import matplotlib.pyplot as plt
import peeking.algorithm
import peeking.concurrent

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('A', type=float)
    parser.add_argument('B', type=float)
    parser.add_argument('--min-sample-size', type=int)
    parser.add_argument('--output')
    parser.add_argument('--p-value', required=True, type=float)
    parser.add_argument('--peeking-frequency', required=True, type=float)
    parser.add_argument('--runs', required=True, type=int)
    parser.add_argument('--sample-size', type=int)
    args = parser.parse_args()

    algorithm = peeking.algorithm.FrequencyTest((args.A, args.B), args.p_value, args.peeking_frequency, args.min_sample_size)

    accept = [[], []]
    with peeking.concurrent.run(algorithm.decision, args.runs, ((args.sample_size,),)) as results:
        for r in results:
            if r:
                winner, i = r
                accept[winner].append(i)

    plt.title('A = {}, B = {}'.format(args.A, args.B))
    for i, a in enumerate(accept):
        a.sort()
        x = a + [args.sample_size]
        y = [float(y + 1) / args.runs for y in xrange(len(a))] + [float(len(a)) / args.runs]
        plt.xlabel('samples')
        plt.xlim((0, args.sample_size))
        plt.ylabel('cummulative probability')
        plt.ylim((0, 1))
        plt.plot(x, y, label=chr(ord('A') + i))
    plt.legend(loc='upper left')
    plt.tight_layout()
    if args.output:
        plt.savefig(args.output)
    else:
        plt.show()
