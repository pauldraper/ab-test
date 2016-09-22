#!/usr/bin/env python2
import argparse
import collections
import functools
import peeking.algorithm
import peeking.concurrent
import matplotlib.pyplot as plt
import numpy

def run(distributions, p, sample_size):
    return sample_size, peeking.algorithm.FixedFrequencyTest(distributions, p, sample_size).decision(sample_size)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('A', type=float)
    parser.add_argument('B', type=float)
    parser.add_argument('--interval-size', required=True, type=int)
    parser.add_argument('--output')
    parser.add_argument('--p-value', required=True, type=float)
    parser.add_argument('--min-sample-size', type=int)
    parser.add_argument('--max-sample-size', type=int)
    parser.add_argument('--runs', required=True, type=int)
    args = parser.parse_args()

    total = collections.Counter()
    accept = [collections.Counter() for _ in xrange(2)]
    f = functools.partial(run, (args.A, args.B), (args.p_value))
    with peeking.concurrent.run(f, args.runs, [[x] for x in xrange(args.min_sample_size, args.max_sample_size)]) as results:
        for r in results:
            sample_size, result = r
            rounded = sample_size / args.interval_size * args.interval_size + args.interval_size / 2
            if result:
                accept[result[0]][rounded] += 1
            total[rounded] += 1

    plt.title('A = {:.2f}, B = {:.2f}'.format(args.A, args.B))
    max_y = 0
    for label, a in zip(('A', 'B'), accept):
        x, y = zip(*sorted((x, a[x] / float(y)) for x, y in total.items()))
        max_y = max(max_y, max(y))
        plt.plot(x, y, label=label)
    plt.xlabel('samples')
    plt.xlim((0, args.max_sample_size))
    plt.ylabel('probability')
    plt.ylim((0, min(1, max_y * 2)))
    plt.legend(loc='upper left')
    plt.tight_layout()
    if args.output:
        plt.savefig(args.output)
    else:
        plt.show()
