#!/usr/bin/env python2
import argparse
import matplotlib.pyplot as plt
import numpy
import peeking.algorithm
import peeking.concurrent

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('A', type=float)
    parser.add_argument('B', type=float)
    parser.add_argument('--output')
    parser.add_argument('--peeking-frequency', type=int, required=True)
    parser.add_argument('--p-value', required=True, type=float)
    parser.add_argument('--min-sample-size', type=int)
    parser.add_argument('--sample-size', type=int)
    parser.add_argument('--runs', required=True, type=int)
    args = parser.parse_args()

    distributions = (args.A, args.B)

    algorithms = [
        peeking.algorithm.FixedFrequencyTest(distributions, args.p_value, 500),
        peeking.algorithm.FixedFrequencyTest(distributions, args.p_value, 2000),
        peeking.algorithm.FixedFrequencyTest(distributions, args.p_value, 4000),
        peeking.algorithm.FrequencyTest(distributions, args.p_value, args.peeking_frequency, args.min_sample_size),
        peeking.algorithm.ThompsonSampling(distributions, (1, 1))
    ]
    results = []
    for algorithm in algorithms:
        # Thompson sampling algorithm results relatively stable
        runs = args.runs / 4 if isinstance(algorithm, peeking.algorithm.ThompsonSampling) else args.runs
        with peeking.concurrent.run(algorithm.success, runs, ((args.sample_size,),)) as successes:
            results.append(numpy.mean(list(successes)))

    plt.title('A = {:.2f}, B = {:.2f}, {} samples'.format(args.A, args.B, args.sample_size))
    plt.barh(range(len(results)), results, align='center')
    plt.xlim(args.sample_size * min(distributions), args.sample_size * max(distributions))
    plt.xlabel('Successes')
    plt.yticks(range(len(results)), (
        '500 samples',
        '2000 samples',
        '4000 samples',
        'peeking',
        'thompson',
    ))
    plt.ylim(len(results) - 0.3, -0.7)
    plt.tight_layout()
    if args.output:
        plt.savefig(args.output)
    else:
        plt.show()
