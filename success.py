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
        ('500 samples', peeking.algorithm.FixedFrequencyTest(distributions, args.p_value, 500)),
        ('2000 samples', peeking.algorithm.FixedFrequencyTest(distributions, args.p_value, 2000)),
        ('4000 samples', peeking.algorithm.FixedFrequencyTest(distributions, args.p_value, 4000)),
        ('peeking', peeking.algorithm.FrequencyTest(distributions, args.p_value, args.peeking_frequency, args.min_sample_size)),
        ('thompson', peeking.algorithm.ThompsonSampling(distributions, (1, 1))),
    ]
    results = []
    for name, algorithm in algorithms:
        # Thompson sampling algorithm results relatively stable
        runs = args.runs / 4 if isinstance(algorithm, peeking.algorithm.ThompsonSampling) else args.runs
        with peeking.concurrent.run(algorithm.success, runs, ((args.sample_size,),)) as successes:
            success_rate = numpy.mean(list(successes)) / float(args.sample_size)
            results.append((name, success_rate))

    plt.title('A = {:.2f}, B = {:.2f}, {} samples'.format(args.A, args.B, args.sample_size))
    plt.barh(range(len(results)), [rate for _, rate in results], align='center')
    plt.xlim(min(distributions), max(distributions))
    plt.xlabel('Cummulative success rate')
    plt.yticks(range(len(results)), [name for name, _ in results])
    plt.ylim(len(results) - 0.3, -0.7)
    plt.tight_layout()
    if args.output:
        plt.savefig(args.output)
    else:
        plt.show()
