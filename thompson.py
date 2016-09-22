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
    parser.add_argument('--sample-size', type=int)
    parser.add_argument('--runs', required=True, type=int)
    args = parser.parse_args()
    algorithm = peeking.algorithm.ThompsonSampling((args.A, args.B), (1, 1))
    with peeking.concurrent.run(algorithm.proportions, args.runs, ((args.sample_size,),)) as results:
        proportions = list(results)

    plt.title('A = {:.2f}, B = {:.2f}'.format(args.A, args.B))
    x = range(args.sample_size)
    y_mean = numpy.mean(proportions, axis=0)
    #y_25 = numpy.percentile(proportions, 25, axis=0)
    #y_75 = numpy.percentile(proportions, 75, axis=0)
    #plt.plot(x, y_25, color='green', label='25th percentile')
    plt.plot(x, y_mean, color='blue', label='mean')
    #plt.plot(x, y_75, color='green', label='75th percentile')
    plt.xlabel('samples')
    plt.xlim((0, args.sample_size))
    plt.ylabel('B sampling proportion')
    plt.ylim((0, 1))
    #plt.legend(loc='upper left')
    plt.tight_layout()
    if args.output:
        plt.savefig(args.output)
    else:
        plt.show()
