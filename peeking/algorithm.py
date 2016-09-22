import numpy
import random
import scipy.stats

class FixedFrequencyTest:
    def __init__(self, distributions, p, sample_size):
        self.distributions = distributions
        self.p = p
        self.sample_size = sample_size

    def decision(self, sample_size):
        table = [[] for _ in self.distributions]
        for distribution, row in zip(self.distributions, table):
            success = numpy.random.binomial(sample_size / 2, distribution)
            row.append(success)
            row.append(sample_size / 2 - success)
        p = scipy.stats.chi2_contingency(table, correction=False)[1]
        if p < self.p:
            winner = numpy.argmax([success for success, _ in table])
            return winner, sample_size

    def success(self, sample_size):
        successes = [numpy.random.binomial(self.sample_size / 2, d) for d in self.distributions]
        success = sum(successes)
        if self.sample_size < sample_size:
            winner = numpy.argmax(successes)
            success += numpy.random.binomial(sample_size - self.sample_size, self.distributions[winner])
        return success

class FrequencyTest:
    def __init__(self, distributions, p, peeking_frequency, min_sample_size):
        self.distributions = distributions
        self.p = p
        self.peeking_frequency = peeking_frequency
        self.min_sample_size = min_sample_size

    def _results(self, sample_size):
        table = [[0, 0] for _ in self.distributions]
        for i in xrange(0, sample_size, 2):
            for distribution, row in zip(self.distributions, table):
                row[random.random() >= distribution] += 1
            if i >= self.min_sample_size and not (i + 2) % ((self.peeking_frequency + 1) / 2 * 2):
                if scipy.stats.chi2_contingency(table, correction=False)[1] < self.p:
                    return table
        return table

    def decision(self, sample_size):
        table = self._results(sample_size)
        i = sum(map(sum, table))
        if i < sample_size - 1:
            winner = numpy.argmax([success for success, _ in table])
            return winner, i

    def success(self, sample_size):
        table = self._results(sample_size)
        i = sum(map(sum, table))
        winner = numpy.argmax([success for success, _ in table])
        return sum(success for success, _ in table) + \
            numpy.random.binomial(sample_size - i, self.distributions[winner])

class ThompsonSampling:
    def __init__(self, distributions, prior):
        self.distributions = distributions
        self.prior = prior

    def success(self, sample_size):
        table = [[0, 0] for _ in self.distributions]
        for i in xrange(sample_size):
            if not i % 20:
                dists = [scipy.stats.beta(table[d][0] + self.prior[0], table[d][1] + self.prior[1]) for d in xrange(len(self.distributions))]
            choice = numpy.argmax([d.rvs() for d in dists])
            table[choice][random.random() >= self.distributions[choice]] += 1
        return sum(success for success, _ in table)

    def proportions(self, sample_size):
        proportions = []
        table = [list(self.prior) for _ in self.distributions]
        for i in xrange(sample_size):
            if not i % 20:
                dists = [scipy.stats.beta(*table[d]) for d in xrange(len(self.distributions))]
                proportion = sum(numpy.argmax(r) for r in zip(*(d.rvs(size=500) for d in dists))) / 500.
            choice = numpy.argmax([d.rvs() for d in dists])
            table[choice][random.random() >= self.distributions[choice]] += 1
            proportions.append(proportion)
        return proportions

class PeekingThompson:
    def __init__(self, distributions, p, prior, min_sample_size):
        self.distributions = distributions
        self.p = p
        self.prior = prior
        self.min_sample_size = min_sample_size

    def decision(self, sample_size):
        table = [list(self.prior) for _ in self.distributions]
        for i in xrange(sample_size):
            if not i % 20:
                dists = [scipy.stats.beta(*table[d]) for d in xrange(len(self.distributions))]
                proportion = sum(numpy.argmax(r) for r in zip(*(d.rvs(size=500) for d in dists))) / 500.
            choice = numpy.argmax([d.rvs() for d in dists])
            table[choice][random.random() >= self.distributions[choice]] += 1
            if self.min_sample_size <= i:
                if proportion < self.p / 2:
                    return 0, i
                if 1 - self.p / 2 < proportion:
                    return 1, i
