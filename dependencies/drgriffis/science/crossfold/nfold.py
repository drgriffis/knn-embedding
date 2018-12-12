'''
N-fold evaluation for intrinsic tasks
'''

import os
import numpy as np
import random
from evals.prm import PRM

class NFoldManager:
    
    def __init__(self, n):
        self.n = n

    def getSplits(self, *datasets):
        chunked_splits = [self._splitSingleDataset(ds) for ds in datasets]
        if len(datasets) == 1:
            return chunked_splits[0]
        else:
            # interpolate (e.g. if 2 datasets with 2 chunks, returns
            # [ds1[0], ds2[0]], [ds1[1], ds2[1]])
            return [
                [chunked_splits[j][i] for j in range(len(datasets))]
                    for i in range(self.n)
            ]

    def _splitSingleDataset(self, dataset):
        split_size = int(len(dataset) / self.n)
        splits = []
        for i in range(self.n):
            upper_limit = ((i+1)*split_size) if i < self.n-1 else len(dataset)
            splits.append(dataset[i*split_size:upper_limit])
        return splits

    def runSplits(self, ev_method, splits, *args, **kwargs):
        return [
            ev_method(split, *args, **kwargs)
                for split in splits
        ]

    def saveSplits(self, save_method, results, split_prms, *args, **kwargs):
        for split in range(len(results)):
            result, this_split_prms = results[split], split_prms.getSplit(split)
            save_method(result, this_split_prms, *args, **kwargs)

    def loadSplitPRMs(self, *basenames, default_shape=(1,)):
        prms = []
        for i in range(self.n):
            full_names = ['%s.split%d.npy' % (os.path.splitext(basename)[0], i) for basename in basenames]

            split_prms = []
            for full_name in full_names:
                if os.path.isfile(full_name): split_prms.append(PRM.load(full_name))
                else:
                    prm = PRM(*default_shape, path=full_name)
                    prm.save()
                    split_prms.append(prm)

            if len(basenames) == 1:
                prms.append(split_prms[0])
            else:
                prms.append(split_prms)

        return SplitPRMs(self.n, prms, prms_per_split=len(basenames))

    def getSplitPRMValue(self, split_prms, *ixes, prm_ixes=None):
        if prm_ixes is None:
            prm_ixes = range(split_prms._prms_per_split)
        elif type(prm_ixes) is int:
            prm_ixes = [prm_ixes]

        vals = np.array([
            [
                split_prms[prm_ix][split][ixes]
                    for split in range(self.n)
            ]
                for prm_ix in prm_ixes
        ])

        if len(prm_ixes) == 1: return vals[0]
        else: return vals

class SplitPRMs:
    
    _prm_splits = None
    _prms_per_split = 0
    nsplit = -1
    
    def __init__(self, nsplit, prm_splits, prms_per_split=1):
        self._prm_splits = prm_splits
        self._prms_per_split = prms_per_split
        self.nsplit = nsplit

    def __getitem__(self, key):
        if (not type(key) is int) or \
                key > self._prms_per_split:
            raise KeyError("Key %s is invalid (key must be in range 0-%d)" % (str(key), self._prms_per_split-1))
        if self._prms_per_split == 1:
            return self._prm_splits
        else:
            return [split_prms[key] for split_prms in self._prm_splits]

    def getSplit(self, split):
        return self._prm_splits[split]

    def apply(self, method):
        return [
            method(split_prms) for split_prms in self._prm_splits
        ]

    def sumSplits(self):
        first_split = self.getSplit(0)
        sum_prms = [np.zeros(first_split[i].shape) for i in range(self._prms_per_split)]

        for i in range(self.nsplit):
            split = self.getSplit(i)
            for j in range(self._prms_per_split):
                this_prm_split = split[j].matr() if type(split[j]) is PRM else split[j]
                sum_prms[j] = sum_prms[j] + this_prm_split
        return sum_prms

    def __add__(self, other):
        if not type(other) is SplitPRMs:
            return NotImplemented

        if self.nsplit != other.nsplit:
            raise ValueError("SplitPRM instances must have same number of splits")
        elif self._prms_per_split != other._prms_per_split:
            raise ValueError("SplitPRM instances must contain same number of PRMs")
        else:
            new_prm_splits = []
            for i in range(self.nsplit):
                if self._prms_per_split > 1:
                    split_prms = []
                    for j in range(self._prms_per_split):
                        split_prms.append(self._prm_splits[i][j] + other._prm_splits[i][j])
                    new_prm_splits.append(split_prms)
                else:
                    new_prm_splits.append(self._prm_splits[i] + other._prm_splits[i])
            return SplitPRMs(self.nsplit, new_prm_splits, self._prms_per_split)

    def __truediv__(self, x):
        if not type(x) in [float, int]:
            return NotImplemented

        new_prm_splits = []
        for i in range(self.nsplit):
            if self._prms_per_split > 1:
                split_prms = []
                for j in range(self._prms_per_split):
                    split_prms.append(self._prm_splits[i][j] / x)
                new_prms_splits.append(split_prms)
            else:
                new_prm_splits.append(self._prm_splits[i] / x)
        return SplitPRMs(self.nsplit, new_prm_splits, self._prms_per_split)
