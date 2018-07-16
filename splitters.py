"""
Contains an abstract base class that supports chemically aware data splits.
"""
from __future__ import division
from __future__ import unicode_literals


import tempfile
import numpy as np
import itertools
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina
from dataset import DiskDataset
from utils import ScaffoldGenerator


def log(string, verbose=True):
  """Print string if verbose."""
  if verbose:
    print(string)


def generate_scaffold(smiles, include_chirality=False):
  mol = Chem.MolFromSmiles(smiles)
  engine = ScaffoldGenerator(include_chirality=include_chirality)
  scaffold = engine.get_scaffold(mol)
  return scaffold

def randomize_arrays(array_list):
  # assumes that every array is of the same dimension
  num_rows = array_list[0].shape[0]
  perm = np.random.permutation(num_rows)
  permuted_arrays = []
  for array in array_list:
    permuted_arrays.append(array[perm])
  return permuted_arrays

class Splitter(object):
  """
    Abstract base class for chemically aware splits..
    """

  def __init__(self, verbose=False):
    """Creates splitter object."""
    self.verbose = verbose

  def k_fold_split(self, dataset, k, directories=None, **kwargs):
    """
    Parameters
    ----------
    dataset: Dataset
    Dataset to do a k-fold split

    k: int
    number of folds

    directories: list of str
    list of length 2*k filepaths to save the result disk-datasets

    kwargs

    Returns
    -------
    list of length k tuples of (train, cv)

    """
    """
    :param dataset:
    :param k:
    :param directories:
    :param kwargs:
    :return: list of length k tuples of (train, cv)
    """
    log("Computing K-fold split", self.verbose)
    if directories is None:
      directories = [tempfile.mkdtemp() for _ in range(2 * k)]
    else:
      assert len(directories) == 2 * k
    cv_datasets = []
    train_ds_base = None
    train_datasets = []
    # rem_dataset is remaining portion of dataset
    if isinstance(dataset, DiskDataset):
      rem_dataset = dataset
    else:
      rem_dataset = DiskDataset.from_numpy(dataset.X, dataset.y, dataset.w,
                                           dataset.ids)
    for fold in range(k):
      # Note starts as 1/k since fold starts at 0. Ends at 1 since fold goes up
      # to k-1.
      frac_fold = 1. / (k - fold)
      train_dir, cv_dir = directories[2 * fold], directories[2 * fold + 1]
      fold_inds, rem_inds, _ = self.split(
          rem_dataset,
          frac_train=frac_fold,
          frac_valid=1 - frac_fold,
          frac_test=0)
      cv_dataset = rem_dataset.select(fold_inds, select_dir=cv_dir)
      cv_datasets.append(cv_dataset)
      rem_dataset = rem_dataset.select(rem_inds)

      train_ds_to_merge = filter(lambda x: x is not None,
                                 [train_ds_base, rem_dataset])
      train_ds_to_merge = filter(lambda x: len(x) > 0, train_ds_to_merge)
      train_dataset = DiskDataset.merge(train_ds_to_merge, merge_dir=train_dir)
      train_datasets.append(train_dataset)

      update_train_base_merge = filter(lambda x: x is not None,
                                       [train_ds_base, cv_dataset])
      train_ds_base = DiskDataset.merge(update_train_base_merge)
    return list(zip(train_datasets, cv_datasets))

  def train_valid_test_split(self,
                             dataset,
                             train_dir=None,
                             valid_dir=None,
                             test_dir=None,
                             frac_train=.8,
                             frac_valid=.1,
                             frac_test=.1,
                             seed=None,
                             log_every_n=1000,
                             verbose=True):
    """
        Splits self into train/validation/test sets.

        Returns Dataset objects.
        """
    log("Computing train/valid/test indices", self.verbose)
    train_inds, valid_inds, test_inds = self.split(
        dataset,
        frac_train=frac_train,
        frac_test=frac_test,
        frac_valid=frac_valid,
        log_every_n=log_every_n)
    if train_dir is None:
      train_dir = tempfile.mkdtemp()
    if valid_dir is None:
      valid_dir = tempfile.mkdtemp()
    if test_dir is None:
      test_dir = tempfile.mkdtemp()
    train_dataset = dataset.select(train_inds, train_dir)
    if frac_valid != 0:
      valid_dataset = dataset.select(valid_inds, valid_dir)
    else:
      valid_dataset = None
    test_dataset = dataset.select(test_inds, test_dir)

    return train_dataset, valid_dataset, test_dataset

  def train_test_split(self,
                       dataset,
                       train_dir=None,
                       test_dir=None,
                       seed=None,
                       frac_train=.8,
                       verbose=True):
    """
        Splits self into train/test sets.
        Returns Dataset objects.
        """
    valid_dir = tempfile.mkdtemp()
    train_dataset, _, test_dataset = self.train_valid_test_split(
        dataset,
        train_dir,
        valid_dir,
        test_dir,
        frac_train=frac_train,
        frac_test=1 - frac_train,
        frac_valid=0.,
        verbose=verbose)
    return train_dataset, test_dataset

  def split(self,
            dataset,
            frac_train=None,
            frac_valid=None,
            frac_test=None,
            log_every_n=None,
            verbose=False):
    """
        Stub to be filled in by child classes.
        """
    raise NotImplementedError

class RandomSplitter(Splitter):
  """
    Class for doing random data splits.
    """

  def split(self,
            dataset,
            seed=None,
            frac_train=.8,
            frac_valid=.1,
            frac_test=.1,
            log_every_n=None):
    """
        Splits internal compounds randomly into train/validation/test.
        """
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.)
    if not seed is None:
      np.random.seed(seed)
    num_datapoints = len(dataset)
    train_cutoff = int(frac_train * num_datapoints)
    valid_cutoff = int((frac_train + frac_valid) * num_datapoints)
    shuffled = np.random.permutation(range(num_datapoints))
    return (shuffled[:train_cutoff], shuffled[train_cutoff:valid_cutoff],
            shuffled[valid_cutoff:])

class IndexSplitter(Splitter):
  """
    Class for simple order based splits.
    """

  def split(self,
            dataset,
            seed=None,
            frac_train=.8,
            frac_valid=.1,
            frac_test=.1,
            log_every_n=None):
    """
        Splits internal compounds into train/validation/test in provided order.
        """
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.)
    num_datapoints = len(dataset)
    train_cutoff = int(frac_train * num_datapoints)
    valid_cutoff = int((frac_train + frac_valid) * num_datapoints)
    indices = range(num_datapoints)
    return (indices[:train_cutoff], indices[train_cutoff:valid_cutoff],
            indices[valid_cutoff:])

def ClusterFps(fps, cutoff=0.2):
  # (ytz): this is directly copypasta'd from Greg Landrum's clustering example.
  dists = []
  nfps = len(fps)
  for i in range(1, nfps):
    sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
    dists.extend([1 - x for x in sims])
  cs = Butina.ClusterData(dists, nfps, cutoff, isDistData=True)
  return cs

class ButinaSplitter(Splitter):
  """
    Class for doing data splits based on the butina clustering of a bulk tanimoto
    fingerprint matrix.
    """

  def split(self,
            dataset,
            frac_train=None,
            frac_valid=None,
            frac_test=None,
            log_every_n=1000,
            cutoff=0.18):
    """
        Splits internal compounds into train and validation based on the butina
        clustering algorithm. This splitting algorithm has an O(N^2) run time, where N
        is the number of elements in the dataset. The dataset is expected to be a classification
        dataset.

        This algorithm is designed to generate validation data that are novel chemotypes.

        Note that this function entirely disregards the ratios for frac_train, frac_valid,
        and frac_test. Furthermore, it does not generate a test set, only a train and valid set.

        Setting a small cutoff value will generate smaller, finer clusters of high similarity,
        whereas setting a large cutoff value will generate larger, coarser clusters of low similarity.
        """
    print("Performing butina clustering with cutoff of", cutoff)
    mols = []
    for ind, smiles in enumerate(dataset.ids):
      mols.append(Chem.MolFromSmiles(smiles))
    n_mols = len(mols)
    fps = [AllChem.GetMorganFingerprintAsBitVect(x, 2, 1024) for x in mols]

    scaffold_sets = ClusterFps(fps, cutoff=cutoff)
    scaffold_sets = sorted(scaffold_sets, key=lambda x: -len(x))

    ys = dataset.y
    valid_inds = []
    for c_idx, cluster in enumerate(scaffold_sets):
      # for m_idx in cluster:
      valid_inds.extend(cluster)
      # continue until we find an active in all the tasks, otherwise we can't
      # compute a meaningful AUC
      # TODO (ytz): really, we want at least one active and inactive in both scenarios.
      # TODO (Ytz): for regression tasks we'd stop after only one cluster.
      active_populations = np.sum(ys[valid_inds], axis=0)
      if np.all(active_populations):
        print("# of actives per task in valid:", active_populations)
        print("Total # of validation points:", len(valid_inds))
        break

    train_inds = list(itertools.chain.from_iterable(scaffold_sets[c_idx + 1:]))
    test_inds = []

    return train_inds, valid_inds, []

class ScaffoldSplitter(Splitter):
  """
    Class for doing data splits based on the scaffold of small molecules.
    """

  def split(self,
            dataset,
            frac_train=.8,
            frac_valid=.1,
            frac_test=.1,
            log_every_n=1000):
    """
        Splits internal compounds into train/validation/test by scaffold.
        """
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.)
    scaffolds = {}
    log("About to generate scaffolds", self.verbose)
    data_len = len(dataset)
    for ind, smiles in enumerate(dataset.ids):
      if ind % log_every_n == 0:
        log("Generating scaffold %d/%d" % (ind, data_len), self.verbose)
      scaffold = generate_scaffold(smiles)
      if scaffold not in scaffolds:
        scaffolds[scaffold] = [ind]
      else:
        scaffolds[scaffold].append(ind)
    # Sort from largest to smallest scaffold sets
    scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
    scaffold_sets = [
        scaffold_set
        for (scaffold, scaffold_set) in sorted(
            scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    ]
    train_cutoff = frac_train * len(dataset)
    valid_cutoff = (frac_train + frac_valid) * len(dataset)
    train_inds, valid_inds, test_inds = [], [], []
    log("About to sort in scaffold sets", self.verbose)
    for scaffold_set in scaffold_sets:
      if len(train_inds) + len(scaffold_set) > train_cutoff:
        if len(train_inds) + len(valid_inds) + len(scaffold_set) > valid_cutoff:
          test_inds += scaffold_set
        else:
          valid_inds += scaffold_set
      else:
        train_inds += scaffold_set
    return train_inds, valid_inds, test_inds
