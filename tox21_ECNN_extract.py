"""
Script that trains graph-conv models on Tox21 dataset.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from dataset import DiskDataset
from adj_finger import *
from graph_features import *
from transformers import *
from splitters import *


import numpy as np
import time
import math
import os
import tensorflow as tf
import pandas as pd
import tempfile
import pickle
np.random.seed(123)
tf.set_random_seed(123)


def to_one_hot(y, n_classes=2):
  n_samples = np.shape(y)[0]
  y_hot = np.zeros((n_samples, n_classes))
  y_hot[np.arange(n_samples), y.astype(np.int64)] = 1
  return y_hot


def log(string, verbose=True):
  if verbose:
    print(string)


def load_csv_files(filenames, shard_size=None, verbose=True):
  """Load data as pandas dataframe."""
  # First line of user-specified CSV *must* be header.
  shard_num = 1
  for filename in filenames:
    if shard_size is None:
      yield pd.read_csv(filename)
    else:
      log("About to start loading CSV from %s" % filename, verbose)
      for df in pd.read_csv(filename, chunksize=shard_size):
        log("Loading shard %d of size %s." % (shard_num, str(shard_size)),
            verbose)
        df = df.replace(np.nan, str(""), regex=True)
        shard_num += 1
        yield df


def ecnn_featurize_mol(mol, L, K, max_n_atoms):
  A, V, H, R = get_atom_adj_matrices(mol = mol,n_atom_types=23,max_n_atoms = max_n_atoms, K=K,L=L,EGCNN_flag=True)
  return (A, V, H, R, mol.GetNumAtoms())


def featurize(rdkit_mols, max_n_atoms, K, L):
    cnt = 0
    for idx, mol in enumerate(rdkit_mols):
        if mol.GetNumAtoms() < max_n_atoms:
            cnt = cnt + 1
    featurized_mols = np.empty((cnt), dtype=object)
    for idx, mol in enumerate(rdkit_mols):
        if mol.GetNumAtoms() < max_n_atoms:
            featurized_mol = ecnn_featurize_mol(mol, L, K, max_n_atoms)
            featurized_mols[idx] = featurized_mol
    return (featurized_mols)


def featurize_smiles_df(df, max_n_atoms, K, L, log_every_N=1000):
  sample_elems = df['smiles'].tolist()
  features = []
  import numpy as np
  for ind, elem in enumerate(sample_elems):
    if ind % log_every_N == 0:
      log("Featurizing sample %d" % ind, True)
    mol = Chem.MolFromSmiles(elem)
    if mol:
      features.append(featurize([mol],max_n_atoms,K,L))
    else:
      features.append(np.empty(1))

  valid_inds = np.array([1 if elt.size > 0 else 0 for elt in features], dtype=bool)
  features = [elt for (is_valid, elt) in zip(valid_inds, features) if is_valid]
  return np.squeeze(np.array(features), axis=1), valid_inds


def convert_df_to_numpy(df, tasks):

  n_samples = df.shape[0]
  n_tasks = len(tasks)

  y = np.hstack(
      [np.reshape(np.array(df[task].values), (n_samples, 1)) for task in tasks])

  w = np.ones((n_samples, n_tasks))
  missing = np.zeros_like(y).astype(int)

  for ind in range(n_samples):
    for task in range(n_tasks):
      if y[ind, task] == "":
        missing[ind, task] = 1

  for ind in range(n_samples):
    for task in range(n_tasks):
      if missing[ind, task]:
        y[ind, task] = 0.
        w[ind, task] = 0.

  return y.astype(float), w.astype(float)


def shard_generator(max_n_atoms, K, L, tasks, dataset_file, shard_size=8192):
    if not isinstance(dataset_file, list):
        dataset_file = [dataset_file]
    for shard_num, shard in enumerate(load_csv_files(dataset_file, shard_size,verbose=True)):
        time1 = time.time()
        X, valid_inds = featurize_smiles_df(shard,max_n_atoms,K,L)
        ids = shard['smiles'].values
        ids = ids[valid_inds]
        if len(tasks) > 0:
            y, w = convert_df_to_numpy(shard, tasks)
            y, w = (y[valid_inds], w[valid_inds])
            assert len(X) == len(ids) == len(y) == len(w)
        else:
            y, w = (None, None)
            assert len(X) == len(ids)
        time2 = time.time()
        log("TIMING: featurizing shard %d took %0.3f s" %
            (shard_num, time2 - time1), True)
    yield X, y, w, ids


def get_data_dir():
  """Get the DeepChem data directory."""
  if 'DEEPCHEM_DATA_DIR' in os.environ:
    return os.environ['DEEPCHEM_DATA_DIR']
  return tempfile.gettempdir()

def load_dataset_from_disk(save_dir):
  train_dir = os.path.join(save_dir, "train_dir")
  valid_dir = os.path.join(save_dir, "valid_dir")
  test_dir = os.path.join(save_dir, "test_dir")
  if not os.path.exists(train_dir) or not os.path.exists(
      valid_dir) or not os.path.exists(test_dir):
    return False, None, list()
  loaded = True
  train = DiskDataset(train_dir)
  valid = DiskDataset(valid_dir)
  test = DiskDataset(test_dir)
  all_dataset = (train, valid, test)
  with open(os.path.join(save_dir, "transformers.pkl"), 'rb') as f:
    transformers = pickle.load(f)
    return loaded, all_dataset, transformers

def save_dataset_to_disk(save_dir, train, valid, test, transformers):
  train_dir = os.path.join(save_dir, "train_dir")
  valid_dir = os.path.join(save_dir, "valid_dir")
  test_dir = os.path.join(save_dir, "test_dir")
  train.move(train_dir)
  valid.move(valid_dir)
  test.move(test_dir)
  with open(os.path.join(save_dir, "transformers.pkl"), 'wb') as f:
    pickle.dump(transformers, f)
  return None

def load_data(K, L, max_atoms, split='index'):
    tox21_tasks = [
        'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
        'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
    ]

    data_dir = get_data_dir()

    save_dir = os.path.join(data_dir, "tox21/ECNN/" + split)

    loaded, all_dataset, transformers = load_dataset_from_disk(save_dir)

    if loaded:
        return tox21_tasks, all_dataset, transformers

    dataset_file = os.path.join(data_dir, "tox21.csv.gz")

    dataset = DiskDataset.create_dataset(
        shard_generator(max_atoms, K, L, tox21_tasks, dataset_file, shard_size=8192),
        tasks=tox21_tasks,
        verbose=True)

    transformer = BalancingTransformer(transform_w=True, dataset=dataset)

    dataset = transformer.transform(dataset)

    splitter = IndexSplitter()

    train, valid, test = splitter.train_valid_test_split(dataset)

    all_dataset = (train, valid, test)

    save_dataset_to_disk(save_dir, train, valid, test, transformers)

    return tox21_tasks, all_dataset, transformers


def dense(x,out_channel):
  activation_fn=tf.nn.relu
  biases_initializer = tf.zeros_initializer
  weights_initializer = tf.contrib.layers.variance_scaling_initializer
  y = tf.contrib.layers.fully_connected(x,
                                  num_outputs=out_channel,
                                  activation_fn=activation_fn,
                                  biases_initializer=biases_initializer,
                                  weights_initializer=weights_initializer(),
                                  trainable=True)
  return y


def ecnn_conv(num_filters,map_fn_flag,regul_scale,N_dynamic,padding_step,N_actual,S,V, A, H,R, mask):
    K = A.get_shape()[2].value  # number of edge type
    C = V.get_shape()[2].value  # (layer l-1) node feature vector dimension
    J = num_filters  # (layer l) node feature vector dimension
    L = H.get_shape()[-1].value  # edge feature vector dimension
    import math
    W = tf.get_variable(
        'conv_weights', [K, L, C, J],
        initializer=tf.truncated_normal_initializer(stddev=math.sqrt(1.0 / (K * (C + 1) * (L + 1) * 1.0))),
        regularizer=tf.contrib.layers.l2_regularizer(scale=regul_scale), dtype=tf.float32)
    b = tf.get_variable(
        'conv_bias', [num_filters],
        initializer=tf.constant_initializer(0.1),
        dtype=tf.float32)
    W_reshape = tf.reshape(W, shape=[L * K, -1])

    if map_fn_flag:
        def map_fn_func(x):
            N_a = x[3]
            N_a = tf.reshape(N_a, shape=[-1])
            padding_step = x[4]
            padding_step = tf.reshape(padding_step, shape=[-1])
            s = tf.concat([N_a, tf.Variable([-1])], axis=0)
            s = tf.concat([s, N_a], axis=0)
            A_item = tf.slice(x[0], [0, 0, 0], s)
            s = tf.concat([N_a, N_a], axis=0)
            s = tf.concat([s, tf.Variable([-1])], axis=0)
            H_item = tf.slice(x[1], [0, 0, 0], s)
            s = tf.concat([N_a, N_a], axis=0)
            R_item = tf.slice(x[2], [0, 0], s)
            s = tf.concat([N_a, tf.Variable([-1])], axis=0)
            V_item = tf.slice(x[5], [0, 0], s)
            A_item_reshape = tf.expand_dims(A_item, axis=-1)
            H_item_reshape = tf.expand_dims(H_item, axis=-3)
            A_H_item = tf.multiply(A_item_reshape, H_item_reshape)
            A_H_item_reshape = tf.concat([A_H_item[:, k, :, :] for k in range(K)], axis=2)
            A_H_item_reshape = tf.reshape(A_H_item_reshape, shape=[-1, L * K])  # concat all dimension except last(-1)
            A_H_W_item = tf.matmul(A_H_item_reshape, W_reshape)
            s = tf.concat([N_a, N_a], axis=0)
            s = tf.concat([s, tf.Variable([C, J])], axis=0)
            A_H_W_item_reshape = tf.reshape(A_H_W_item, shape=s)
            R_item = tf.expand_dims(R_item, axis=-1)
            R_item = tf.expand_dims(R_item, axis=-1)
            V_item = tf.expand_dims(V_item, axis=0)
            V_item = tf.expand_dims(V_item, axis=-1)
            A_H_W_V_item = tf.multiply(A_H_W_item_reshape, tf.multiply(R_item, V_item))
            init_step = tf.constant([0, 0, 0, 0])
            step = tf.concat([tf.constant([0]), padding_step], axis=0)
            step = tf.concat([step, step], axis=0)
            step = tf.concat([step, init_step], axis=0)
            step = tf.reshape(step, shape=[-1, 2])
            A_H_W_V_item = tf.pad(A_H_W_V_item, step, "CONSTANT")
            return A_H_W_V_item, H_item, H_item, H_item, H_item, H_item

        A_H_W_V, _, _, _, _, _ = tf.map_fn(lambda x: map_fn_func(x), (A, H, R, N_actual, padding_step, V))
        A_H_W_V = tf.concat(A_H_W_V, axis=0)
        A_H_W_V = tf.nn.relu(tf.reduce_sum(A_H_W_V, axis=[2, 3]) + b)
    else:
        A_reshape = tf.expand_dims(A, axis=-1)
        H_reshape = tf.expand_dims(H, axis=-3)
        A_H = tf.multiply(A_reshape, H_reshape)
        A_H_reshape = tf.concat([A_H[:, :, k, :, :] for k in range(K)], axis=3)
        A_H_reshape = tf.reshape(A_H_reshape, shape=[-1, L * K])  # concat all dimension except last(-1)
        A_H_W = tf.matmul(A_H_reshape, W_reshape)
        R = tf.reshape(R, shape=[-1])
        R = tf.expand_dims(R, axis=-1)
        A_H_W_R = tf.multiply(A_H_W, R)
        s = tf.concat([tf.Variable([1]), N_dynamic], axis=0)
        s = tf.concat([s, tf.Variable([J])], axis=0)
        V = tf.tile(V, s)
        V = tf.reshape(V, shape=[-1, C * J])
        result = tf.multiply(A_H_W_R, V)
        A_H_W_V = tf.reshape(result, shape=[-1, C])
        A_H_W_V = tf.reduce_sum(A_H_W_V, axis=1)
        s = tf.concat([S, tf.Variable([J])], axis=0)
        A_H_W_V = tf.reshape(A_H_W_V, shape=s)  # -1,N,N,J
        A_H_W_V = tf.reduce_sum(A_H_W_V, axis=-2)
        A_H_W_V = tf.nn.relu(A_H_W_V + b)
    result = A_H_W_V
    return result


def ecnn_agg(regul_scale,nodeTypeNum,egcnn_conv,VType_list):
    V = egcnn_conv
    w_VType_fc = tf.get_variable(
        'agg_w_VType_fc_task', [nodeTypeNum, 1],
        initializer=tf.truncated_normal_initializer(stddev=math.sqrt(1.0 / (nodeTypeNum * 1.0))),
        regularizer=tf.contrib.layers.l2_regularizer(scale=regul_scale),dtype=tf.float32)
    agg_max = tf.reduce_max(V, axis=1)
    VType_list = tf.expand_dims(VType_list, axis=-1)
    V = tf.expand_dims(V, axis=1)
    V_sum = tf.reduce_sum(tf.multiply(VType_list, V), axis=2)
    agg_node_label = tf.reduce_sum(tf.multiply(V_sum, w_VType_fc), axis=1)
    agg = tf.concat([agg_max, agg_node_label], axis=1)
    result = agg
    return result


tasks=[
    'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
    'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
]

K = 1
L = 1
nodeTypeNum = 44
map_fn_flag = False
regul_scale = 0.005
batch_size = 2
N = 50
model_dir = "tmp/graphconv"
batch_inner_dynamic_pad = True
if batch_inner_dynamic_pad:
    max_atoms = None
else:
    max_atoms = 150

_, tox21_datasets, _ = load_data(K, L, N)
train_dataset, valid_dataset, test_dataset = tox21_datasets
print(train_dataset.data_dir)
print(valid_dataset.data_dir)

Padding_step_features = tf.placeholder(shape=(None,), dtype=tf.int32, name="Padding_step_features")
N_actual_features = tf.placeholder(shape=(None,), dtype=tf.int32, name="N_actual_features")
N_dynamic_features = tf.placeholder(shape=(None,), dtype=tf.int32, name="N_dynamic_features")
Shape_features = tf.placeholder(shape=([3]), dtype=tf.int32, name="Shape_features")
V_features = tf.placeholder(shape=(None, max_atoms, 75), dtype=tf.float32, name="V_feature")
A_features = tf.placeholder(shape=(None, max_atoms, K, max_atoms), dtype=tf.float32, name="A_feature")
H_features = tf.placeholder(shape=(None, max_atoms, max_atoms, L), dtype=tf.float32, name="H")
R_features = tf.placeholder(shape=(None, max_atoms, max_atoms), dtype=tf.float32, name="R")
VType_list_features = tf.placeholder(shape=(None, nodeTypeNum, max_atoms), dtype=tf.float32, name="VType_list_features")
mask_features = tf.placeholder(shape=(None, max_atoms, 1), dtype=tf.float32)
egcnn_conv = ecnn_conv(32,
                        map_fn_flag,
                        regul_scale,
                        N_dynamic_features,
                        Padding_step_features,
                        N_actual_features,
                        Shape_features,
                        V_features,
                        A_features,
                        H_features,
                        R_features,
                        mask_features)
egcnn_conv = tf.layers.batch_normalization(egcnn_conv)
readout = ecnn_agg(regul_scale, nodeTypeNum, egcnn_conv, VType_list_features)

costs = []
labels = []
for task in tasks:
    classification = dense(readout, 2)
    softmax = tf.contrib.layers.softmax(classification[:50, :])
    label = tf.placeholder(shape=(None, 2), dtype=tf.float32)
    labels.append(label)
    cost = tf.nn.softmax_cross_entropy_with_logits_v2(logits=softmax, labels=label)
    cost = cost + tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    costs.append(cost)

entropy = tf.stack(costs, axis=1)

task_weights = tf.placeholder(shape=(None, len(tasks)), dtype=tf.float32)

with tf.name_scope('loss'):
    cross_entropy = tf.reduce_sum(entropy * task_weights)

with tf.name_scope('adam_optimizer'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(2):
        for ind, (X_b, y_b, w_b, ids_b) in enumerate(
                train_dataset.iterbatches(
                    batch_size, pad_batches=True, deterministic=False)):
            train_dict = {}
            mask = np.zeros(shape=(batch_size, N, 1))
            VType_list = np.zeros([batch_size, nodeTypeNum, N], dtype=np.float32)
            VType_str = [
                'C',
                'N',
                'O',
                'S',
                'F',
                'Si',
                'P',
                'Cl',
                'Br',
                'Mg',
                'Na',
                'Ca',
                'Fe',
                'As',
                'Al',
                'I',
                'B',
                'V',
                'K',
                'Tl',
                'Yb',
                'Sb',
                'Sn',
                'Ag',
                'Pd',
                'Co',
                'Se',
                'Ti',
                'Zn',
                'H',  # H?
                'Li',
                'Ge',
                'Cu',
                'Au',
                'Ni',
                'Cd',
                'In',
                'Mn',
                'Zr',
                'Cr',
                'Pt',
                'Hg',
                'Pb',
                'Unknown'
            ]
            R1_Pooling = np.zeros([batch_size, N, N], dtype=np.float32)
            pooling_mask = np.zeros([batch_size, N, 1], dtype=np.float32)
            for i, smile_string in enumerate(ids_b):
                mol = Chem.MolFromSmiles(smile_string)
                for a_idx in range(0, mol.GetNumAtoms()):
                    atom = mol.GetAtomWithIdx(a_idx)
                    a_type = atom.GetSymbol()
                    if a_type in VType_str:
                        VType_list[i, VType_str.index(a_type), a_idx] = 1
                    else:
                        VType_list[i, -1, a_idx] = 1
                    neighbor_num = len(atom.GetNeighbors())
                    if neighbor_num > 1:
                        pooling_mask[i, a_idx, 0] = 1
                        for n_idx, neighbor in enumerate(atom.GetNeighbors()):
                            R1_Pooling[i, a_idx, neighbor.GetIdx()] = 1.0
            train_dict[task_weights] = w_b
            for index, label in enumerate(labels):
                train_dict[label] = to_one_hot(y_b[:, index])
            max_N = -1
            for i in range(batch_size):
                mask_size = X_b[i][4]
                mask[i][:mask_size][0] = 1
                if max_N < mask_size:
                    max_N = mask_size
            if not batch_inner_dynamic_pad:
                train_dict[A_features] = np.array([x[0] for x in X_b])
                train_dict[H_features] = np.array([x[2] for x in X_b])
                train_dict[R_features] = np.array([x[3] for x in X_b])
                train_dict[V_features] = np.array([x[1] for x in X_b])
                train_dict[Shape_features] = [-1, max_atoms, max_atoms]
                train_dict[VType_list_features] = VType_list
                train_dict[mask] = mask
                train_dict[N_dynamic_features] = np.array([N])
                train_dict[N_actual_features] = np.array([x[4] for x in X_b])
                train_dict[Padding_step_features] = np.array([(max_atoms - x[4]) for x in X_b])
            else:
                train_dict[A_features] = np.array([x[0][:max_N, :, :max_N] for x in X_b])
                train_dict[H_features] = np.array([x[2][:max_N, :max_N, :] for x in X_b])
                train_dict[R_features] = np.array([x[3][:max_N, :max_N] for x in X_b])
                train_dict[V_features] = np.array([x[1][:max_N, :] for x in X_b])
                train_dict[Shape_features] = [-1, max_N, max_N]
                train_dict[VType_list_features] = VType_list[:, :, :max_N]
                train_dict[mask_features] = mask[:, :max_N, :]
                train_dict[N_dynamic_features] = np.array([max_N])
                train_dict[N_actual_features] = np.array([x[4] for x in X_b])
                train_dict[Padding_step_features] = np.array([(max_N - x[4]) for x in X_b])
            train_loss=0
            sess.run(train_step, feed_dict=train_dict)
            train_loss=sess.run(cross_entropy, feed_dict=train_dict)
            train_loss = train_loss / batch_size
            print(train_loss)

            test_loss=0
            for ind, (X_b, y_b, w_b, ids_b) in enumerate(
                    test_dataset.iterbatches(
                        batch_size, pad_batches=True, deterministic=False)):
                test_dict = {}
                mask = np.zeros(shape=(batch_size, N, 1))
                VType_list = np.zeros([batch_size, nodeTypeNum, N], dtype=np.float32)
                VType_str = [
                    'C',
                    'N',
                    'O',
                    'S',
                    'F',
                    'Si',
                    'P',
                    'Cl',
                    'Br',
                    'Mg',
                    'Na',
                    'Ca',
                    'Fe',
                    'As',
                    'Al',
                    'I',
                    'B',
                    'V',
                    'K',
                    'Tl',
                    'Yb',
                    'Sb',
                    'Sn',
                    'Ag',
                    'Pd',
                    'Co',
                    'Se',
                    'Ti',
                    'Zn',
                    'H',  # H?
                    'Li',
                    'Ge',
                    'Cu',
                    'Au',
                    'Ni',
                    'Cd',
                    'In',
                    'Mn',
                    'Zr',
                    'Cr',
                    'Pt',
                    'Hg',
                    'Pb',
                    'Unknown'
                ]
                R1_Pooling = np.zeros([batch_size, N, N], dtype=np.float32)
                pooling_mask = np.zeros([batch_size, N, 1], dtype=np.float32)
                for i, smile_string in enumerate(ids_b):
                    mol = Chem.MolFromSmiles(smile_string)
                    for a_idx in range(0, mol.GetNumAtoms()):
                        atom = mol.GetAtomWithIdx(a_idx)
                        a_type = atom.GetSymbol()
                        if a_type in VType_str:
                            VType_list[i, VType_str.index(a_type), a_idx] = 1
                        else:
                            VType_list[i, -1, a_idx] = 1
                        neighbor_num = len(atom.GetNeighbors())
                        if neighbor_num > 1:
                            pooling_mask[i, a_idx, 0] = 1
                            for n_idx, neighbor in enumerate(atom.GetNeighbors()):
                                R1_Pooling[i, a_idx, neighbor.GetIdx()] = 1.0
                test_dict[task_weights] = w_b
                for index, label in enumerate(labels):
                    test_dict[label] = to_one_hot(y_b[:, index])
                max_N = -1
                for i in range(batch_size):
                    mask_size = X_b[i][4]
                    mask[i][:mask_size][0] = 1
                    if max_N < mask_size:
                        max_N = mask_size
                if not batch_inner_dynamic_pad:
                    test_dict[A_features] = np.array([x[0] for x in X_b])
                    test_dict[H_features] = np.array([x[2] for x in X_b])
                    test_dict[R_features] = np.array([x[3] for x in X_b])
                    test_dict[V_features] = np.array([x[1] for x in X_b])
                    test_dict[Shape_features] = [-1, max_atoms, max_atoms]
                    test_dict[VType_list_features] = VType_list
                    test_dict[mask] = mask
                    test_dict[N_dynamic_features] = np.array([N])
                    test_dict[N_actual_features] = np.array([x[4] for x in X_b])
                    test_dict[Padding_step_features] = np.array([(max_atoms - x[4]) for x in X_b])
                else:
                    test_dict[A_features] = np.array([x[0][:max_N, :, :max_N] for x in X_b])
                    test_dict[H_features] = np.array([x[2][:max_N, :max_N, :] for x in X_b])
                    test_dict[R_features] = np.array([x[3][:max_N, :max_N] for x in X_b])
                    test_dict[V_features] = np.array([x[1][:max_N, :] for x in X_b])
                    test_dict[Shape_features] = [-1, max_N, max_N]
                    test_dict[VType_list_features] = VType_list[:, :, :max_N]
                    test_dict[mask_features] = mask[:, :max_N, :]
                    test_dict[N_dynamic_features] = np.array([max_N])
                    test_dict[N_actual_features] = np.array([x[4] for x in X_b])
                    test_dict[Padding_step_features] = np.array([(max_N - x[4]) for x in X_b])

                test_loss+=sess.run(cross_entropy, feed_dict=test_dict)
            test_loss=test_loss/len(test_dataset)
            print(test_loss)