
from rdkit import Chem

import numpy as np
from graph_features import atom_features,one_of_k_encoding_unk

class Featurizer(object):

  def featurize(self, mols, verbose=True, log_every_n=1000):
    mols = list(mols)
    features = []
    for i, mol in enumerate(mols):
      if mol is not None:
        features.append(self._featurize(mol))
      else:
        features.append(np.array([]))

    features = np.asarray(features)
    return features

  def _featurize(self, mol):
    raise NotImplementedError('Featurizer is not defined.')

  def __call__(self, mols):
    return self.featurize(mols)

def get_atom_type(atom):
  elem = atom.GetAtomicNum()
  hyb = str(atom.GetHybridization).lower()
  if elem == 1:
    return (0)
  if elem == 4:
    return (1)
  if elem == 5:
    return (2)
  if elem == 6:
    if "sp2" in hyb:
      return (3)
    elif "sp3" in hyb:
      return (4)
    else:
      return (5)
  if elem == 7:
    if "sp2" in hyb:
      return (6)
    elif "sp3" in hyb:
      return (7)
    else:
      return (8)
  if elem == 8:
    if "sp2" in hyb:
      return (9)
    elif "sp3" in hyb:
      return (10)
    else:
      return (11)
  if elem == 9:
    return (12)
  if elem == 15:
    if "sp2" in hyb:
      return (13)
    elif "sp3" in hyb:
      return (14)
    else:
      return (15)
  if elem == 16:
    if "sp2" in hyb:
      return (16)
    elif "sp3" in hyb:
      return (17)
    else:
      return (18)
  if elem == 17:
    return (19)
  if elem == 35:
    return (20)
  if elem == 53:
    return (21)
  return (22)


def get_atom_adj_matrices(mol,
                          n_atom_types,
                          max_n_atoms=150,
                          max_valence=6,
                          K = 1,
                          L = 1,
                          graph_conv_features=True,
                          EGCNN_flag = False,
                          nxn=True):
  if not graph_conv_features:
    bond_matrix = np.zeros((max_n_atoms, 4 * max_valence)).astype(np.uint8)
  if EGCNN_flag:
    EGCNN_A = np.zeros((max_n_atoms, K, max_n_atoms)).astype(np.uint8)
    EGCNN_H = np.zeros((max_n_atoms, max_n_atoms, L)).astype(np.uint8)
    for b in mol.GetBonds():
      t = b.GetBondType()
      k = 0
      if (K != 1):
        if t == Chem.rdchem.BondType.SINGLE:
          k = 0
        elif t == Chem.rdchem.BondType.DOUBLE:
          k = 1
        elif t == Chem.rdchem.BondType.TRIPLE:
          k = 2
        elif t == Chem.rdchem.BondType.AROMATIC:
          k = 3
        else:
          k= 3
      EGCNN_A[b.GetEndAtomIdx(),k ,b.GetBeginAtomIdx()] = 1
      EGCNN_A[b.GetBeginAtomIdx(), k, b.GetEndAtomIdx()] = 1
      source_atom = mol.GetAtomWithIdx(b.GetBeginAtomIdx())
      target_atom = mol.GetAtomWithIdx(b.GetEndAtomIdx())
      source_atom_valence = source_atom.GetImplicitValence()
      target_atom_valence = target_atom.GetImplicitValence()
      edge_vector = one_of_k_encoding_unk(source_atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6])+one_of_k_encoding_unk(target_atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6])
      edge_vector = np.array(edge_vector,dtype=np.int32)
      if (L != 1):
        # l_edge_vector_bin = list(bin(source_atom_valence)[2:])
        # r_edge_vector_bin = list(bin(target_atom_valence)[2:])
        # for i in range(L//2):
        #   if i<len(l_edge_vector_bin):
        #     edge_vector[i+L//2] = l_edge_vector_bin[i]
        # for i in range(L//2):
        #   if i<len(r_edge_vector_bin):
        #     edge_vector[i] = r_edge_vector_bin[i]
        EGCNN_H[b.GetBeginAtomIdx(), b.GetEndAtomIdx(), :] = edge_vector
        EGCNN_H[b.GetEndAtomIdx(), b.GetBeginAtomIdx(), :] = edge_vector
      else:
        EGCNN_H[b.GetBeginAtomIdx(), b.GetEndAtomIdx(), :] = 1
        EGCNN_H[b.GetEndAtomIdx(), b.GetBeginAtomIdx(), :] = 1
    EGCNN_R = np.zeros((max_n_atoms, max_n_atoms)).astype(np.uint8)
    for a_idx in range(0, mol.GetNumAtoms()):
      atom = mol.GetAtomWithIdx(a_idx)
      neighbor_num = len(atom.GetNeighbors())
      for n_idx, neighbor in enumerate(atom.GetNeighbors()):
        EGCNN_R[a_idx, neighbor.GetIdx()] = 1.0 / neighbor_num
      EGCNN_R[a_idx,a_idx] = 1.0
  if nxn:
    adj_matrix = np.zeros((max_n_atoms, max_n_atoms)).astype(np.uint8)
  else:
    adj_matrix = np.zeros((max_n_atoms, max_valence)).astype(np.uint8)
    adj_matrix += (adj_matrix.shape[0] - 1)

  if not graph_conv_features:
    atom_matrix = np.zeros((max_n_atoms, n_atom_types + 3)).astype(np.uint8)
    atom_matrix[:, atom_matrix.shape[1] - 1] = 1

  atom_arrays = []
  for a_idx in range(0, mol.GetNumAtoms()):
    atom = mol.GetAtomWithIdx(a_idx)
    if graph_conv_features:
      atom_arrays.append(atom_features(atom))
      #atom_arrays.append(atom_features(atom))

    else:

      atom_type = get_atom_type(atom)
      atom_matrix[a_idx][-1] = 0
      atom_matrix[a_idx][atom_type] = 1

    for n_idx, neighbor in enumerate(atom.GetNeighbors()):
      if nxn:
        adj_matrix[a_idx][neighbor.GetIdx()] = 1
        adj_matrix[a_idx][a_idx] = 1
      else:
        adj_matrix[a_idx][n_idx] = neighbor.GetIdx()

      if not graph_conv_features:
        bond = mol.GetBondBetweenAtoms(a_idx, neighbor.GetIdx())
        bond_type = str(bond.GetBondType()).lower()
        if "single" in bond_type:
          bond_order = 0
        elif "double" in bond_type:
          bond_order = 1
        elif "triple" in bond_type:
          bond_order = 2
        elif "aromatic" in bond_type:
          bond_order = 3
        bond_matrix[a_idx][(4 * n_idx) + bond_order] = 1

  if graph_conv_features:
    n_feat = len(atom_arrays[0])
    atom_matrix = np.zeros((max_n_atoms, n_feat)).astype(np.uint8)
    for idx, atom_array in enumerate(atom_arrays):
      atom_matrix[idx, :] = atom_array
  else:
    atom_matrix = np.concatenate(
        [atom_matrix, bond_matrix], axis=1).astype(np.uint8)
  if EGCNN_flag:
    return (EGCNN_A.astype(np.uint8), atom_matrix.astype(np.uint8),EGCNN_H.astype(np.uint8),EGCNN_R.astype(np.uint8))
  else:
    return (adj_matrix.astype(np.uint8), atom_matrix.astype(np.uint8))


def featurize_mol(mol, n_atom_types, max_n_atoms, max_valence,
                  num_atoms_feature):
  adj_matrix, atom_matrix = get_atom_adj_matrices(mol, n_atom_types,
                                                  max_n_atoms, max_valence)
  if num_atoms_feature:
    return ((adj_matrix, atom_matrix, mol.GetNumAtoms()))
  return ((adj_matrix, atom_matrix))


def EGCNN_featurize_mol(mol,L,K,max_n_atoms):
  A,V,H,R = get_atom_adj_matrices(mol = mol,n_atom_types=23,max_n_atoms = max_n_atoms,K=K,L=L,EGCNN_flag=True)
  return (A,V,H,R,mol.GetNumAtoms())


class EGCNNFeaturizer(Featurizer):

  def __init__(self,
               max_n_atoms=150,
               K=1,
               L=1,
               ):
    self.max_n_atoms = max_n_atoms
    self.K = K
    self.L = L

  def featurize(self, rdkit_mols):
    cnt = 0
    for idx, mol in enumerate(rdkit_mols):
      if mol.GetNumAtoms() < self.max_n_atoms:
        cnt = cnt + 1
    featurized_mols = np.empty((cnt), dtype=object)
    # featurized_mols = np.empty((len(rdkit_mols)), dtype=object)
    for idx, mol in enumerate(rdkit_mols):
      if mol.GetNumAtoms() < self.max_n_atoms:
        featurized_mol = EGCNN_featurize_mol(mol, self.L, self.K, self.max_n_atoms)
        featurized_mols[idx] = featurized_mol
    return (featurized_mols)
