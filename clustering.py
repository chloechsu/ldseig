# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utils for experiments on clustering linear dynamical systems."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from functools import partial
import logging
import timeit
import warnings

from dtaidistance import dtw
import numpy as np
import pandas as pd
from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.utils.metric import type_metric, distance_metric;
import sklearn
from sklearn import cluster
from sklearn import metrics
import tslearn
from tslearn import clustering as tslearn_clustering

import arma
import lds

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=PendingDeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


def pad_seqs_to_matrix(seqs, max_seq_len=None):
  if max_seq_len is None:
    max_seq_len = np.max([s.seq_len for s in seqs])
  output_dim = seqs[0].output_dim
  padded = np.zeros((len(seqs), max_seq_len * output_dim))
  for i, s in enumerate(seqs):
    for j in range(output_dim):
      padded[i, max_seq_len * j:max_seq_len * j + s.seq_len] = s.outputs[:, j]
  return padded


def kshape(seqs, n_clusters):
  # kShape is supposed to work on monomodal data.
  max_seq_len = np.max([s.seq_len for s in seqs])
  x_train = pad_seqs_to_matrix(seqs)[:, :max_seq_len]
  ts = tslearn.utils.to_time_series_dataset(x_train)
  return tslearn_clustering.KShape(n_clusters=n_clusters,
          verbose=False).fit_predict(ts)


def dtw_kmedoids(seqs, n_clusters):
  max_seq_len = np.max([s.seq_len for s in seqs])
  n_seqs = len(seqs)
  x_train = pad_seqs_to_matrix(seqs)[:, :max_seq_len]
  dtw_distance_matrix = dtw.distance_matrix_fast(x_train)
  # The returned distance matrix is only upper triangular.
  # Post-process to fill in the distance matrix via symmetry.
  for i in range(n_seqs):
    dtw_distance_matrix[i][i] = 0.0
    for j in range(i):
      dtw_distance_matrix[i][j] = dtw_distance_matrix[j][i]
  initial_medoids = np.random.choice(n_seqs, n_clusters, replace=False)
  kmedoids_instance = kmedoids(dtw_distance_matrix, initial_medoids,
      data_type='distance_matrix')
  kmedoids_instance.process()
  clusters = kmedoids_instance.get_clusters()
  # The returned cluster info is a list of the indices in each cluster.
  cluster_ids = np.zeros(n_seqs, dtype=int)
  for cluster_id, cluster_elems in enumerate(clusters):
    for idx in cluster_elems:
      cluster_ids[idx] = cluster_id
  return cluster_ids


def ar_kmeans(seqs, n_clusters, hdim):
  ar_coeffs = np.stack([arma.fit_ar(
      s.outputs, None, hdim).flatten() for s in seqs], axis=0)
  return cluster.KMeans(n_clusters=n_clusters).fit_predict(ar_coeffs)


def arma_iter_kmeans(seqs, n_clusters, hdim):
  ar_coeffs = np.stack([arma.fit_arma_iter(s.outputs, None, hdim,
      l2_reg=0.01).flatten() for s in seqs], axis=0)
  return cluster.KMeans(n_clusters=n_clusters).fit_predict(ar_coeffs)


def arma_mle_kmeans(seqs, n_clusters, hdim):
  arma_params = []
  for s in seqs:
    arparams, maparams = arma.fit_arma_mle(s.outputs, None, hdim)
    arma_params.append(np.concatenate([arparams, maparams]).flatten())
  return cluster.KMeans(n_clusters=n_clusters).fit_predict(
      np.stack(arma_params, axis=0))


def pca_kmeans(seqs, n_clusters, hdim):
  pca_model = sklearn.decomposition.PCA(n_components=hdim)
  pca = pca_model.fit_transform(np.stack([s.outputs.flatten() for s in seqs],
      axis=0))
  return cluster.KMeans(n_clusters=n_clusters).fit_predict(pca)


def lds_em_kmeans(seqs, n_clusters, hdim):
  lds_eigs = np.stack([lds.fit_lds_pylds(
      s.outputs, None, hdim).flatten() for s in seqs], axis=0)
  return cluster.KMeans(n_clusters=n_clusters).fit_predict(
      np.concatenate([lds_eigs.real, lds_eigs.imag], axis=1))


def generate_cluster_centers(num_clusters, hidden_state_dim, input_dim,
        cluster_center_dist_lower_bound):
  """Generates cluster center eigenvalues with distance requirement.

  The generated eigenvalues are drawn uniformly from (-1, 1) until the
  pairwise distance between cluster center eigenvalues >= lower bound.
  """
  min_cluster_center_dist = -1.
  while min_cluster_center_dist < cluster_center_dist_lower_bound:
    cluster_centers = []
    for _ in range(num_clusters):
      c = lds.generate_linear_dynamical_system(hidden_state_dim, input_dim)
      cluster_centers.append(c)
    min_cluster_center_dist = np.inf
    for s1 in range(num_clusters):
      for s2 in range(s1 + 1, num_clusters):
        d = lds.eig_dist(cluster_centers[s1], cluster_centers[s2])
        if d < min_cluster_center_dist:
          min_cluster_center_dist = d
  logging.info('generated min_cluster_center_dist %.2f',
               min_cluster_center_dist)
  return cluster_centers


def generate_lds_clusters(cluster_centers, num_systems, cluster_radius):
  """Generates clusters of linear dynamical systems.

  Args:
    cluster_centers: A list of LinearDynamicalSystem instances.
    num_systems: Total number of systems in all clusters.
    cluster_radius: Desired mean distance from the centers.

  Returns:
    - A list of LinearDynamicalSystem of size num_systems.
    - A list of true cluster ids.
  """
  num_clusters = len(cluster_centers)
  cluster_id = np.random.randint(0, num_clusters, num_systems)
  hidden_state_dim = cluster_centers[0].hidden_state_dim
  for c in cluster_centers:
    if c.hidden_state_dim != hidden_state_dim:
      raise ValueError('Hidden state dimension mismatch.')
  generated_systems = []
  dist_to_center = np.zeros(num_systems)
  for i in range(num_systems):
    c = cluster_centers[cluster_id[i]]
    eigvalues_new = c.get_spectrum() + cluster_radius / np.sqrt(
        hidden_state_dim) * np.random.randn(hidden_state_dim)
    generated_systems.append(
        lds.generate_linear_dynamical_system(
            hidden_state_dim, eigvalues=eigvalues_new))
    dist_to_center[i] = lds.eig_dist(c, generated_systems[-1])

  # For logging purpose.
  dist_bw_centers = np.zeros((num_clusters, num_clusters))
  for i in range(num_clusters):
    for j in range(num_clusters):
      dist_bw_centers[i, j] = lds.eig_dist(cluster_centers[i],
                                           cluster_centers[j])
  logging.info('Distances between cluster centers:\n%s', str(dist_bw_centers))
  logging.info('Average distance from cluster centers: %.3f',
               np.average(dist_to_center))
  for i in range(num_clusters):
    logging.info('Eigenvalues of cluster center %d: %s', i,
                 str(cluster_centers[i].get_spectrum()))

  return generated_systems, cluster_id


def get_results(sequences, num_clusters, guessed_hidden_dim, true_cluster_ids):
  """Compares clustering results with different methods.

  See Section 2.3.9 in https://scikit-learn.org/stable/modules/clustering.html
  for more details on metrics.

  Args:
    sequences: A list of LinearDynamicalSystemSequence objects.
    num_clusters: Desired number of clusters, may differ from ground truth.
    guessed_hidden_dim: Hidden dim for lds and arma methods.
    true_cluster_ids: Ground truth from generated data.

  Returns:
    A pandas DataFrame with columns `method`, `t_secs`, `failed_ratio`, and
    columns for clustering metrics such as `adj_mutual_info` and `v_measure`.
  """
  cluster_fns = {
      'true': (lambda a, n: true_cluster_ids),
      'kshape': kshape,
      'dtw': dtw_kmedoids,
      'pca': partial(pca_kmeans, hdim=guessed_hidden_dim),
      'ar': partial(ar_kmeans, hdim=guessed_hidden_dim), 
      'arma_iter': partial(arma_iter_kmeans, hdim=guessed_hidden_dim),
      'lds_em': partial(lds_em_kmeans, hdim=guessed_hidden_dim),
  }
  metric_fns_with_truth = {
      'adj_rand_score':
          metrics.adjusted_rand_score,
      'adj_mutual_info':
          metrics.adjusted_mutual_info_score,
      'fowlkes_mallows':
          metrics.fowlkes_mallows_score,
      'homogeneity':
          metrics.homogeneity_score,
      'completeness':
          metrics.completeness_score,
      'v_measure': (lambda t, pred: metrics.homogeneity_completeness_v_measure(
          t, pred)[2]),
  }

  # Computer clusters.
  cluster_ids = collections.OrderedDict()
  t_record = collections.OrderedDict()
  for k, fn in cluster_fns.items():
    logging.info('Running clustering method %s.', k)
    start_t = timeit.default_timer()
    cluster_ids[k] = fn(sequences, num_clusters)
    t_record[k] = timeit.default_timer() - start_t
    # In case there is an error from tslearn.
    if cluster_ids[k] is None:
      cluster_ids[k] = np.random.randint(
          low=0, high=num_clusters, size=len(sequences))
      logging.info('Warning: null cluster ids.')

  # Construct pandas dataframe with metrics.
  metric_dict = {k: [] for k in metric_fns_with_truth.keys()}
  metric_dict['t_secs'] = np.array(list(t_record.values()))
  metric_dict['method'] = []
  for k, pred_ids in cluster_ids.items():
    for metric_key, metric_fn in metric_fns_with_truth.items():
      try:
        metric_dict[metric_key].append(metric_fn(true_cluster_ids, pred_ids))
      except Exception as e:  # pylint: disable=broad-except
        metric_dict[metric_key].append(0.0)
        logging.info('Error computing %s: %s', metric_key)
    metric_dict['method'].append(k)
  return pd.DataFrame(data=metric_dict).set_index('method').reset_index()
