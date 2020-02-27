from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import logging
import os

from absl import app
from absl import flags
import matplotlib
matplotlib.use('Agg')
from matplotlib import pylab
import numpy as np
import pandas as pd
import seaborn as sns
import tqdm

import clustering
import lds

sns.set(style='whitegrid')

FLAGS = flags.FLAGS

# Flags for IO and plotting.
flags.DEFINE_string('output_dir', None, 'Output filepath.')
flags.DEFINE_boolean(
    'load_results', False, 'Whether to skip experiments '
    'and only plot existing results from output_dir.')

flags.DEFINE_integer('num_clusters', 2, 'Number of clusters in experiments.')
flags.DEFINE_integer('num_systems', 100,
                     'Number of dynamical systems to cluster.')
flags.DEFINE_integer('hidden_state_dim', 2, 'Hidden state dim in experiments.')
flags.DEFINE_integer('input_dim', 1, 'Input dim in experiments.')
flags.DEFINE_float(
    'cluster_center_dist_lower_bound', 0.2, 'Desired distance lower bound '
    'between cluster centers. Generate cluster centers until distance '
    'between cluster centers is above this value.')
flags.DEFINE_float('cluster_radius', 0.05,
                   'Radius of each dynamical system cluster.')
flags.DEFINE_integer('num_repeat', 1,
                     'Number of repeated runs for each fixed seq len.')

# Flags for output sequences from LDSs.
flags.DEFINE_integer('min_seq_len', 10, 'Min seq len in experiments.')
flags.DEFINE_integer('max_seq_len', 1000, 'Max seq len in experiments.')
flags.DEFINE_integer(
    'num_sampled_seq_len', 10, 'Number of sampled seq len '
    'values in between min and max seq len.')
flags.DEFINE_float('output_noise_stddev', 0.01, 'Output noise stddev.')

# Flags for hparams in clustering algorithms.
flags.DEFINE_integer('guessed_hidden_dim', 0,
                     'Assumed hidden dim. If 0, use true hidden dim.')
flags.DEFINE_integer(
    'guessed_num_clusters', 0,
    'Desired number of clusters. If 0, find best number '
    'adaptively from maximizing kmeans objective score.')


def get_results(cluster_center_dist_lower_bound, hidden_state_dim, input_dim,
        guessed_hidden_dim, num_clusters, guessed_num_clusters, min_seq_len,
        max_seq_len, num_sampled_seq_len, num_repeat, num_systems,
        cluster_radius, output_noise_stddev, results_path=None):
  """Get results for varying sequence lengths.

  Args:
    cluster_center_dist_lower_bound: Desired distance lower bound between
      clusters. When generating cluster centers, try repeatedly until distance
      is greater than cluster_center_dist_lower_bound.
    hidden_state_dim: True hidden state dim.
    input_dim: The input dim.
    guessed_hidden_dim: Assumed hidden dim. If 0, use true hidden dim.
    num_clusters: True number of clusters.
    guessed_num_clusters: Desired number of clusters. If 0, use true number.
    min_seq_len: Min seq len in experiments.
    max_seq_len: Max seq len in experiments.
    num_sampled_seq_len: Number of sampled seq len values in between min and max
      seq len.
    num_repeat: Number of repeated experiments for each seq_len.
    num_systems: Number of dynamical system in each clustering experiments.
    cluster_radius: Expected distance of generated systems from cluster centers.
    output_noise_stddev: Scalar.

  Returns:
    A pandas DataFrame with columns `method`, `seq_len`, `t_secs`,
    `failed_ratio`, and columns for clustering metrics such as `adj_mutual_info`
    and `v_measure`. The same method and seq_len will appear in num_repeat many
    rows.
  """
  np.random.seed(0)
  progress_bar = tqdm.tqdm(total=num_repeat * num_sampled_seq_len)
  # Generator for output sequences.
  gen = lds.SequenceGenerator(output_noise_stddev=output_noise_stddev)
  seq_len_vals = np.linspace(min_seq_len, max_seq_len, num_sampled_seq_len)
  seq_len_vals = [int(round(x)) for x in seq_len_vals]
  if guessed_hidden_dim == 0:
    guessed_hidden_dim = hidden_state_dim
  if guessed_num_clusters == 0:
    guessed_num_clusters = num_clusters
  results_dfs = []
  for i in range(num_repeat):
    logging.info('---Starting experiments in repeat run #%d---', i)
    cluster_centers = clustering.generate_cluster_centers(num_clusters,
            hidden_state_dim, input_dim, cluster_center_dist_lower_bound)
    true_systems, true_cluster_ids = clustering.generate_lds_clusters(
            cluster_centers, num_systems, cluster_radius)
    for seq_len in seq_len_vals:
      logging.info('Running experiment with seq_len = %d.', seq_len)
      seqs = [gen.generate_seq(s, seq_len=seq_len) for s in true_systems]
      # Get clustering results.
      results_df = clustering.get_results(
          seqs,
          guessed_num_clusters,
          guessed_hidden_dim,
          true_cluster_ids)
      results_df['seq_len'] = seq_len
      results_df['n_guessed_clusters'] = guessed_num_clusters
      results_df['n_true_clusters'] = num_clusters
      results_df['true_hidden_dim'] = hidden_state_dim
      results_df['guessed_hidden_dim'] = guessed_hidden_dim
      results_dfs.append(results_df)
      logging.info('Results:\n%s', str(results_df))
      plot_filepath = os.path.join(
          FLAGS.output_dir,
          'cluster_visualization_run_%d_seq_len_%d.png' % (i, seq_len))
      progress_bar.update(1)
    if results_path:
      with open(results_path, 'w+') as f:
        pd.concat(results_dfs).to_csv(f, index=False)
  progress_bar.close()
  return pd.concat(results_dfs)


def plot_results(results_df, output_dir):
  """Plots metrics and saves plots as png files."""
  for metric_name in results_df.columns:
    if metric_name == 'seq_len' or metric_name == 'method':
      continue
    # Other than the silhouette metric, the metric value for ground truth is
    # always 1 for adj_mutual_info, adj_rand_score etc., so skip ground truth
    # in plotting.
    if metric_name != 'silhouette':
      results_df = results_df[results_df.method != 'true']
    pylab.figure()
    sns.pointplot(
        x='seq_len',
        y=metric_name,
        data=results_df,
        hue='method',
        scale=0.5,
        estimator=np.mean,
        err_style='bars',
        capsize=.1)
    pylab.savefig(os.path.join(output_dir, metric_name + '.png'))


def main(unused_argv):
  if FLAGS.load_results:
    with open(os.path.join(FLAGS.output_dir, 'results.csv'), 'r') as f:
      combined_result_df = pd.read_csv(f, index_col=False)
    plot_results(combined_result_df, FLAGS.output_dir)
    return

  if not os.path.exists(FLAGS.output_dir):
    os.mkdir(FLAGS.output_dir)
  with open(os.path.join(FLAGS.output_dir, 'flags.txt'), 'w+') as f:
    f.write(str(FLAGS.flag_values_dict()))
  results_path = os.path.join(FLAGS.output_dir, 'results_inc.csv')
  df = get_results(
      cluster_center_dist_lower_bound=FLAGS.cluster_center_dist_lower_bound,
      hidden_state_dim=FLAGS.hidden_state_dim,
      input_dim=FLAGS.input_dim,
      guessed_hidden_dim=FLAGS.guessed_hidden_dim,
      num_clusters=FLAGS.num_clusters,
      guessed_num_clusters=FLAGS.guessed_num_clusters,
      min_seq_len=FLAGS.min_seq_len,
      max_seq_len=FLAGS.max_seq_len,
      num_sampled_seq_len=FLAGS.num_sampled_seq_len,
      num_repeat=FLAGS.num_repeat,
      num_systems=FLAGS.num_systems,
      cluster_radius=FLAGS.cluster_radius,
      output_noise_stddev=FLAGS.output_noise_stddev,
      results_path=results_path)
  with open(os.path.join(FLAGS.output_dir, 'results.csv'), 'w+') as f:
    df.to_csv(f, index=False)
  plot_results(df, FLAGS.output_dir)


if __name__ == '__main__':
  flags.mark_flag_as_required('output_dir')
  app.run(main)
