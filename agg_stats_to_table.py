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

"""Script to produce tables."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd

df_2d2c = pd.read_csv('cluster_2d_2c/results.csv')
df_2d3c = pd.read_csv('cluster_2d_3c/results.csv')
df_2d5c = pd.read_csv('cluster_2d_5c/results.csv')
df_2d10c = pd.read_csv('cluster_2d_10c/results.csv')
df_2d2c = pd.read_csv('cluster_2d_2c_arma_mle/results.csv')
df_2d3c = pd.read_csv('cluster_2d_3c_arma_mle/results.csv')
df_2d5c = pd.read_csv('cluster_2d_5c_arma_mle/results.csv')
df_2d10c = pd.read_csv('cluster_2d_10c_arma_mle/results.csv')
df_2d2c['num_clusters'] = 2
df_2d2c['hidden_dim'] = 2
df_2d3c['num_clusters'] = 3
df_2d3c['hidden_dim'] = 2
df_2d5c['num_clusters'] = 5
df_2d5c['hidden_dim'] = 2
df_2d10c['hidden_dim'] = 2
df_2d10c['num_clusters'] = 10
df = pd.concat([df_2d2c, df_2d3c, df_2d5c, df_2d10c])
df = df[df.method != 'true']
df = df[df.seq_len == 1000]

metric_names = ['adj_mutual_info', 'adj_rand_score', 'v_measure', 't_secs']
stats_list = []
for metric in metric_names:
  stats = df.groupby(['hidden_dim', 'num_clusters', 'seq_len',
                      'method'])[metric].agg(['mean', 'count', 'std'])
  ci95_hi = []
  ci95_lo = []
  mean_w_ci = []
  for i in stats.index:
    m, c, s = stats.loc[i]
    ci95_hi.append(m + 1.96 * s / np.sqrt(c))
    ci95_lo.append(m - 1.96 * s / np.sqrt(c))
    mean_w_ci.append('%.2f (%.2f-%.2f)' %
                     (m, m - 1.96 * s / np.sqrt(c), m + 1.96 * s / np.sqrt(c)))
  stats['ci95_hi'] = ci95_hi
  stats['ci95_lo'] = ci95_lo
  stats['mean_w_ci'] = mean_w_ci
  stats['metric'] = metric
  stats = stats.reset_index()
  stats = stats.reset_index().set_index(['method', 'num_clusters', 'metric'])
  stats.to_csv(metric + '_agg.csv')
  stats_list.append(stats['mean_w_ci'])
agg_df = pd.DataFrame(data={'val': pd.concat(stats_list)})
agg_df = agg_df.pivot_table(
    index=['num_clusters', 'method'],
    columns=['metric'],
    values='val',
    aggfunc=lambda x: ''.join(str(v) for v in x))
agg_df = agg_df[metric_names]
print(agg_df.to_latex())
