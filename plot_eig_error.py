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

"""Script to plot results on learning eigenvalues."""

import matplotlib
matplotlib.use('Agg')
from matplotlib import pylab
import numpy as np
import pandas as pd
import seaborn as sns

sns.set(style='whitegrid')
paper_rc = {'lines.linewidth': 1, 'lines.markersize': 10}
sns.set_context("paper", rc = paper_rc)

df_2d = pd.read_csv('eig_2d/eig_results.csv')
df_3d = pd.read_csv('eig_3d/eig_results.csv')
df_2d['hidden_dim'] = 2
df_3d['hidden_dim'] = 3
df = pd.concat([df_2d, df_3d])
df = df[df.method != 'true']
method_name_mapping = {
    'AR': 'AR',
    'ARMA_RLS': 'ARMA',
    'LDS_EM': 'LDS',
    'ARMA_MLE': 'ARMA_MLE',
}
df['method'] = df.method.map(method_name_mapping)
df['Hidden dim'] = df['hidden_dim']
df = df[~df.method.isnull()]
hue_order = ['AR', 'ARMA', 'LDS', 'ARMA_MLE']
g = sns.catplot(
    x='seq_len',
    y='l2_a_error',
    hue='method',
    sharex=False,
    col='Hidden dim',
    data=df,
    kind='point',
    capsize=.15,
    palette=sns.xkcd_palette(['purple', 'pale red', 'denim blue', 'grey']),
    height=4,
    aspect=1,
    hue_order=hue_order,
    ci=95,
    scale=1,
    markers=['x', 'v', '>', '.', 'o', '+', '<', '1', '2', '3', '4'],
    #linestyles=['--', '-.', '-', '--', '-.', '-'],
    join=True)
g.set_axis_labels('Sequence length',
                  'Abs. l-2 error in eigenvalues')
g.despine(left=True)
pylab.gcf().subplots_adjust(bottom=0.20)
pylab.savefig('eig_agg.png', dpi=300)
