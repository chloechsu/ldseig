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

"""Class and utils for linear dynamical systems."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import logging
import warnings

import numpy as np
from pylds.models import DefaultLDS
from scipy.stats import ortho_group


class LinearDynamicalSystem(object):
  """Class to represent a linear dynamical system."""

  def __init__(self, transition_matrix, input_matrix, output_matrix):
    """Initializes a linear dynamical system object.

    Args:
      transition_matrix: The transition matrix of shape [hidden_state_dim,
        hidden_state_dim].
      input_matrix: The input matrix of shape [hidden_state_dim, input_dim].
      output_matrix: The measurement matrix of shape [output_dim,
        hidden_state_dim].
    """
    self.hidden_state_dim = transition_matrix.shape[0]
    self.input_dim = input_matrix.shape[1]
    self.output_dim = output_matrix.shape[0]
    if transition_matrix.shape != (self.hidden_state_dim,
                                   self.hidden_state_dim):
      raise ValueError('Dimension mismatch.')
    if input_matrix.shape != (self.hidden_state_dim, self.input_dim):
      raise ValueError('Dimension mismatch.')
    if output_matrix.shape != (self.output_dim, self.hidden_state_dim):
      raise ValueError('Dimension mismatch.')
    self.transition_matrix = transition_matrix
    self.input_matrix = input_matrix
    self.output_matrix = output_matrix

  def get_spectrum(self):
    eigs = np.linalg.eig(self.transition_matrix)[0]
    return eigs[np.argsort(eigs.real)[::-1]]

  def get_expected_arparams(self):
    return -np.poly(self.get_spectrum())[1:]


class LinearDynamicalSystemSequence(object):
  """Wrapper around input seq, hidden state seq, and output seq from LDS."""

  def __init__(self, input_seq, hidden_state_seq, output_seq):
    self.seq_len = np.shape(input_seq)[0]
    if self.seq_len != np.shape(hidden_state_seq)[0]:
      raise ValueError('Sequence length mismatch.')
    if self.seq_len != np.shape(output_seq)[0]:
      raise ValueError('Sequence length mismatch.')
    self.inputs = input_seq
    self.hidden_states = hidden_state_seq
    self.outputs = output_seq
    self.input_dim = np.shape(input_seq)[1]
    self.output_dim = np.shape(output_seq)[1]


class SequenceGenerator(object):
  """Class for generating sequences according to linear dynamical systems."""

  def __init__(self, output_noise_stddev):
    """Initializes SequenceGenerator.

    Args:
      output_noise_stddev: The stddev of the output noise distribution.
    """
    self.output_noise_stddev = output_noise_stddev

  def _random_normal(self, mean, stddev, dim):
    return mean + stddev * np.random.randn(np.prod(dim)).reshape(dim)

  def generate_seq(self, system, seq_len):
    """Generate seq with random initial state, inputs, and output noise.

    Args:
      system: A LinearDynamicalSystem instance.
      seq_len: The desired length of the sequence.

    Returns:
      A LinearDynamicalSystemSequence object with:
      - outputs: A numpy array of shape [seq_len, output_dim].
      - hidden_states: A numpy array of shape [seq_len, hidden_state_dim].
      - inputs: A numpy array of shape [seq_len, input_dim].
    """
    inputs = self._random_normal(0., 1., [seq_len, system.input_dim])
    outputs = np.zeros([seq_len, system.output_dim])
    output_noises = self._random_normal(0., self.output_noise_stddev,
                                        [seq_len, system.output_dim])
    hidden_states = np.zeros([seq_len, system.hidden_state_dim])
    # Initial state.
    hidden_states[0, :] = self._random_normal(0., 1., system.hidden_state_dim)
    for j in range(1, seq_len):
      hidden_states[j, :] = (
          np.matmul(system.transition_matrix, hidden_states[j - 1, :]) +
          np.matmul(system.input_matrix, inputs[j, :]))
    for j in range(seq_len):
      outputs[j, :] = np.matmul(system.output_matrix,
                                hidden_states[j, :]) + output_noises[j, :]
    return LinearDynamicalSystemSequence(inputs, hidden_states, outputs)


def _generate_stable_symmetric_matrix(hidden_state_dim, eigvalues=None):
  """Generates a symmetric matrix with spectral radius <= 1.

  Args:
    hidden_state_dim: Desired dimension.
    eigvalues: Specified eigenvalues, optional. If None, random eigenvalues will
      be generated from uniform[-1, 1].

  Returns:
    A numpy array of shape [hidden_state_dim, hidden_state_dim] represending a
    symmetric matrix with spectral radius <= 1.
  """
  # Generate eigenvalues.
  if eigvalues is None:
    eigvalues = np.random.uniform(-1.0, 1.0, hidden_state_dim)
  diag_matrix = np.diag(eigvalues)
  if hidden_state_dim == 1:
    change_of_basis = np.ones([1, 1])
  else:
    change_of_basis = ortho_group.rvs(hidden_state_dim)
  # transition_matrix = change_of_basis diag_matrix change_of_basis^T
  transition_matrix = np.matmul(
      np.matmul(change_of_basis, diag_matrix), change_of_basis.transpose())
  # Check that the transition_matrix has to recover the correct eigvalues.
  if np.linalg.norm(
      np.sort(np.linalg.eigvals(transition_matrix)) -
      np.sort(eigvalues)) > 1e-6:
    raise ValueError('Eigenvalues do not match.')
  return transition_matrix


def generate_linear_dynamical_system(hidden_state_dim,
                                     input_dim=1,
                                     output_dim=1,
                                     eigvalues=None,
                                     diagonalizable=True):
  """Generates a diagonalizable LinearDynamicalSystem with given dimensions.

  Args:
    hidden_state_dim: Desired hidden state dim.
    input_dim: The input dim.
    output_dim: Desired output dim.
    eigvalues: Specified eigenvalues, optional. If None, random eigenvalues will
      be generated from uniform[-1, 1] when diagonalizable = True.

  Returns:
    A LinearDynamicalSystem object with
    - A random stable symmetric transition matrx.
    - Identity input matrix.
    - A random output matrix.
  """
  eigvalues = np.random.uniform(size=hidden_state_dim)
  # Generate a symmetric transition matrix A.
  transition_matrix = _generate_stable_symmetric_matrix(
          hidden_state_dim, eigvalues)
  input_matrix = np.random.rand(hidden_state_dim, input_dim)
  output_matrix = np.random.rand(output_dim, hidden_state_dim)
  return LinearDynamicalSystem(transition_matrix, input_matrix, output_matrix)


def fit_lds_pylds(seq, inputs, guessed_dim):
  """Fits LDS model via Gibbs sampling and EM. Returns fitted eigenvalues."""
  if inputs is None:
    model = DefaultLDS(D_obs=1, D_latent=guessed_dim, D_input=0)
  else:
    model = DefaultLDS(D_obs=1, D_latent=guessed_dim, D_input=1)
  model.add_data(seq, inputs=inputs)
  # Initialize with a few iterations of Gibbs.
  for _ in range(10):
    model.resample_model()
  # Run EM
  def update(model):
    model.EM_step()
    return model.log_likelihood()
  ll = [update(model) for _ in range(100)]
  eigs = np.linalg.eigvals(model.A)
  return eigs[np.argsort(eigs.real)[::-1]]
