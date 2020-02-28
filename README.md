## Code for paper ["Linear Dynamics: Clustering without identification"](https://arxiv.org/abs/1908.01039)

### Installation

The dependencies are documented in environment.yml and can be installed via conda.
<pre><code>
conda env create -f environment.yml
conda activate lds
</code></pre>
For more details, see [conda documentation for creating an environment from yml file](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).
If there is an error due to pybasicbayes and scipy version incompatibility, try installing the latest version of pybasicbayes from GitHub as the pip version might be outdated.

### Simulated experiments for eigenvalue estimation
To run the eigenvalue estimation experiments in the paper, use the following commands.
<pre><code>
python experiment_learn_eig.py --output_dir=eig_2d/ --min_seq_len=500 --max_seq_len=50000 --num_sampled_seq_len=5 --num_repeat=100 --hidden_dim=2 --output_noise_stddev=0.01
python experiment_learn_eig.py --output_dir=eig_3d/ --min_seq_len=500 --max_seq_len=50000 --num_sampled_seq_len=5 --num_repeat=100 --hidden_dim=3 --output_noise_stddev=0.01
python experiment_learn_eig.py --output_dir=eig_5d/ --min_seq_len=500 --max_seq_len=50000 --num_sampled_seq_len=5 --num_repeat=100 --hidden_dim=5 --output_noise_stddev=0.01
python plot_eig_error.py
</code></pre>

### Simulated experiments for clustering
<pre><code>
python experiments.py --output_dir=cluster_2d_2c --hidden_state_dim=2 --min_seq_len=1000 --max_seq_len=1000 --num_sampled_seq_len=1 --num_systems=100 --num_clusters=2 --num_repeat=100
python experiments.py --output_dir=cluster_2d_3c --hidden_state_dim=2 --min_seq_len=1000 --max_seq_len=1000 --num_sampled_seq_len=1 --num_systems=100 --num_clusters=3 --num_repeat=100
python experiments.py --output_dir=cluster_2d_5c --hidden_state_dim=2 --min_seq_len=1000 --max_seq_len=1000 --num_sampled_seq_len=1 --num_systems=100 --num_clusters=5 --num_repeat=100
python experiments.py --output_dir=cluster_2d_10c --hidden_state_dim=2 --min_seq_len=1000 --max_seq_len=1000 --num_sampled_seq_len=1 --num_systems=100 --num_clusters=10 --num_repeat=100
python agg_stats_to_table.py
</code></pre>
