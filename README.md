# SIA Portfolio: Search, Learning, Optimization, and Generation

This repository is a full AI/ML journey implemented across six projects, from classical search and evolutionary computation to supervised learning, unsupervised learning, and deep generative models.

The structure below intentionally uses topic names (instead of generic `tp0`, `tp1`, ..., `tpN`) so visitors can immediately see the real scope of the work.

## Topic-First Map

| Topic Name | Folder | What is inside |
| --- | --- | --- |
| Pokemon Capture Probability and Data Analysis | `tp0` | Data analysis workflow for modeling Pokemon capture function behavior |
| State-Space Search: 8-Puzzle + Sokoban | `tp1` | BFS, DFS, A*, Local Greedy, Global Greedy, heuristics, benchmarking |
| Genetic Algorithms for RPG Build Optimization | `tp2` | Evolutionary operators, adaptive mutation, selection/replacement strategies |
| Supervised Learning: Perceptrons and MLPs | `tp3` | Step/linear/non-linear perceptrons, MLPs, optimizer studies, noise robustness |
| Unsupervised Learning and Associative Memory | `tp4` | Kohonen SOM, PCA, Oja, Sanger, Hopfield |
| Autoencoders and Variational Latent Spaces | `tp5` | AE, DAE, VAE, latent-space exploration, custom image experiments |

## Why This Repo Is Interesting

- It covers both symbolic AI and statistical learning.
- It includes from-scratch implementations, not just high-level library wrappers.
- It focuses on experimentation: configuration-driven runs, metrics, plots, and visual analysis.
- It demonstrates breadth and depth across core AI foundations and modern ML methods.

## Concept Depth Covered

- Search and planning: backtracking, repeated-state handling, informed vs uninformed search.
- Evolutionary computation: crossover/mutation design, dynamic mutation schedules, mixed selection policies.
- Supervised learning: perceptrons, MLP topology tuning, optimizer effects, noisy data robustness.
- Unsupervised learning: topology-preserving maps, dimensionality reduction, Hebbian rules.
- Associative memory: Hopfield pattern storage and recovery dynamics.
- Generative modeling: AE/DAE/VAE and latent-space geometry.

## Project Narratives

### 1) Pokemon Capture Probability and Data Analysis (`tp0`)

This project introduces the portfolio with a data analysis workflow focused on Pokemon capture probability modeling. It establishes reproducible experimentation through config-based execution and report-driven interpretation.

- Folder README: [`tp0/README.md`](tp0/README.md)
- Docs: [`tp0/docs/SIA_TP0.pdf`](tp0/docs/SIA_TP0.pdf), [`tp0/docs/Informe-TP0.pdf`](tp0/docs/Informe-TP0.pdf)
- Run:

```bash
cd tp0
pipenv install
pipenv run python main.py ./configs
```

### 2) State-Space Search: 8-Puzzle + Sokoban (`tp1`)

This project explores classical AI search deeply, combining algorithmic implementations with visual and metric-based comparisons.

- Implemented search methods include BFS, DFS, A*, Local Greedy, and Global Greedy.
- Includes heuristic experimentation and analysis pipelines for runtime, expanded nodes, and movement cost.
- Provides both interactive Sokoban visualization and reproducible batch experiments.

Folder docs:
- [`tp1/README.md`](tp1/README.md)
- Core algorithms: [`tp1/core/algorithms`](tp1/core/algorithms)
- Heuristics: [`tp1/core/heuristics.py`](tp1/core/heuristics.py)

Representative visuals:

![tp1-bfs-1](tp1/resources/gifs/bfs_1.gif)
![tp1-all-algorithms-time](tp1/output/graphs/t_all_algorithms.png)
![tp1-admissible-vs-inadmissible](tp1/output/graphs/h_adm_vs_inadm.png)

### 3) Genetic Algorithms for RPG Build Optimization (`tp2`)

This project builds an optimization engine to find high-performing RPG character configurations under game-time constraints.

- Rich operator design: multiple crossover types, mutation modes, and mutation distributions.
- Dynamic mutation rates (constant, sinusoidal, exponential decay).
- Weighted composition of parent selection and replacement strategies.
- Extensive config system for reproducibility and hyperparameter analysis.

Folder docs:
- [`tp2/README.md`](tp2/README.md)
- Main run entry: [`tp2/src/master.py`](tp2/src/master.py)

Representative visuals:

![tp2-population-evolution](tp2/output/video/population_evolution_comp.gif)
![tp2-exponential-decay](tp2/output/video/exponential_decay.gif)
![tp2-heatmap-best](tp2/output/data/heatmap_best.png)

### 4) Supervised Learning: Perceptrons and MLPs (`tp3`)

This project progresses from simple perceptrons to deeper MLP architectures, including practical tooling for training analysis and robustness testing.

- Step, linear, and non-linear perceptrons.
- MLP training for XOR, parity, and MNIST-like digit tasks.
- Optimizer comparisons (gradient descent, momentum, Adam).
- Noise-injection experiments and confusion-matrix evolution.

Folder docs:
- [`tp3/README.md`](tp3/README.md)
- Model implementations: [`tp3/src/models`](tp3/src/models)

Representative visuals:

![tp3-interface](tp3/res/assets/adam_95.gif)
![tp3-perceptron-3d](tp3/src/output/ej1/perceptron_training_3d_boundary.gif)
![tp3-confusion-evolution](tp3/src/output/ej4/confusion_matrix_evolution_adam.gif)

### 5) Unsupervised Learning and Associative Memory (`tp4`)

This project focuses on structure discovery and memory dynamics via classic unsupervised and recurrent neural approaches.

- Kohonen SOM for topology-preserving projection and clustering.
- PCA implementation via covariance analysis.
- Oja and Sanger rules for principal component extraction.
- Hopfield network for pattern storage and denoising-style recovery.

Folder docs:
- [`tp4/README.md`](tp4/README.md)
- Core implementations: [`tp4/src/core`](tp4/src/core)

Representative visuals:

![tp4-kohonen-map](tp4/assets/kohonen_map_evolution.gif)
![tp4-bmu-count](tp4/assets/bmu_count.gif)
![tp4-hopfield-energy](tp4/out/hopfield_recovery_energy.gif)

### 6) Autoencoders and Variational Latent Spaces (`tp5`)

This project extends the neural stack into representation learning and generation with AE, DAE, and VAE.

- AE/DAE abstractions built over MLP foundations.
- VAE implementation with encoder/decoder pipelines and latent sampling.
- Latent-space trajectory visualizations for interpretability.
- Custom image workflow for face dataset experimentation.

Folder docs:
- [`tp5/README.md`](tp5/README.md)
- Core implementations: [`tp5/src/core`](tp5/src/core)

Representative visuals:

![tp5-vae](tp5/res/out_vae.png)
![tp5-latent-smooth](tp5/src/latent_space_smooth_large_sine.gif)
![tp5-latent-labels](tp5/src/latent_space_with_labels.gif)

## Full Project READMEs

- [`tp0/README.md`](tp0/README.md)
- [`tp1/README.md`](tp1/README.md)
- [`tp2/README.md`](tp2/README.md)
- [`tp3/README.md`](tp3/README.md)
- [`tp4/README.md`](tp4/README.md)
- [`tp5/README.md`](tp5/README.md)

## Complete Visual Atlas (All Images and GIFs)

All visual assets found in the repository are embedded below so the full experimentation trail is visible in one place.

<details>
<summary><strong>Pokemon Capture Probability and Data Analysis</strong> (<code>tp0</code>, 0 assets)</summary>

No images were found in this folder.
</details>


<details>
<summary><strong>State-Space Search: 8-Puzzle + Sokoban</strong> (<code>tp1</code>, 43 assets)</summary>

![tp1-output-graphs-a_star_avg_time.png](<tp1/output/graphs/a_star_avg_time.png>)

![tp1-output-graphs-a_star_expanded_nodes.png](<tp1/output/graphs/a_star_expanded_nodes.png>)

![tp1-output-graphs-a_star_total_movements.png](<tp1/output/graphs/a_star_total_movements.png>)

![tp1-output-graphs-en_all_algorithms.png](<tp1/output/graphs/en_all_algorithms.png>)

![tp1-output-graphs-en_bfs_vs_a_star.png](<tp1/output/graphs/en_bfs_vs_a_star.png>)

![tp1-output-graphs-en_bfs_vs_a_star_rigged.png](<tp1/output/graphs/en_bfs_vs_a_star_rigged.png>)

![tp1-output-graphs-en_bfs_vs_dfs.png](<tp1/output/graphs/en_bfs_vs_dfs.png>)

![tp1-output-graphs-en_bfs_vs_dfs_rigged.png](<tp1/output/graphs/en_bfs_vs_dfs_rigged.png>)

![tp1-output-graphs-en_dfs_vs_local.png](<tp1/output/graphs/en_dfs_vs_local.png>)

![tp1-output-graphs-en_dfs_vs_local_rigged.png](<tp1/output/graphs/en_dfs_vs_local_rigged.png>)

![tp1-output-graphs-global_greedy_avg_time.png](<tp1/output/graphs/global_greedy_avg_time.png>)

![tp1-output-graphs-global_greedy_expanded_nodes.png](<tp1/output/graphs/global_greedy_expanded_nodes.png>)

![tp1-output-graphs-global_greedy_total_movements.png](<tp1/output/graphs/global_greedy_total_movements.png>)

![tp1-output-graphs-h_adm_vs_inadm.png](<tp1/output/graphs/h_adm_vs_inadm.png>)

![tp1-output-graphs-h_perm_1.png](<tp1/output/graphs/h_perm_1.png>)

![tp1-output-graphs-h_perm_2.png](<tp1/output/graphs/h_perm_2.png>)

![tp1-output-graphs-t_all_algorithms.png](<tp1/output/graphs/t_all_algorithms.png>)

![tp1-output-graphs-t_bfs_vs_a_star.png](<tp1/output/graphs/t_bfs_vs_a_star.png>)

![tp1-output-graphs-t_bfs_vs_a_star_rigged.png](<tp1/output/graphs/t_bfs_vs_a_star_rigged.png>)

![tp1-output-graphs-t_bfs_vs_dfs.png](<tp1/output/graphs/t_bfs_vs_dfs.png>)

![tp1-output-graphs-t_bfs_vs_dfs_rigged.png](<tp1/output/graphs/t_bfs_vs_dfs_rigged.png>)

![tp1-output-graphs-t_dfs_vs_local.png](<tp1/output/graphs/t_dfs_vs_local.png>)

![tp1-output-graphs-t_dfs_vs_local_rigged.png](<tp1/output/graphs/t_dfs_vs_local_rigged.png>)

![tp1-resources-gifs-a_star_1.gif](<tp1/resources/gifs/a_star_1.gif>)

![tp1-resources-gifs-a_star_2.gif](<tp1/resources/gifs/a_star_2.gif>)

![tp1-resources-gifs-a_star_4.gif](<tp1/resources/gifs/a_star_4.gif>)

![tp1-resources-gifs-bfs_1.gif](<tp1/resources/gifs/bfs_1.gif>)

![tp1-resources-gifs-dfs_3.gif](<tp1/resources/gifs/dfs_3.gif>)

![tp1-resources-gifs-global_3.gif](<tp1/resources/gifs/global_3.gif>)

![tp1-resources-gifs-local_3_dumb.gif](<tp1/resources/gifs/local_3_dumb.gif>)

![tp1-resources-gifs-local_3_smart.gif](<tp1/resources/gifs/local_3_smart.gif>)

![tp1-resources-texture_packs-default-box.png](<tp1/resources/texture_packs/default/box.png>)

![tp1-resources-texture_packs-default-box_on_goal.png](<tp1/resources/texture_packs/default/box_on_goal.png>)

![tp1-resources-texture_packs-default-empty.png](<tp1/resources/texture_packs/default/empty.png>)

![tp1-resources-texture_packs-default-goal.png](<tp1/resources/texture_packs/default/goal.png>)

![tp1-resources-texture_packs-default-player.gif](<tp1/resources/texture_packs/default/player.gif>)

![tp1-resources-texture_packs-default-wall.png](<tp1/resources/texture_packs/default/wall.png>)

![tp1-resources-texture_packs-minecraft-box.jpeg](<tp1/resources/texture_packs/minecraft/box.jpeg>)

![tp1-resources-texture_packs-minecraft-box_on_goal.png](<tp1/resources/texture_packs/minecraft/box_on_goal.png>)

![tp1-resources-texture_packs-minecraft-empty.jpg](<tp1/resources/texture_packs/minecraft/empty.jpg>)

![tp1-resources-texture_packs-minecraft-goal.png](<tp1/resources/texture_packs/minecraft/goal.png>)

![tp1-resources-texture_packs-minecraft-player.jpg](<tp1/resources/texture_packs/minecraft/player.jpg>)

![tp1-resources-texture_packs-minecraft-wall.jpg](<tp1/resources/texture_packs/minecraft/wall.jpg>)

</details>


<details>
<summary><strong>Genetic Algorithms for RPG Build Optimization</strong> (<code>tp2</code>, 19 assets)</summary>

![tp2-output-data-heatmap_avg.png](<tp2/output/data/heatmap_avg.png>)

![tp2-output-data-heatmap_avg_2.png](<tp2/output/data/heatmap_avg_2.png>)

![tp2-output-data-heatmap_best.png](<tp2/output/data/heatmap_best.png>)

![tp2-output-data-heatmap_best_2.png](<tp2/output/data/heatmap_best_2.png>)

![tp2-output-data-mutation_rate_comparison_avg.png](<tp2/output/data/mutation_rate_comparison_avg.png>)

![tp2-output-data-selection_rate_comparison_avg.png](<tp2/output/data/selection_rate_comparison_avg.png>)

![tp2-output-video-3D_Surface_Plots.gif](<tp2/output/video/3D_Surface_Plots.gif>)

![tp2-output-video-exponential_decay.gif](<tp2/output/video/exponential_decay.gif>)

![tp2-output-video-genotype_changes_with_fitness.gif](<tp2/output/video/genotype_changes_with_fitness.gif>)

![tp2-output-video-good_mutation_comp_2d_contour.gif](<tp2/output/video/good_mutation_comp_2d_contour.gif>)

![tp2-output-video-local_vs_global_height2.gif](<tp2/output/video/local_vs_global_height2.gif>)

![tp2-output-video-offspring_domain_comparison.gif](<tp2/output/video/offspring_domain_comparison.gif>)

![tp2-output-video-offspring_domain_crossover.gif](<tp2/output/video/offspring_domain_crossover.gif>)

![tp2-output-video-png_sumidero.png](<tp2/output/video/png_sumidero.png>)

![tp2-output-video-population_evolution2.gif](<tp2/output/video/population_evolution2.gif>)

![tp2-output-video-population_evolution_comp.gif](<tp2/output/video/population_evolution_comp.gif>)

![tp2-output-video-too_much_mutation_2d_contour.gif](<tp2/output/video/too_much_mutation_2d_contour.gif>)

![tp2-output-video-xxx.gif](<tp2/output/video/xxx.gif>)

![tp2-output-zero_mutation.gif](<tp2/output/zero_mutation.gif>)

</details>


<details>
<summary><strong>Supervised Learning: Perceptrons and MLPs</strong> (<code>tp3</code>, 38 assets)</summary>

![tp3-res-assets-adam_95.gif](<tp3/res/assets/adam_95.gif>)

![tp3-src-output-ej1-perceptron_training_2d_projected.gif](<tp3/src/output/ej1/perceptron_training_2d_projected.gif>)

![tp3-src-output-ej1-perceptron_training_3d_boundary.gif](<tp3/src/output/ej1/perceptron_training_3d_boundary.gif>)

![tp3-src-output-ej1-perceptron_training_fast.gif](<tp3/src/output/ej1/perceptron_training_fast.gif>)

![tp3-src-output-ej1-perceptron_training_slow.gif](<tp3/src/output/ej1/perceptron_training_slow.gif>)

![tp3-src-output-ej1-perceptron_training_xor.gif](<tp3/src/output/ej1/perceptron_training_xor.gif>)

![tp3-src-output-ej2-an_l_vs_n.png](<tp3/src/output/ej2/an_l_vs_n.png>)

![tp3-src-output-ej2-an_l_vs_n_prediction.png](<tp3/src/output/ej2/an_l_vs_n_prediction.png>)

![tp3-src-output-ej2-an_l_vs_n_rate.png](<tp3/src/output/ej2/an_l_vs_n_rate.png>)

![tp3-src-output-ej2-an_why1.png](<tp3/src/output/ej2/an_why1.png>)

![tp3-src-output-ej2-an_why2.png](<tp3/src/output/ej2/an_why2.png>)

![tp3-src-output-ej2-ffa.png](<tp3/src/output/ej2/ffa.png>)

![tp3-src-output-ej2-ffa_sigm_reul.png](<tp3/src/output/ej2/ffa_sigm_reul.png>)

![tp3-src-output-ej2-ffa_tanh.png](<tp3/src/output/ej2/ffa_tanh.png>)

![tp3-src-output-ej2-loss_vs_epoch.png](<tp3/src/output/ej2/loss_vs_epoch.png>)

![tp3-src-output-ej2-losses.png](<tp3/src/output/ej2/losses.png>)

![tp3-src-output-ej2-meta_metrics.png](<tp3/src/output/ej2/meta_metrics.png>)

![tp3-src-output-ej2-metrics.png](<tp3/src/output/ej2/metrics.png>)

![tp3-src-output-ej2-metrics_pseudo.png](<tp3/src/output/ej2/metrics_pseudo.png>)

![tp3-src-output-ej2-n_vs_l.png](<tp3/src/output/ej2/n_vs_l.png>)

![tp3-src-output-ej3-cake_for_original_digits.png](<tp3/src/output/ej3/cake_for_original_digits.png>)

![tp3-src-output-ej3-digit_accuracy_vs_epochs_clean_clean.png](<tp3/src/output/ej3/digit_accuracy_vs_epochs_clean_clean.png>)

![tp3-src-output-ej3-digit_accuracy_vs_epochs_clean_noisy1_mean_0_stddev_0.4.png](<tp3/src/output/ej3/digit_accuracy_vs_epochs_clean_noisy1_mean_0_stddev_0.4.png>)

![tp3-src-output-ej3-digit_accuracy_vs_epochs_clean_noisy1_mean_0_stddev_0.75.png](<tp3/src/output/ej3/digit_accuracy_vs_epochs_clean_noisy1_mean_0_stddev_0.75.png>)

![tp3-src-output-ej3-noisy1_noisy2_cross_val.png](<tp3/src/output/ej3/noisy1_noisy2_cross_val.png>)

![tp3-src-output-ej3-noisy1_noisy2_precision.png](<tp3/src/output/ej3/noisy1_noisy2_precision.png>)

![tp3-src-output-ej3-parity_accuracy_vs_epochs.png](<tp3/src/output/ej3/parity_accuracy_vs_epochs.png>)

![tp3-src-output-ej3-topology_1hl_300pc.png](<tp3/src/output/ej3/topology_1hl_300pc.png>)

![tp3-src-output-ej3-topology_1hl_7pc.png](<tp3/src/output/ej3/topology_1hl_7pc.png>)

![tp3-src-output-ej3-topology_2hl_100pc.png](<tp3/src/output/ej3/topology_2hl_100pc.png>)

![tp3-src-output-ej3-topology_2hl_300pc.png](<tp3/src/output/ej3/topology_2hl_300pc.png>)

![tp3-src-output-ej3-topology_3hl.png](<tp3/src/output/ej3/topology_3hl.png>)

![tp3-src-output-ej3-topology_5hl.png](<tp3/src/output/ej3/topology_5hl.png>)

![tp3-src-output-ej3-topology_5hl_long.png](<tp3/src/output/ej3/topology_5hl_long.png>)

![tp3-src-output-ej3-topology_gap.png](<tp3/src/output/ej3/topology_gap.png>)

![tp3-src-output-ej3-training_with_salt_and_pepper_cross_val.png](<tp3/src/output/ej3/training_with_salt_and_pepper_cross_val.png>)

![tp3-src-output-ej4-confusion_matrix_evolution.gif](<tp3/src/output/ej4/confusion_matrix_evolution.gif>)

![tp3-src-output-ej4-confusion_matrix_evolution_adam.gif](<tp3/src/output/ej4/confusion_matrix_evolution_adam.gif>)

</details>


<details>
<summary><strong>Unsupervised Learning and Associative Memory</strong> (<code>tp4</code>, 5 assets)</summary>

![tp4-assets-bmu_count.gif](<tp4/assets/bmu_count.gif>)

![tp4-assets-kohonen_map_evolution.gif](<tp4/assets/kohonen_map_evolution.gif>)

![tp4-out-hopfield_recovery_energy.gif](<tp4/out/hopfield_recovery_energy.gif>)

![tp4-src-convergence.gif](<tp4/src/convergence.gif>)

![tp4-src-non_convergence.gif](<tp4/src/non_convergence.gif>)

</details>


<details>
<summary><strong>Autoencoders and Variational Latent Spaces</strong> (<code>tp5</code>, 9 assets)</summary>

![tp5-data-pibardos-image-copy-2.png](<tp5/data/pibardos/image copy 2.png>)

![tp5-data-pibardos-image-copy-3.png](<tp5/data/pibardos/image copy 3.png>)

![tp5-data-pibardos-image-copy-4.png](<tp5/data/pibardos/image copy 4.png>)

![tp5-data-pibardos-image-copy.png](<tp5/data/pibardos/image copy.png>)

![tp5-data-pibardos-image.png](<tp5/data/pibardos/image.png>)

![tp5-res-out_vae.png](<tp5/res/out_vae.png>)

![tp5-src-latent_space_slow_smooth_path.gif](<tp5/src/latent_space_slow_smooth_path.gif>)

![tp5-src-latent_space_smooth_large_sine.gif](<tp5/src/latent_space_smooth_large_sine.gif>)

![tp5-src-latent_space_with_labels.gif](<tp5/src/latent_space_with_labels.gif>)

</details>

## Running the Projects

Most folders use independent configs and environments. A practical flow is:

```bash
# example pattern
cd tpX
pipenv install
# run script defined in that folder README
```

## Final Note

This repository is designed as an AI/ML learning and engineering portfolio: each project focuses on a different problem class, and together they show end-to-end understanding of core concepts across search, optimization, supervised learning, unsupervised learning, and generative modeling.
