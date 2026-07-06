# Revision v2 integration notes

This version integrates the user's latest scientific framing:

1. Beam dynamics now begins with linearized beam transport and treats collective effects as nonlinear, density-dependent corrections.
2. Data generation is separated into linear tracking datasets, collective-effects datasets, parameter-space coverage, computational cost, and data representation.
3. Neural operators are introduced from the PDE solution-map perspective before the FNO architecture.
4. The VAE/CVAE section now includes mathematical nuance: the latent space is not automatically physical; its physical meaning must be tested through reconstruction, moments, dynamics, and latent--physics correlations.
5. Results now include:
   - FNO longitudinal results,
   - CVAE histogram/marginal/macroparticle reconstruction,
   - Pearson latent--physics correlation study,
   - CVAE + Transformer,
   - CVAE + GNO,
   - Window Transformer vs no-window comparison,
   - window-size ablation,
   - physical-statistics predictor study,
   - runtime comparison.
6. The discussion reframes the exploratory work as an inductive-bias discovery: locality and operator structure determine stable latent beam dynamics.
