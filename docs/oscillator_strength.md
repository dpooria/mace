# Oscillator-strength transfer learning

This implementation is independent of X-MACE. It uses MACE-OFF as an
equivariant geometry encoder and a molecule-level readout for TDDFT oscillator
strengths.

## Use the direct TDDFT targets

For the S1 nuclear ensemble, the intended targets are the adiabatic values
`osce_s1`, `osce_s2`, and `osce_s3` from the TDDFT result of each geometry.
Generate them with:

```bash
cd /home/dpooria/Documents/divdi/osce_predict_mace
venv/bin/python db2xyz.py \
  s1/geometries.db \
  s1/s1_adiabatic \
  s1 \
  --state-mode adiabatic \
  --seed 100
```

This creates 800 training and 200 held-out structures. Every written target has
been checked against the direct database columns. Do not train this model on
the older `s1_train.xyz`: that file reassigns states by electronic character
and changes 636 of the 1,000 TDDFT target triplets.

Character-following targets remain available as an explicitly different task:

```bash
venv/bin/python db2xyz.py \
  s1/geometries.db \
  s1/s1_character \
  s1 \
  --state-mode character \
  --seed 100
```

## Model

The model is symmetry safe by construction:

- relative-coordinate MACE messages give translation invariance;
- scalar channels and norms of equivariant channels give rotation and
  reflection invariance;
- mean and standard-deviation pooling over atoms give permutation invariance;
- exponential and logistic output transforms keep strengths positive.

The S1 and S2 outputs are parameterized as their total intensity and their
mixing fraction. This matches the strong S1/S2 intensity exchange in the
downloaded reference data. S3 is predicted in log space.

## Fast training

The frozen `medium_off` backbone is evaluated once. Its descriptors are cached,
standardized across the training set, and used for fast head training. If
`--fine-tune last` is selected, the best cached-head checkpoint then initializes
a shorter end-to-end stage that unfreezes only the final MACE interaction and
product blocks.

```bash
cd /home/dpooria/Documents/divdi/osce_predict_mace/mace
../venv/bin/python -m mace.cli.run_train_oscillator \
  --train-file ../s1/s1_adiabatic_train.xyz \
  --foundation-model medium_off \
  --output ../s1/oscillator-s1.model \
  --head-epochs 500 \
  --epochs 100 \
  --fine-tune last \
  --device cuda
```

Use `--fine-tune head` for the fast frozen-backbone baseline. On the supplied
deterministic split, that baseline reaches held-out R2 values of approximately
0.48 for S1 and 0.46 for S2. The final-layer stage is intended to improve on
that baseline and must be selected using validation data, not the held-out test
set.

The project root contains `train_oscillator.sh` with the recommended Slurm
configuration.

## Evaluate

```bash
../venv/bin/python -m mace.cli.eval_oscillator \
  --model ../s1/oscillator-s1.model \
  --configs ../s1/s1_adiabatic_test.xyz \
  --output ../s1/s1_adiabatic_test_eval.xyz \
  --device cuda
```

Predictions are written to `MACE_osce`. Per-transition MAE, RMSE, R2, and
log-MAE are written beside the output file. For geometries without TDDFT
labels, add `--no-targets`.
