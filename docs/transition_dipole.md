# Excited-state model: transition dipoles, energies and spin-orbit couplings

`mace.modules.transition_dipole.TransitionDipoleMACE` is a pretrained MACE
backbone with an equivariant head that predicts, for one geometry,

| quantity | symbol | how |
| --- | --- | --- |
| singlet excitation energies | `E_S` | eigenvalues of a smooth mixing matrix `H_S` |
| transition dipoles `<S0|mu|S_n>` | `mu` | diabatic polar vectors rotated by `eigh(H_S)` |
| oscillator strengths | `f = 2/3 dE[Ha] |mu|^2` | derived, never regressed |
| triplet excitation energies | `E_T` | eigenvalues of a second mixing matrix `H_T` |
| spin-orbit couplings `|<S_m|H_SO|T_k>|`, m = 0..nS | `soc` | diabatic axial vectors rotated by both manifolds |

Labels a database does not have are NaN and drop out of every loss and
metric, so one model definition and one command serve the S0, S1 and T1
ensembles.

## Why not regress `f` directly

The oscillator strength is a derived quantity: `f_n = 2/3 dE_n[Hartree] |mu_n|^2`
holds for every geometry in the DiKTA ensembles to `2e-5`. Predicting `(dE, mu)`
uses the equivariant part of the backbone instead of discarding it, carries
three numbers per state instead of one, and lets the state mixing that makes
`f` discontinuous act linearly on `mu`, where it can be represented exactly.
Adiabatic labels exchange intensity wherever two states approach (14% of the
S1-ensemble geometries have an S1-S2 gap below 0.05 eV), so `f_S1(R)` has
genuine steps that no smooth regressor can fit.

## The model

### Backbone

Any MACE backbone whose product blocks expose `linear.irreps_out` works:

| `--foundation-model` | channels (layer 1) | note |
| --- | --- | --- |
| `medium_off` (also `small_off`, `large_off`) | 128x0e + 128x1o | MACE-OFF23, organic only |
| `mh-1` (default; also `mh-0`) | 512x0e + 512x1o | MACE-MH multi-head (omol, spice heads) |
| `omol` | 1024 channels | MACE-OMOL extra large, 420 MB, ASL licence |
| a `.model` path | | any pickled MACE module |

The head reads the invariant (`0e` and norms of `l > 0`) channels and the
`1o` channels of every layer. The `1o` channels pass through an equivariant
channel mixing (`--vector-dim`, default 32) so the vector readouts stay small
whatever the backbone width.

### Manifolds and the diabatic basis (`--basis diabatic`, default)

Each spin manifold has a smooth symmetric mixing matrix built in standardised
units: diagonal entries are `E_ref + sigma_E * output` per state, off-diagonal
couplings are in units of the mean `sigma_E`, both taken from the training
labels. Diagonalising it gives the energies and the rotation `U` that turns
smooth diabatic state properties into the discontinuous adiabatic ones:

```
E, U = eigh(H)          mu_i = sum_k U[k, i] mu_d[k]
```

`safe_eigh` clamps the `1/(lambda_j - lambda_i)` factor of the eigenvector
gradient (`--eigh-eps`, 1 meV, the only physical constant in the head).
`--basis adiabatic` regresses energies directly and predicts adiabatic
properties without a rotation; it is the control.

### Vectors: magnitude x direction

Polar vectors (dipoles) are transition charges with zero net charge times
centroid-relative positions plus a contraction of the `1o` features. Axial
vectors (spin-orbit couplings: `<S|H_SO|T_a>` is `i` times a real axial vector
in the Cartesian sublevel basis) are cross products of such contractions.
Both are factorised as `v = m(invariants) * v_hat`: `|v|` is a rotational
invariant, so it is read from the invariant features and direction error
never reaches `|mu|^2` or `|SOC|`. Dipole magnitudes are multiples of the
training RMS; SOC magnitudes are log-normal around the training log-mean
because they span decades. `--direct-dipole` disables the factorisation
(ablation: it then fails to fit even the training set).

### Spin-orbit couplings

`--num-triplets N` adds the triplet manifold and the SOC head for the block
`(S_0..S_nS) x (T_1..T_N)`. Two modes:

* `--soc-mode diabatic` (default): diabatic axial vectors `V[m, k]`, rotated
  `U_S0^T V U_T` with `S_0` unrotated; `|SOC| = |V_adiabatic|`. The
  Frobenius norm of the block is a diabatic invariant (unit-tested).
* `--soc-mode direct`: one invariant log-magnitude per pair, no rotation.
  The earlier per-target regression baseline.

Both are supervised in log space, NaN-masked. In `diabatic` mode the vectors
themselves are also supervised (`--soc-vector-weight`, sign-matched like the
dipoles) against `REF_soc_vec`: the Q-Chem sublevel components
`<S|H_SO|T,m_s>` converted to the real Cartesian axial vector with
`V_x = (c_-1 + c_+1)/sqrt2`, `V_y = i(c_-1 - c_+1)/sqrt2`, `V_z = c_0` (Q-Chem's
sublevel convention has no Condon-Shortley sign; `V = iW` with `W` real to
machine precision, and `|W|` equals the root-sum-square magnitude).

Log-fitted magnitudes estimate a geometric mean while rates integrate the
arithmetic one, so `--calibrate mean` (default) rescales every SOC pair and
`osce_t` state to the training-set mean ratio after training, with 3-MAD
outlier rejection; the factors are stored in the checkpoint.

### Auxiliary branch: solvation and phosphorescence

`--solvation` predicts the PCM susceptibilities `d_n` of both manifolds and the
ground-state polarization `gp`. The susceptibilities are rotated as a diabatic
matrix, `d_i = (U^T D U)_ii`, with `D = B B^T` positive semi-definite
(`B = sqrt(mean d) (I + A)`, `A` from the network), so every `d_i >= 0`: a
negative `d` makes NEMO's reorganisation width `sqrt(2 lambda kT + kT^2)` NaN
and NEMO then drops the affected rate channel silently; `--phosphorescence` predicts the corrected
`T_n -> S0` strengths `REF_osce_t` as a log-normal magnitude. Both live on a
separate atom trunk and receive the manifold rotations *detached*, so their
losses cannot move any parameter that energies, dipoles or couplings depend on
(unit-tested: zero gradient on every main parameter). The main outputs are
therefore identical with or without them.

### Fine-tuning

`--fine-tune head|last|full` with `--fine-tune-layers N` (how many interaction
and product blocks from the top train in `last` mode).

### Phase

The sign of `mu_n` is an arbitrary wavefunction phase (48.6% of the DiKTA S1
dipoles are anti-aligned with the ensemble mean). The loss uses
`phase_matched_error`, `sign(mu_pred . mu_ref) * mu_ref`, which is phase free
and quadratic; `mu mu^T` is quartic, puts a zero-gradient saddle at `mu = 0`,
and is kept only as a supporting term (`--tensor-weight`). Metrics are `|mu|^2`
and `|cos|`, never signed components.

## Data

`db2xyz.py` in each `dikta-*` directory writes, per configuration:

```
REF_energy_ex  REF_tdm  REF_osce            singlets (xtraee parse)
REF_energy_t                                triplets (logs, else NEMO e_t*)
REF_soc  REF_soc_shape  REF_soc_vec         |SOC| block (n_S+1, n_T) and Cartesian vectors, eV
REF_osce_t                                  T_k -> S0 strengths, corrected second-order sum
REF_d_s  REF_d_t  REF_gp                    (--solvation) PCM susceptibilities and gp, eV
```

`gather_data.py --update-from-logs DIR` adds `e_t_parsed`, `soc_rss`, `soc_vec`
and `osce_t_corrected` from `Geometry-N-.log` to an existing database. The
magnitudes agree with NEMO's `soc_*` columns to 1e-3 relative. The strengths
do **not** agree with NEMO's `osce_t*` columns: NEMO's `moment()` broadcast a
column of couplings against a row of dipoles, summing
`(sum_k SOC_k)(sum_j mu_j/E_j)` instead of `sum_k SOC_k mu_k/E_k`, and used
`<T1|H|S0>` for every `T_n`. Both are fixed in the local NEMO clone
(`nemo/parser.py`, `nemo/eom.py`); the fixed code and `borrowed_strengths`
agree to four digits, and the corrected `f_T1` is about three orders of
magnitude smaller than the released columns.

## Pipeline

```bash
cd dikta-s1                 # or dikta-s0, dikta-t1: the scripts are identical
./workflow.sh               # db -> extxyz -> train -> eval -> plots
FOUNDATION=medium_off TAG=s1-off ./train_tdm.sh      # another backbone
NUM_TRIPLETS=0 ./train_tdm.sh                        # singlets only
SOC_MODE=direct ./train_tdm.sh
```

`eval_tdm --energy-source model|reference|info` chooses the energies that
enter `f`; `reference` isolates the dipole error.

`nemo_solvents.py <prefix> <out> --state s0|s1|t1 [--solvent NAME:EPS:NR ...]`
(in each `dikta-*` directory) builds NEMO ensemble tables from the evaluated
files, one from the TDDFT labels and one from the model outputs, and runs
`nemo.analysis.rates` / `nemo.analysis.absorption` on both for every solvent:
radiative and ISC rates, peak positions and spectra with NEMO's own solvent
interpolation. It needs the local NEMO clone installed (`pip install -e
osce_predict_mace/NEMO`, plus `LeoX`, `joblib`, `requests`, `tabulate`).

## Checkpoints

Version 2 checkpoints store the constructor options and the training
arguments; models trained with the earlier head (`checkpoint_version` 1,
fixed 0.2 eV coupling scale, MACE-OFF only) are refused by `eval_tdm` and
must be retrained.
