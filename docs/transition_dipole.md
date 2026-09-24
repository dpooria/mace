# Predicting oscillator strengths through the transition dipole

## Why not regress `f` directly

The oscillator strength is a derived quantity:

```
f_n = 2/3 * dE_n[Hartree] * |mu_n|^2
```

Checked against every geometry in the DiKTA ensemble, this reproduces the
`osce_s*` columns to `2e-5`, so nothing is lost by predicting `(dE, mu)`
instead of `f`. Three things are gained: `mu` is a vector, so the equivariant
part of the backbone gets used instead of discarded; it carries three numbers
per state rather than one; and the state mixing that makes `f` discontinuous
acts on `mu` linearly, so it can be represented exactly.

The discontinuity is the real problem. In this ensemble the S1-S2 gap reaches
0.005 eV, 14% of geometries sit below 0.05 eV, and the bright HOMO->LUMO
configuration is the dominant character of adiabatic S1 in 725/1000 geometries
but of adiabatic S2 in 356/1000. The two adiabatic labels exchange intensity
(`corr(f_S1, f_S2) = -0.886`). `f_S1(R)` therefore has genuine steps, and no
smooth regressor can fit it.

## The model

`mace.modules.transition_dipole.TransitionDipoleMACE` is a MACE backbone with a
`TransitionDipoleHead` on top. MACE rather than X-MACE: MACE-OFF supplies a
pretrained organic backbone whose first interaction layer already exposes 128
`1o` channels, which is exactly how a polar vector transforms.

The dipole is built from two terms, both translation invariant, rotation
equivariant and permutation invariant:

* **transition charges** `q_i` constrained to `sum_i q_i = 0`, contracted with
  the atom positions -- the standard transition-charge ansatz, which carries
  the long-range part;
* a learned contraction of the backbone `1o` node features, the local
  correction a point-charge picture misses.

By default the result is **factorised** into an invariant magnitude and a unit
direction, `mu = m(invariants) * u_hat`. `|mu|` is a rotational invariant, so
reading it off the invariant features directly stops direction error from
leaking into the magnitude -- and `|mu|^2` is what `f` uses. Pass
`--direct-dipole` to take the raw vector instead.

### Bases

`--basis diabatic` (default)
: The head also predicts a smooth symmetric mixing matrix `H`. Diagonalising it
  gives the adiabatic rotation and, as a by-product, the excitation energies:

  ```
  E, U = eigh(H)
  mu_i = sum_k U[k, i] mu_d[k]
  ```

  Everything the network emits is smooth in the nuclear coordinates; the
  eigendecomposition manufactures the non-smooth adiabatic labelling the same
  way the physics does.

`--basis adiabatic`
: One vector per adiabatic state, predicted directly. Simpler, and a useful
  control, but it inherits the discontinuity.

### Phase

The sign of `mu_n` is an arbitrary wavefunction phase: 48.6% of the geometries
in this ensemble have `mu_S1` anti-aligned with the ensemble mean direction.
Nothing -- loss or metric -- may depend on it.

The loss is therefore built on `phase_matched_error`, which compares against
`sign(mu_pred . mu_ref) * mu_ref`. This is phase free *and quadratic*. The
other obvious phase-free choice, `mu mu^T`, is quartic, which makes `mu = 0` a
critical point with vanishing gradient; a head initialised near zero never
leaves it, and trains to a direction accuracy below that of a constant vector.
`mu mu^T` is kept as a supporting term (`--tensor-weight`), never alone.

Metrics follow the same rule: `|mu|^2` and `|cos|`, never signed components.

### Near-degeneracies

`safe_eigh` clamps the `1 / (lambda_j - lambda_i)` factor in the eigenvector
gradient. Without it the gradient diverges exactly where two states approach
each other, which is the regime the diabatic basis exists to handle. There is a
unit test asserting the clamped gradient is finite on a degenerate matrix where
`torch.linalg.eigh` returns NaN, and that it matches autograd away from
degeneracy.

## Pipeline

```bash
./workflow.sh
```

runs the four steps individually available as:

```bash
# 1. QChem TDDFT logs -> ASE database, via the xtraee parser
venv/bin/python gather_data.py Dikta --db=s1/geometries.db --state=s1

# 2. database -> extxyz with REF_tdm / REF_energy_ex / REF_osce
venv/bin/python db2xyz.py s1/geometries.db s1/dikta s1 --num-states 3

# 3. train
./train_tdm.sh

# 4. evaluate
venv/bin/python -m mace.cli.eval_tdm \
  --model s1/tdm-s1.model \
  --configs s1/dikta_test.xyz \
  --output s1/dikta_test_eval.xyz \
  --energy-source model
```

`--energy-source` selects where the energies in `f` come from: `model` (the
mixing-matrix eigenvalues), `info --energy-key XMACE_energy_ex` (energies
written into the configurations by another model), or `reference`, which uses
the TDDFT values and so isolates the error contributed by the dipole alone.

## Results

Held-out test split (200 geometries), model selected on the validation split
(100), trained on 700. Diabatic basis, factorised dipole, MACE-OFF medium
backbone:

| | S1 | S2 | S3 |
| --- | --- | --- | --- |
| oscillator strength `f` | **0.856** | **0.869** | 0.189 |
| `\|mu\|^2` | 0.852 | 0.872 | 0.195 |
| excitation energy `E` | 0.940 | 0.933 | 0.949 |

S3 is weak because it carries almost no intensity (mean `f_S3 = 0.0085`, two
orders of magnitude below S1), so there is little signal to fit.

### Why this beats the smooth-regression ceiling

Kernel ridge on inverse interatomic distances -- a strong baseline for
*smoothly regressing the target as a function of geometry* -- tops out at:

| target | KRR test R2 |
| --- | --- |
| `f_S1` | 0.318 |
| `f_S1 + f_S2` | 0.489 |
| `f_S1 + f_S2 + f_S3` | 0.858 |
| `\|mu_S1\|^2` | 0.311 |
| `E_S1` | 0.638 |

The diabatic model is not doing smooth regression of `f`. It regresses a smooth
mixing matrix and smooth diabatic dipoles, and the eigendecomposition generates
the discontinuity, so it can represent a target the smooth ceiling forbids. The
energies show the same pattern: 0.638 regressed directly, 0.94 as eigenvalues
of the learned matrix.

### Ablations

Same split, same backbone, `hidden_dim=64`, `weight_decay=1e-3`:

| basis | factorised | **train** `\|mu_S1\|^2` | test `f_S1` | test `f_S2` | test `f_S3` | test `E_S1` |
| --- | --- | --- | --- | --- | --- | --- |
| diabatic | yes | 0.97 | **0.856** | **0.869** | 0.189 | **0.940** |
| diabatic | no | 0.23 | -1.776 | 0.103 | -0.022 | 0.875 |
| adiabatic | yes | 0.96 | 0.656 | 0.686 | **0.427** | n/a |

**Factorisation is the dominant effect.** Without it the head cannot even fit
its own training set (`|mu_S1|^2` train R2 = 0.23; an unregularised 512-wide
version reaches only 0.46 after 3000 epochs). The bottleneck is arithmetic, not
capacity: the intensity has to squeeze through three pooled vector components,
and component errors compound when squared. Reading the magnitude off the
invariant features instead lifts `f_S1` from -1.78 to 0.856. Direction accuracy
stays modest either way (`|cos| ~ 0.73`), which is exactly the point -- once
the magnitude is factored out, direction error no longer reaches `f`.

**The diabatic basis is a real but smaller second effect**, worth about +0.20
on `f_S1` and `f_S2` on top of factorisation, and it is the only route to
excitation energies (0.94 against nothing in the adiabatic head, where the
energies are a fixed reference). `f_S3` moves the other way -- 0.427 adiabatic
against 0.189 diabatic -- because S3 barely mixes, so the three-state rotation
mostly injects noise into it. If S3 matters for an application, train it in the
adiabatic basis separately.

Note that the adiabatic head fits the training set perfectly well (0.96). The
discontinuity of the adiabatic labelling is what limits *generalisation* here,
not trainability.

### Selecting between configurations

Validation losses are **not comparable across bases**: the diabatic loss
carries an energy term the adiabatic loss does not, so the adiabatic runs post
a lower number (0.504) while being clearly worse on `f` (0.642 against 0.856).
Compare within a basis, or on a shared metric such as validation `f` R2.

### What did not work

Supervising the summed intensity directly (`--total-weight 0.3`) was tried
because `sum(f)` comes out at 0.804, below the 0.858 that kernel ridge reaches
on that quantity. It made everything worse, including the sum:

| | per-state `f` | `sum(f)` | `E` |
| --- | --- | --- | --- |
| default | 0.873 / 0.874 / 0.605 | **0.804** | 0.945 / 0.935 / 0.955 |
| `--total-weight 0.3` | 0.816 / 0.824 / 0.441 | 0.659 | 0.937 / 0.932 / 0.939 |

`sum(f)` is a function of predictions the per-state terms already constrain, so
the term carries no new information and only tilts the objective towards a
coarser target. The flag is kept, defaulting to 0, so the experiment can be
repeated.

## Caveats

`f_S3` is near the noise floor of this dataset: test R2 came out at 0.189 in
one configuration and 0.605 in another differing only in optimiser schedule.
Mean `f_S3` is 0.0085, two orders of magnitude below S1. Do not quote a single
number for it without repeated seeds. And the split between S1 and S2
remains the sensitive quantity: it is set by the near-degenerate energy
ordering, which is why energies and dipoles are predicted from one shared
mixing matrix rather than by two independent models.

For a nuclear-ensemble spectrum the per-state assignment cancels anyway -- the
spectrum needs the `(E, f)` pairs summed over states.
