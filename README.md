# OmniLearn for Heavy-Ion Collisions

Score-based diffusion model for generating heavy-ion collision events using the Point-Edge Transformer (PET) architecture within the OmniLearn framework. 

For the general-purpose OmniLearn framework, see the original repository: [github.com/ViniciusMikuni/OmniLearn](https://github.com/ViniciusMikuni/OmniLearn)


## Overview

This repository adapts OmniLearn structure from [Point cloud-based diffusion models for the Electron-Ion Collider](https://arxiv.org/abs/2410.22421v1) for heavy-ion event generation with a two-stage training strategy:
- **Stage 1:** Train on O-O collisions (~10³ particles/event)
- **Stage 2:** Fine-tune on Pb-Pb collisions (~10⁴ particles/event)

The model generates both event-level properties and per-particle kinematics in a single pipeline. 


## Repository structure

```
scripts/
├── PET_lhcHI.py           # Model definition (PET + ResNet)
├── train_lhcHI.py         # Training script 
├── evaluate_lhcHI.py      # Used for event generation
└── utils.py               # Data loader, preprocessing, helper functions 
```

## Data format

Training data is stored in HDF5 (`.h5`) with the following structure:

```
file.h5
├── particle    : float32, shape (N_events, N_max, 6)     # particle features
└── event       : float32, shape (N_events, 11)           # event-level features
```

**Event-level features** (11 components):

    ⟨pT⟩, ΣpT, Qx, Qy, ψ_EP, b, N_coll, N_part, ⟨η⟩, ⟨φ⟩, N_particles

where Qx,Qy are the second-harmonic flow vector components, ψ_EP the event-plane angle, 
b the impact parameter, and N_coll/N_part the Glauber-model collision and participant counts.
`N_particles` is stored as the last entry in the event array internally but handled separately by the model via `revert_npart`.


**Particle features** — (6 components), zero-padded to N_max:

    φ_rel, pT_rel, η_rel, pT, η, φ

where the relative coordinates are defined with respect to the event mean, 
and pT_rel = log(pT / ⟨pT⟩).


## Preprocessing

All features are normalised using mean and standard deviations computed from the training set. These constants are stored in `utils.py`. 

```python
_jet_mean  = [...]   # event-level, 11 values
_jet_std   = [...]
_part_mean = [...]   # particle-level, 6 values 
_part_std  = [...]
```

**These constants must be recomputed when using a different dataset.** The same constants are applied at generation time to revert model output to physical units.

## Training

### Prerequisites

Training uses [Horovod](https://github.com/horovod/horovod) for distributed data-parallel execution across multiple GPUs.

### Stage 1: O-O training

```bash
python scripts/train_lhcHI.py \
    --local --layer_scale \
    --folder /path/to/OO_data/ \
    --batch B \
    --epoch E \
    ...
```

The data folder should contain:
```
OO_data/
│   ├── train_eic.h5
│   └── test_eic.h5
├── checkpoints/
└── histories/
```

### Stage 2: Pb-Pb fine-tuning

Fine-tune from the Stage 1 checkpoint. Due to GPU memory constraints with ~10⁴ particles/event, batch size must be reduced to 1.

```bash
python scripts/train_lhcHI.py \
    --local --layer_scale \
    --folder /path/to/PbPb_data/ \
    --resume_from /path/to/OO_data/checkpoints/ \
    --batch 1 \
    --epoch E
    ...
```
**Important:** Update the mean/std constants in `utils.py` to match the Pb-Pb dataset before fine-tuning.


## Evaluation

Only purpose of data generation in this pipeline: 

```bash
python scripts/evaluate_lhcHI.py \
    --weights /path/to/weights.h5 \
    --data /path/to/test_eic.h5 \
    --output /path/to/output_plots/
    ... 
```


## Data and models

Links to the training datasets and pretrained model weights will be posted here. 
