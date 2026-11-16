"""
Preprocess ALICE O2 heavy-ion generator trees (O2omnigenevent / O2omnigenpart)
into the HDF5 format read by utils.lhcHIDataLoader (OmniLearn heavy-ion fork).

Output: <out_dir>/lhcHI/{train,test,val}_lhcHI.h5, each with
  particle : float32 (N_events, N_max, F), zero-padded, particles sorted by pT (descending)
             F = 6 kinematic features  [phi_rel, pT_rel, eta_rel, pT, eta, phi]
             (+ with --pid: charge, then one-hot species, see SPECIES below)
  jet      : float32 (N_events, 11) event features, in the order the loader expects:
             [mean_pT, sum_pT, Qx, Qy, mean_eta, ep_angle, b, n_coll, n_part, mean_phi, n_particles]
  pdg      : int32 (N_events, N_max) raw PDG codes (0 = padding). Not read by the loader;
             kept for evaluation (e.g. per-species plots).
plus <out_dir>/lhcHI/normalisation.json with the constants to copy into utils.py.

File roles in the existing scripts:
  train_lhcHI.h5 -> training (train_lhcHI.py)
  test_lhcHI.h5  -> validation during training (train_lhcHI.py)
  val_lhcHI.h5   -> generation / evaluation (evaluate_lhcHI.py)

Feature definitions (matching the existing heavy-ion setup):
  phi_rel = phi - <phi>, wrapped to [-pi, pi)
  pT_rel  = log(pT / <pT>)
  eta_rel = eta - <eta>
  Qx, Qy  = sum_i pT_i * cos(2 phi_i), sum_i pT_i * sin(2 phi_i)   (as in PET_lhcHI.psi2_loss_from_x0)
  ep_angle in [-pi/2, pi/2): true reaction plane (default) or Q-vector event plane (--ep_source q_vector)
  n_part  = fNPartProj + fNPartTarg
  n_particles = number of particles stored for the event (after cuts and --max_part)
All event features are computed from the particles that are actually stored.

Usage:
  python preprocess_lhcHI.py --files "data/run_oo_batch_*/AnalysisResults_trees.root" \
      --out_dir /path/to/OO_data [--pid] [--max_part 5000] [--no_cuts]
"""

import os
import glob
import json
import argparse
import numpy as np
import pandas as pd


PT_MIN_CUT = 1.0
ETA_ABS_CUT = 0.9

EVENT_TREE_NAME = "O2omnigenevent"
PART_TREE_NAME = "O2omnigenpart"
EVENT_COLS = ["fImpactParameter", "fReactionPlane", "fNColl", "fNPartProj", "fNPartTarg", "fPosZ", "fNParticles"]
PART_COLS = ["fIndexOmniGenEvents", "fPt", "fEta", "fPhi", "fPdgCode", "fCharge"]

KIN_NAMES = ["phi_rel", "pT_rel", "eta_rel", "pT", "eta", "phi"]
JET_NAMES = ["e_pT", "sum_pT", "Qx", "Qy", "eta_jet", "ep_angle", "b", "n_coll", "n_part", "e_phi", "n_particles"]

# One-hot species, matched on |PDG code| (particle and antiparticle share a flag;
# the charge column distinguishes them). Everything not listed goes to "other",
# so every real particle has exactly one flag set.
SPECIES = {
    "pion": [211],
    "kaon": [321],
    "proton": [2212],
    "electron": [11],
    "muon": [13],
}
SPECIES_NAMES = list(SPECIES) + ["other"]

# Column 2 (eta_rel) is used by the loader to build the mask (X[:, :, 2] != 0).
# A real particle with eta_rel exactly 0 (always the case for single-particle events)
# would be treated as padding, so exact zeros are nudged by this amount.
MASK_EPS = 1e-6


def wrap_angle(x, period):
    """Wrap angles to [-period/2, period/2)."""
    return (x + period / 2) % period - period / 2


# ----------------------------------------------------------------------------
# Reading AO2D files 
# ----------------------------------------------------------------------------
def read_trees(filepaths, apply_cuts, pt_min, eta_max, charged_only):
    """Read all DF_* directories into one event table and one particle table
    with a global event index ('evt')."""
    import uproot

    ev_list, pt_list = [], []
    offset = 0
    n_raw = 0
    for filepath in filepaths:
        print(f"Reading {filepath}")
        with uproot.open(filepath) as root_file:
            df_keys = sorted({k.split(";")[0] for k in root_file.keys() if k.startswith("DF_")})
            for df_key in df_keys:
                df_dir = root_file[df_key]
                dir_keys = {k.split(";")[0] for k in df_dir.keys()}
                if EVENT_TREE_NAME not in dir_keys or PART_TREE_NAME not in dir_keys:
                    continue
                ev = df_dir[EVENT_TREE_NAME].arrays(EVENT_COLS, library="pd")
                pt = df_dir[PART_TREE_NAME].arrays(PART_COLS, library="pd")
                n_ev = len(ev)
                n_raw += len(pt)

                if apply_cuts:
                    keep = (pt["fPt"] >= pt_min) & (pt["fEta"].abs() <= eta_max)
                    if charged_only:
                        keep &= pt["fCharge"].abs() == 1
                    pt = pt[keep]

                # the particle index is local to each DF_ directory -> make it global
                local = pt["fIndexOmniGenEvents"].to_numpy()
                ok = (local >= 0) & (local < n_ev)
                pt = pt[ok].copy()
                pt["evt"] = local[ok] + offset
                ev = ev.copy()
                ev["evt"] = np.arange(n_ev) + offset
                offset += n_ev

                ev_list.append(ev)
                pt_list.append(pt.drop(columns=["fIndexOmniGenEvents"]))

    ev_df = pd.concat(ev_list, ignore_index=True)
    pt_df = pd.concat(pt_list, ignore_index=True)
    print(f"Read {len(ev_df)} events; particles before cuts: {n_raw}, after cuts: {len(pt_df)}")
    return ev_df, pt_df


# ----------------------------------------------------------------------------
# Feature computation
# ----------------------------------------------------------------------------
def compute_features(ev_df, pt_df, max_part=None, min_particles=1, use_pid=False):
    """Return (ev_out, pt_out, feat_cols):
       ev_out : DataFrame, one row per kept event, columns JET_NAMES (+ 'evt')
       pt_out : DataFrame, one row per stored particle, with 'row' (event row in ev_out),
                'slot' (position in the padded array), 'fPdgCode' and the feature columns
    """
    pt = pt_df.sort_values(["evt", "fPt"], ascending=[True, False], kind="stable").copy()
    pt["slot"] = pt.groupby("evt").cumcount()
    if max_part is not None:
        pt = pt[pt["slot"] < max_part]

    counts = pt.groupby("evt").size()
    good_evts = counts.index[counts >= min_particles]   # with enough particles
    pt = pt[pt["evt"].isin(good_evts)]
    ev = ev_df[ev_df["evt"].isin(good_evts)].sort_values("evt").reset_index(drop=True)
    ev["row"] = np.arange(len(ev))
    pt = pt.merge(ev[["evt", "row"]], on="evt", how="inner")

    # --- event-level quantities from the stored particles; their use is described in Rita's paper 
    pt["_c2"] = pt["fPt"] * np.cos(2 * pt["fPhi"])
    pt["_s2"] = pt["fPt"] * np.sin(2 * pt["fPhi"])
    g = pt.groupby("row")
    agg = pd.DataFrame({
        "e_pT": g["fPt"].mean(),
        "sum_pT": g["fPt"].sum(),
        "Qx": g["_c2"].sum(),
        "Qy": g["_s2"].sum(),
        "eta_jet": g["fEta"].mean(),
        "e_phi": g["fPhi"].mean(),
        "n_particles": g.size(),
    }).reindex(ev["row"])

    ep = 0.5 * np.arctan2(agg["Qy"].to_numpy(), agg["Qx"].to_numpy())

    ev_out = pd.DataFrame({
        "e_pT": agg["e_pT"].to_numpy(),
        "sum_pT": agg["sum_pT"].to_numpy(),
        "Qx": agg["Qx"].to_numpy(),
        "Qy": agg["Qy"].to_numpy(),
        "eta_jet": agg["eta_jet"].to_numpy(),
        "ep_angle": ep,
        "b": ev["fImpactParameter"].to_numpy(),
        "n_coll": ev["fNColl"].to_numpy(),
        "n_part": (ev["fNPartProj"] + ev["fNPartTarg"]).to_numpy(),
        "e_phi": agg["e_phi"].to_numpy(),
        "n_particles": agg["n_particles"].to_numpy(),
        "evt": ev["evt"].to_numpy(),
    })

    # --- particle-level features (relative to the event means)
    pt = pt.merge(ev_out[["e_pT", "eta_jet", "e_phi"]].assign(row=np.arange(len(ev_out))), on="row")
    pt["phi_rel"] = wrap_angle(pt["fPhi"] - pt["e_phi"], 2 * np.pi)
    pt["pT_rel"] = np.log(pt["fPt"] / pt["e_pT"])
    pt["eta_rel"] = pt["fEta"] - pt["eta_jet"]
    pt.loc[pt["eta_rel"] == 0, "eta_rel"] = MASK_EPS
    pt["pT"] = pt["fPt"]
    pt["eta"] = pt["fEta"]
    pt["phi"] = pt["fPhi"]
    feat_cols = list(KIN_NAMES)

    if use_pid:
        pt["charge"] = pt["fCharge"].astype(np.float32)
        absid = pt["fPdgCode"].abs()
        listed = np.zeros(len(pt), dtype=bool)
        for name, codes in SPECIES.items():
            pt[f"is_{name}"] = absid.isin(codes).astype(np.float32)
            listed |= absid.isin(codes).to_numpy()
        pt["is_other"] = (~listed).astype(np.float32)
        feat_cols += ["charge"] + [f"is_{n}" for n in SPECIES_NAMES]

    return ev_out, pt[["row", "slot", "fPdgCode"] + feat_cols], feat_cols


def pad_events(pt_rows, rows, n_max, feat_cols):
    """Build the zero-padded particle array and PDG array for the given event rows."""
    local = {r: i for i, r in enumerate(rows)}
    sub = pt_rows[pt_rows["row"].isin(rows)]
    ii = sub["row"].map(local).to_numpy()
    jj = sub["slot"].to_numpy()
    X = np.zeros((len(rows), n_max, len(feat_cols)), dtype=np.float32)
    P = np.zeros((len(rows), n_max), dtype=np.int32)
    X[ii, jj] = sub[feat_cols].to_numpy(dtype=np.float32)
    P[ii, jj] = sub["fPdgCode"].to_numpy(dtype=np.int32)
    return X, P


def normalisation(ev_out, pt_rows, train_rows, feat_cols):
    """Mean/std on the training split. One-hot and charge columns keep mean 0, std 1
    (as in the EIC loader), so they stay exactly 0/1 and +-1."""
    tr = pt_rows[pt_rows["row"].isin(train_rows)]
    mean_part, std_part = [], []
    for c in feat_cols:
        if c == "charge" or c.startswith("is_"):
            mean_part.append(0.0); std_part.append(1.0)
        else:
            s = float(tr[c].std())
            mean_part.append(float(tr[c].mean())); std_part.append(s if s > 0 else 1.0)
    ev_tr = ev_out.iloc[train_rows]
    mean_jet = [float(ev_tr[c].mean()) for c in JET_NAMES]
    std_jet = [float(ev_tr[c].std()) if ev_tr[c].std() > 0 else 1.0 for c in JET_NAMES]
    return {
        "part_names": feat_cols, "mean_part": mean_part, "std_part": std_part,
        "jet_names": JET_NAMES, "mean_jet": mean_jet, "std_jet": std_jet,
    }


# ----------------------------------------------------------------------------
# Writing the hdf5 file; utils.lhcHIDataLoader will later load it to memory
# In the written file, features are NOT normalised. Normalisation happens at loading.
# ----------------------------------------------------------------------------
def write_split(path, ev_out, pt_rows, rows, n_max, feat_cols, chunk=500):
    import h5py

    n = len(rows)
    with h5py.File(path, "w") as f:
        dX = f.create_dataset("particle", (n, n_max, len(feat_cols)), dtype="float32",
                              chunks=(1, n_max, len(feat_cols)), compression="lzf")
        dP = f.create_dataset("pdg", (n, n_max), dtype="int32", chunks=(1, n_max), compression="lzf")
        f.create_dataset("jet", data=ev_out.iloc[rows][JET_NAMES].to_numpy(dtype=np.float32))
        for s in range(0, n, chunk):
            X, P = pad_events(pt_rows, rows[s:s + chunk], n_max, feat_cols)
            dX[s:s + len(X)] = X
            dP[s:s + len(P)] = P
        f.attrs["part_names"] = feat_cols
        f.attrs["jet_names"] = JET_NAMES
    print(f"Wrote {path}: {n} events, particle array {(n, n_max, len(feat_cols))}")


def summary(ev_out, pt_rows, use_pid):
    print("\n" + "=" * 70)
    print(f"Events kept: {len(ev_out)}, particles stored: {len(pt_rows)}")
    print("Multiplicity:", ev_out["n_particles"].describe()[["mean", "std", "min", "max"]].round(1).to_dict())
    if use_pid:
        shares = {n: round(100 * float(pt_rows[f"is_{n}"].mean()), 2) for n in SPECIES_NAMES}
        print("Species shares [%]:", shares)
    print("=" * 70 + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--files", nargs="+", default=["data/run_oo_batch_*/AnalysisResults_trees.root"])
    p.add_argument("--out_dir", required=True, help="folder passed as --folder to train_lhcHI.py")
    p.add_argument("--no_cuts", action="store_true", help="disable pT, eta and charge cuts")
    p.add_argument("--pt_min", type=float, default=PT_MIN_CUT)
    p.add_argument("--eta_max", type=float, default=ETA_ABS_CUT)
    p.add_argument("--all_charges", action="store_true", help="keep neutral particles too")
    p.add_argument("--pid", action="store_true", help="append charge + one-hot species columns")
    p.add_argument("--max_part", type=int, default=None, help="cap on particles per event (highest pT kept)")
    p.add_argument("--min_particles", type=int, default=1)
    p.add_argument("--split", type=float, nargs=3, default=[0.70, 0.15, 0.15], help="train test val fractions")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    files = sorted({f for pat in args.files for f in glob.glob(pat)})
    if not files:
        raise SystemExit(f"No files found for {args.files}")

    ev_df, pt_df = read_trees(files, not args.no_cuts, args.pt_min, args.eta_max, not args.all_charges)
    ev_out, pt_rows, feat_cols = compute_features(ev_df, pt_df, args.max_part, args.min_particles,
                                                  args.pid)
    summary(ev_out, pt_rows, args.pid)

    n_max = int(ev_out["n_particles"].max())
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(ev_out))
    n_tr = int(args.split[0] * len(perm))
    n_te = int(args.split[1] * len(perm))
    splits = {"train": np.sort(perm[:n_tr]), "test": np.sort(perm[n_tr:n_tr + n_te]),
              "val": np.sort(perm[n_tr + n_te:])}

    out = os.path.join(args.out_dir, "lhcHI")
    os.makedirs(out, exist_ok=True)
    for name, rows in splits.items():
        write_split(os.path.join(out, f"{name}_lhcHI.h5"), ev_out, pt_rows, rows, n_max, feat_cols)

    norm = normalisation(ev_out, pt_rows, splits["train"], feat_cols)
    i_n = JET_NAMES.index("n_particles")
    norm["revert_npart"] = {"key": str(n_max), "mean": norm["mean_jet"][i_n], "std": norm["std_jet"][i_n]}
    with open(os.path.join(out, "normalisation.json"), "w") as f:
        json.dump(norm, f, indent=2)

    print("Copy into utils.lhcHIDataLoader:")
    print(f"  self.part_names = {feat_cols}")
    print(f"  self.mean_part = {np.round(norm['mean_part'], 5).tolist()}")
    print(f"  self.std_part  = {np.round(norm['std_part'], 5).tolist()}")
    print(f"  self.mean_jet  = {np.round(norm['mean_jet'], 5).tolist()}")
    print(f"  self.std_jet   = {np.round(norm['std_jet'], 5).tolist()}")
    print("and into utils.revert_npart stats:")
    print(f'  "{n_max}": ({norm["revert_npart"]["mean"]:.5f}, {norm["revert_npart"]["std"]:.5f}),')
    print("PET_lhcHI._part_mean/_part_std (pT and phi, used by the L_psi loss) and "
          "_jet_mean/_jet_std must be updated as well.")


if __name__ == "__main__":
    main()