import os
import glob
import argparse
import uproot
import numpy as np
import pandas as pd

APPLY_MANUAL_CUTS = True

PT_MIN_CUT = 1.0
ETA_ABS_CUT = 0.9
CHARGED_ONLY = True

EVENT_TREE_NAME = "O2omnigenevent"
PART_TREE_NAME = "O2omnigenpart"

EVENT_COLS = ["fImpactParameter", "fReactionPlane", "fNColl", "fNPartProj", "fNPartTarg", "fPosZ", "fNParticles"]
PART_COLS = ["fIndexOmniGenEvents", "fPt", "fEta", "fPhi", "fPdgCode", "fCharge"]

PDG_NAMES = {
    211: "pi+", -211: "pi-", 111: "pi0", 321: "K+", -321: "K-",
    2212: "proton (p)", -2212: "antiproton (p-bar)", 2112: "neutron (n)", -2112: "antineutron (n-bar)",
    11: "e-", -11: "e+", 13: "mu-", -13: "mu+", 22: "gamma",
}


def process_and_convert(filepaths: list, apply_cuts: bool):
    all_events = []
    all_particles = []
    raw_mult_list = []

    raw_total_particles = 0
    filtered_total_particles = 0

    for filepath in filepaths:
        if not os.path.exists(filepath):
            continue

        h5_path = os.path.splitext(filepath)[0] + ".h5"
        if os.path.exists(h5_path):
            os.remove(h5_path)

        print(f"Processing and converting: {filepath} -> {h5_path}")

        with uproot.open(filepath) as root_file:
            df_keys = sorted({k.split(";")[0] for k in root_file.keys() if k.startswith("DF_")})

            for df_key in df_keys:
                df_dir = root_file[df_key]
                dir_keys = {k.split(";")[0] for k in df_dir.keys()}

                if EVENT_TREE_NAME not in dir_keys or PART_TREE_NAME not in dir_keys:
                    continue

                ev_df = df_dir[EVENT_TREE_NAME].arrays(EVENT_COLS, library="pd")
                pt_df = df_dir[PART_TREE_NAME].arrays(PART_COLS, library="pd")

                n_ev = len(ev_df)
                raw_total_particles += len(pt_df)
                raw_mult_list.append(ev_df["fNParticles"].copy())

                if apply_cuts:
                    mask = (pt_df["fPt"] >= PT_MIN_CUT) & (pt_df["fEta"].abs() <= ETA_ABS_CUT)
                    if CHARGED_ONLY:
                        mask &= (pt_df["fCharge"].abs() == 1)

                    pt_df = pt_df[mask]

                    valid_idx = pt_df["fIndexOmniGenEvents"]
                    valid_idx = valid_idx[(valid_idx >= 0) & (valid_idx < n_ev)]
                    ev_df["fNParticles"] = np.bincount(valid_idx, minlength=n_ev).astype(np.int32)

                filtered_total_particles += len(pt_df)

                ev_df.to_hdf(h5_path, key=f"{df_key}_{EVENT_TREE_NAME}", mode="a", format="table", append=True)
                pt_df.to_hdf(h5_path, key=f"{df_key}_{PART_TREE_NAME}", mode="a", format="table", append=True)

                all_events.append(ev_df)
                all_particles.append(pt_df.drop(columns=["fIndexOmniGenEvents"]))

    full_ev_df = pd.concat(all_events, ignore_index=True)
    full_pt_df = pd.concat(all_particles, ignore_index=True)
    raw_mult_series = pd.concat(raw_mult_list, ignore_index=True)

    stats_cols = ["mean", "std", "min", "25%", "50%", "75%", "max"]

    print("\n" + "=" * 80)
    print(f"DATASET SUMMARY (Cuts: {'YES' if apply_cuts else 'NO'})")
    print("=" * 80)
    kept_pct = (100.0 * filtered_total_particles / raw_total_particles) if raw_total_particles > 0 else 0.0
    print(f"Processed files:             {len(filepaths)}")
    print(f"Total events:                {len(full_ev_df)}")
    print(f"Particles BEFORE cuts:       {raw_total_particles}")
    print(f"Particles AFTER cuts:        {filtered_total_particles} ({kept_pct:.2f}%)")

    print("\n--- Multiplicity (fNParticles): Before vs After Cuts ---")
    mult_stats = pd.DataFrame({
        "Before cuts": raw_mult_series.describe(),
        "After cuts": full_ev_df["fNParticles"].describe()
    }).T[stats_cols]
    print(mult_stats.to_string())

    print("\n--- Feature Distributions: O2omnigenevent (Event-Level) ---")
    print(full_ev_df[EVENT_COLS].describe().T[stats_cols].to_string())

    print("\n--- Feature Distributions: O2omnigenpart (Particle-Level, after cuts) ---")
    print(full_pt_df.describe().T[stats_cols].to_string())

    print("\n--- Particle Distribution by PDG Code (Top 15) ---")
    pdg_counts = full_pt_df["fPdgCode"].value_counts().head(15)
    pdg_table = pd.DataFrame({
        "PDG Code": pdg_counts.index,
        "Particle Name": [PDG_NAMES.get(int(code), "Other/Nucleus") for code in pdg_counts.index],
        "Count": pdg_counts.values,
        "Share [%]": (100.0 * pdg_counts.values / filtered_total_particles).round(2) if filtered_total_particles > 0 else 0.0
    })
    print(pdg_table.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--FILES_TO_ANALYZE",
        "--FILES_TO_ANALYSE",
        nargs="+",
        default=["data/run_oo_batch_*/AnalysisResults_trees.root"]
    )
    args = parser.parse_args()

    files_to_analyze = sorted({f for pattern in args.FILES_TO_ANALYZE for f in glob.glob(pattern)})

    if not files_to_analyze:
        print(f"[ERROR] No *.root files found for: {args.FILES_TO_ANALYZE}")
    else:
        process_and_convert(files_to_analyze, APPLY_MANUAL_CUTS)
