
import os
import h5py as h5
import numpy as np
import horovod.tensorflow.keras as hvd
import argparse
from PET_lhcHI import PET_lhcHI
import utils
import plot_utils
import logging
import glob
import matplotlib.pyplot as plt

hvd.init()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

base_vars = 4
add_vars = 2


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process elec data.")
    parser.add_argument("--dataset", type=str, default="lhcHI", help="Dataset to use")
    parser.add_argument(
        "--folder",
        default="/pscratch/sd/v/vmikuni/PET/",
        help="Folder containing input files",
    )
    parser.add_argument(
        "--mode",
        default="generator",
        help="Loss type to train the model: [all/classifier/generator]",
    )
    parser.add_argument("--fine_tune", action="store_true", help="Fine tune a model")
    parser.add_argument("--local", action="store_true", help="Use local embedding")
    parser.add_argument(
        "--num_layers", type=int, default=8, help="Number of transformer layers"
    )
    parser.add_argument(
        "--nevts", type=int, default=-1, help="Number of events to load"
    )
    parser.add_argument(
        "--drop_probability",
        type=float,
        default=0.0,
        help="Stochastic Depth drop probability",
    )
    parser.add_argument(
        "--simple", action="store_true", help="Use simplified head model"
    )
    parser.add_argument(
        "--talking_head", action="store_true", help="Use talking head attention"
    )
    parser.add_argument(
        "--layer_scale",
        action="store_true",
        help="Use layer scale in the residual connections",
    )
    parser.add_argument(
        "--sample", action="store_true", help="Sample from trained model"
    )
    parser.add_argument(
        "--plot_folder", default="../plots", help="Folder to save the outputs"
    )
    return parser.parse_args()


def get_data_info(flags):
    test = utils.lhcHIDataLoader(
        os.path.join(flags.folder, "lhcHI", "val_lhcHI.h5"),
        rank=hvd.rank(),
        size=hvd.size(),
    )
    return test


def load_data_and_model(flags):

    test = get_data_info(flags)
    model = PET_lhcHI(
        num_feat=test.num_feat,
        num_jet=test.num_jet,
        num_classes=test.num_classes,
        num_part=test.num_part,
        local=flags.local,
        num_layers=flags.num_layers,
        drop_probability=flags.drop_probability,
        simple=flags.simple,
        layer_scale=flags.layer_scale,
        talking_head=flags.talking_head,
        mode=flags.mode,
        fine_tune=False,
        model_name=None,
    )

    # print(model.summary())

    model_name = os.path.join(
        flags.folder, "checkpoints", utils.get_model_name(flags, flags.fine_tune)
    )
    model.load_weights(model_name)
    return test, model


def sample_data(test, model, flags, sample_name):
    """Sample data using the model and save to file."""
    nevts = flags.nevts if flags.nevts > 0 else len(test.y)
    y, j = test.y[:nevts], None

    nsplit = min(60, len(y))

    p, j = model.generate(y, jets=j, nsplit=nsplit, use_tqdm=hvd.rank() == 0)
    p = test.revert_preprocess(p, p[:, :, 2] != 0)
    j = test.revert_preprocess_jet(j)

    print("Finished generating. preprocess. allgather. saving:")

    p = p.astype(np.float16)
    j = j.astype(np.float16)

    out_rank = sample_name.replace(".h5", f".rank{hvd.rank():04d}.h5")
    with h5.File(out_rank, "w") as h5f:
        h5f.create_dataset("data", data=p, compression="gzip", chunks=True)
        h5f.create_dataset("jet", data=j, compression="gzip", chunks=True)

    if hvd.rank() == 0:
        rank_files = sorted(glob.glob(sample_name.replace(".h5", ".rank*.h5")))
        with h5.File(sample_name, "w") as h5f:
            h5f.create_dataset("data", data=np.concatenate([h5.File(f, "r")["data"][:] for f in rank_files]))
            h5f.create_dataset("jet", data=np.concatenate([h5.File(f, "r")["jet"][:] for f in rank_files]))

def get_generated_data(sample_name, nevts=-1):
    with h5.File(sample_name, "r") as h5f:
        jets_gen = h5f["jet"][:].astype(np.float32)
        particles_gen = h5f["data"][:].astype(np.float32)
    if nevts > 0:
        jets_gen = jets_gen[:nevts]
        particles_gen = particles_gen[:nevts]

    mask_gen = particles_gen[:, :, 2] != 0
    particles_gen = particles_gen * mask_gen[:, :, None]
    return jets_gen, particles_gen


def get_from_dataloader(test, nevts=-1):
    jets, particles = test.jet, test.X
    if nevts < 0:
        nevts = jets.shape[0]
    mask = particles[:nevts, :, 2] != 0
    particles = particles[:nevts] * mask[:, :, None]
    jets = jets[:nevts]
    return jets, particles


def plot(jet1, jet2, var_names, title, plot_folder):
    for ivar in range(len(var_names)):
        feed_dict = {
            "eic_truth": jet1[:, ivar],
            "eic_gen": jet2[:, ivar],
        }

        fig, gs, binning = plot_utils.HistRoutine(
            feed_dict,
            xlabel=var_names[ivar],
            plot_ratio=True,
            logy=ivar == 0,
            reference_name="eic_truth",
            ylabel="Normalized entries",
        )

        ax0 = plt.subplot(gs[0])

        p_dict = {
            "Particle_0": r"$\pi^{\pm}$",
            "Particle_1": r"$K^{\pm}$",
            "Particle_2": r"$p, \bar{p}$",
            "Particle_3": r"$e^{\pm}$",
            "Particle_4": r"$\mu^{\pm}$",
            "Particle_5": "other",
        }
        if title in p_dict:
            plt.title(p_dict[title], fontsize=35)

        fig.savefig("{}/lhcHI_{}_{}.pdf".format(plot_folder, title, ivar), bbox_inches="tight")
        plt.close()


def plot_results(jets, jets_gen, particles, particles_gen, flags):
    jet_var_names = [
        r"$\langle p_T \rangle$ [GeV]", r"$\Sigma p_T$ [GeV]", r"$Q_x$", r"$Q_y$",
        r"$\langle \eta \rangle$", r"$\psi_{EP}$", r"$b$ [fm]", r"$N_{coll}$",
        r"$N_{part}$", r"$\langle \phi \rangle$", "Multiplicity",
    ]

    part_var_names = [
        r"all $\phi_{rel}$", r"all $\log(p_T / \langle p_T \rangle)$", r"all $\eta_{rel}$",
        r"all $p_T$ [GeV]", r"all $\eta$", r"all $\phi$",
        "charge", "is pion", "is kaon", "is proton", "is electron", "is muon", "is other",
    ]

    plot(jets, jets_gen, var_names=jet_var_names, title="Event", plot_folder=flags.plot_folder)

    # Mask zero-padded particles (column 2 is eta_rel)
    particles_gen = particles_gen.reshape((-1, particles_gen.shape[-1]))
    particles_gen = particles_gen[particles_gen[:, 2] != 0.0]
    particles = particles.reshape((-1, particles.shape[-1]))
    particles = particles[particles[:, 2] != 0.0]

    # Inclusive plots with all particles
    plot(particles, particles_gen, var_names=part_var_names[: particles.shape[1]], title="Particle", plot_folder=flags.plot_folder)

    # Separate plots for each type of particle (if PID columns are present)
    if particles.shape[1] >= 13:
        particle_names = [r"$\pi^{\pm}$", r"$K^{\pm}$", r"$p, \bar{p}$", r"$e^{\pm}$", r"$\mu^{\pm}$", "other"]
        for pid in range(len(particle_names)):
            mask_pid = particles[:, 7 + pid] == 1
            mask_pid_gen = particles_gen[:, 7 + pid] == 1
            print(f"Fraction of {particle_names[pid]}: truth={np.mean(mask_pid):.4f}, gen={np.mean(mask_pid_gen):.4f}")

            particles_pid = particles[mask_pid]
            particles_gen_pid = particles_gen[mask_pid_gen]
            if len(particles_pid) == 0 or len(particles_gen_pid) == 0:
                continue

            var_names = [
                f"{particle_names[pid]} " + r"$\phi_{rel}$",
                f"{particle_names[pid]} " + r"$\log(p_T / \langle p_T \rangle)$",
                f"{particle_names[pid]} " + r"$\eta_{rel}$",
                f"{particle_names[pid]} " + r"$p_T$ [GeV]",
                f"{particle_names[pid]} " + r"$\eta$",
                f"{particle_names[pid]} " + r"$\phi$",
                f"{particle_names[pid]} charge",
            ]
            plot(
                particles_pid[:, :7],
                particles_gen_pid[:, :7],
                title=f"Particle_{pid}",
                var_names=var_names,
                plot_folder=flags.plot_folder,
            )

def main():
    plot_utils.SetStyle()
    utils.setup_gpus()
    if hvd.rank() == 0:
        logging.info("Horovod and GPUs initialized successfully.")
    flags = parse_arguments()

    os.makedirs(flags.plot_folder, exist_ok=True)

    sample_name = os.path.join(
        flags.folder,
        "lhcHI",
        utils.get_model_name(flags, flags.fine_tune).replace(".weights.h5", ".h5"),
    )

    print("The sample_name: ", sample_name)

    if flags.sample:
        if hvd.rank() == 0:
            logging.info("Sampling the data.")
        test, model = load_data_and_model(flags)
        sample_data(test, model, flags, sample_name)
    else:
        if hvd.rank() == 0:
            logging.info("Loading saved samples.")
            test = get_data_info(flags)
            jets, particles = get_from_dataloader(test, flags.nevts)
            jets_gen, particles_gen = get_generated_data(sample_name, flags.nevts)
            print(particles_gen.shape, particles.shape)
            plot_results(jets, jets_gen, particles, particles_gen, flags)


if __name__ == "__main__":
    main()
