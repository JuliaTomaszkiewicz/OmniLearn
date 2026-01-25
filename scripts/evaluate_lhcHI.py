from mpi4py import MPI

import os
import h5py as h5
import numpy as np
import tensorflow as tf
from tensorflow import keras
import horovod.tensorflow.keras as hvd
import argparse
import pickle
from PET_lhcHI import PET_lhcHI
import utils
import plot_utils
import matplotlib.pyplot as plt
import logging

hvd.init()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

base_vars = 4
add_vars = 2

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process elec data.")
    parser.add_argument("--dataset", type=str, default="lhcHI", help="Dataset to use")
    parser.add_argument("--folder", default="/pscratch/sd/v/vmikuni/PET/", help="Folder containing input files")
    parser.add_argument("--mode", default="generator", help="Loss type to train the model: [all/classifier/generator]")
    parser.add_argument("--fine_tune", action='store_true', help="Fine tune a model")
    parser.add_argument("--local", action='store_true', help="Use local embedding")
    parser.add_argument("--num_layers", type=int, default=8, help="Number of transformer layers")
    parser.add_argument("--nevts", type=int, default=-1, help="Number of events to load")
    parser.add_argument("--drop_probability", type=float, default=0.0, help="Stochastic Depth drop probability")
    parser.add_argument("--simple", action='store_true', help="Use simplified head model")
    parser.add_argument("--talking_head", action='store_true', help="Use talking head attention")
    parser.add_argument("--layer_scale", action='store_true', help="Use layer scale in the residual connections")
    parser.add_argument("--sample", action='store_true', help="Sample from trained model")
    parser.add_argument("--plot_folder", default="../plots", help="Folder to save the outputs")
    return parser.parse_args()

def get_data_info(flags):
    test = utils.lhcHIDataLoader(os.path.join(flags.folder,'lhcHI','val_lhcHI.h5'), rank=hvd.rank(), size=hvd.size())            
    return test


def load_data_and_model(flags):
    
    test = get_data_info(flags)
    model = PET_lhcHI(num_feat=test.num_feat,
                          num_jet=test.num_jet,
                          num_classes=test.num_classes,
                          num_part=test.num_part,
                          local=flags.local,
                          num_layers=flags.num_layers,
                          drop_probability=flags.drop_probability,
                          simple=flags.simple, layer_scale=flags.layer_scale,
                          talking_head=flags.talking_head,
                          mode=flags.mode, fine_tune=False, model_name=None)

    #print(model.summary())
    
    model_name = os.path.join(flags.folder, 'checkpoints', utils.get_model_name(flags, flags.fine_tune))
    model.load_weights(model_name)
    return test, model


def sample_data(test, model, flags, sample_name):
    """ Sample data using the model and save to file. """
    y, j = test.y[:], None
    
    nsplit = 60 

    p, j = model.generate(y, jets=j, nsplit=nsplit,use_tqdm=hvd.rank()==0)
    p = test.revert_preprocess(p, p[:, :, 2] != 0)
    j = test.revert_preprocess_jet(j)

    print("Finished generating. preprocess. allgather. saving:")

    p = p.astype(np.float16)
    j = j.astype(np.float16)

    out_rank = sample_name.replace(".h5", f".rank{hvd.rank():04d}.h5")
    with h5.File(out_rank, "w") as h5f:
        h5f.create_dataset("data", data=p,  compression="gzip", chunks=True)
        h5f.create_dataset("jet",  data=j,  compression="gzip", chunks=True)



def main():
    plot_utils.SetStyle()
    utils.setup_gpus()
    if hvd.rank()==0:logging.info("Horovod and GPUs initialized successfully.")
    flags = parse_arguments()

    os.makedirs(flags.plot_folder, exist_ok=True)

    sample_name = os.path.join(flags.folder, 'lhcHI',
                               utils.get_model_name(flags, flags.fine_tune).replace(".weights.h5", ".h5"))
   

    print("The sample_name: ", sample_name)


    if hvd.rank()==0:logging.info("Sampling the data.")
    test, model = load_data_and_model(flags)
    sample_data(test, model, flags, sample_name)


if __name__ == '__main__':
    main()


