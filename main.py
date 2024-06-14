import sys
import logging
import numpy as np
from methods.cluster import Cluster
import argparse

# Setup logger
FORMAT = '%(asctime)s - [%(levelname)s] %(message)s'
logging.basicConfig(format=FORMAT, datefmt='%d/%m/%Y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)

# Set up argument parser
argparse = argparse.ArgumentParser(description="Simulate a stream and continuously maintain a cluster tree")
argparse.add_argument("-i", "--input_path", type=str, help="Path to the csv file containing the stream data", required=True) 
argparse.add_argument("-n", "--n_streams",  type=int, help="Number of streams to consider", default=None)
argparse.add_argument("-d", "--duration", type=int, help="Duration of the stream; how many timesteps we want to simulate", default=None)
argparse.add_argument("-t", "--tau", type=float, help="Threshold for the cluster tree", default=1)
argparse.add_argument("--index", type=bool, help="Flag to indicate if the input file has an index column", default=False)
argparse.add_argument("--header", type=bool, help="Flag to indicate if the input file has a header", default=False)

def get_data(path:str, n_streams:int=None, duration:int=None, index:bool=False, header:bool=False):
    """
    Get data from csv file
    """
    try:
        startrow = 1 if header else 0
        startcol = 1 if index else 0

        # Check if n_streams are valid
        with open(path) as f:
            header = f.readline().split(",")
            n_cols = len(header)
            if n_streams is not None and n_streams > n_cols:
                n_streams = n_cols - 1

            # Get ids from header if possible
            names = np.array(header[startcol:n_streams+startcol])

        if n_streams is None and duration is None:
            data = np.genfromtxt(path, delimiter=",", skip_header=startrow, usecols=range(startcol, None))
        elif n_streams is None:
            data = np.genfromtxt(path, delimiter=",", max_rows=duration, skip_header=startrow, usecols=range(startcol, None))
        elif duration is None:
            data = np.genfromtxt(path, delimiter=",", usecols=range(startcol, n_streams+startcol), skip_header=startrow)
        else:
            data = np.genfromtxt(path, delimiter=",", max_rows=duration, usecols=range(startcol, n_streams+startcol), skip_header=startrow)

        if not header:
            names = np.arange(n_streams)

        return data, names
    except Exception as e:
        logging.error(f"Error while reading data: {e}, data should be in csv format with columns [Date, Stream1, Stream2, ...]")
        sys.exit(1)

def main(input_path:str, n_streams:int, duration:int, tau:float, index:bool, header:bool):
    """
    Main function; simulate a stream and continuously maintain a cluster tree
    """

    logging.info(f"Starting main function with n_streams={n_streams} and duration={duration}")

    # Get data
    data, names = get_data(input_path, n_streams, duration, index, header)

    duration, n_streams = data.shape

    # Initialize root node of the tree (the initial cluster) and set
    root = Cluster(ids=np.arange(n_streams), names=names, tau=tau)

    T = 0

    # Simulate stream
    while T < duration:
        # Get all positive updates
        arrivals = data[T]
        update_ids = np.nonzero(arrivals)[0]
        update_vals = arrivals[update_ids]

        if len(update_ids) == 0:
            T += 1
            continue

        logging.info(f"T={T} - Number of updates: {len(update_ids)}")

        # Update the clusters
        for c in root.get_leaves():
            if c.is_singleton():
                continue

            c.update(update_ids, update_vals)

            # Check if the cluster needs to split or merge
            actions = []
            if c.check_split():
                actions.append("split")
            if c.check_merge():
                actions.append("merge")
            if len(actions) > 0:
                logging.info(f"T={T} - New tree after {' + '.join(actions)} of cluster {c.identifier}:")
                root.print_tree()

        # Increment time
        T += 1

    logging.info(f"Final tree:")
    root.print_tree()

if __name__ == "__main__":

    logging.info(f"Arguments: {sys.argv}")

    if len(sys.argv) == 1:
        input_path = "/home/jens/ownCloud/Documents/3.Werk/0.TUe_Research/0.STELAR/1.Agroknow/data/weekly.csv"
        n_streams = 100
        duration = None
        tau = 1
        index = True
        header = True
        g = main(input_path, n_streams, duration, tau, index, header)
    else:
        args = argparse.parse_args()
        main(args.input_path, args.n_streams, args.duration, args.tau, args.index, args.header)