import sys
import logging
import numpy as np
from methods.cluster import Cluster


# Setup logger
FORMAT = '%(asctime)s - [%(levelname)s] %(message)s'
logging.basicConfig(format=FORMAT, datefmt='%d/%m/%Y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)


def get_data(n_streams:int, duration:int):
    """
    Get synthetic data from /home/jens/tue/0.STELAR/1.Agroknow/data/weekly_syn_r.csv
    """
    return np.genfromtxt("/home/jens/tue/0.STELAR/1.Agroknow/data/weekly_syn_r.csv", delimiter=",", max_rows=n_streams, usecols=range(duration))

def main(n_streams:int, duration:int, tau:float):
    """
    Main function; simulate a stream and continuously maintain a cluster tree
    """

    logging.info(f"Starting main function with n_streams={n_streams} and duration={duration}")

    # Get data
    data = get_data(n_streams, duration)

    # Initialize root node of the tree (the initial cluster) and set
    root = Cluster(ids=np.arange(n_streams), tau=tau)

    T = 0

    # Simulate stream
    while T < duration:
        # Get all positive updates
        tcol = data[:,T]
        update_ids = np.nonzero(tcol)[0]
        update_vals = tcol[update_ids]

        logging.info(f"T={T} - Number of updates: {len(update_ids)}")

        # Update the clusters
        for c in root.get_leaves():
            if c.is_singleton():
                continue

            c.update(update_ids, update_vals)

            # Check if the cluster needs to split or merge
            if c.check_split() or c.check_merge():
                pass
                # logging.info(f"T={T} - New tree after split/merge of cluster {c.identifier}:")
                # root.print_tree()

        # Increment time
        T += 1

    logging.info(f"Final tree:")
    root.print_tree()

if __name__ == "__main__":

    logging.info(f"Arguments: {sys.argv}")

    # Take in arguments n_streams, duration from command line, otherwise use default values
    if len(sys.argv) > 2:
        n_streams = int(sys.argv[1])
        duration = int(sys.argv[2])
    else:
        n_streams = 100
        duration = 1000

    # Optional parameters
    if len(sys.argv) > 3:
        tau = float(sys.argv[3])
    else:
        tau = 1

    main(n_streams, duration, tau)
