from dataclasses import dataclass
from collections import OrderedDict
from anytree import NodeMixin
import numpy as np
import logging

@dataclass
class Cluster(NodeMixin):
    # Cluster attributes
    is_active: bool = True
    ids: np.ndarray # shape: (n, )

    # Diameter statistics attributes
    d0: float = None
    d0_ids: tuple = None
    d1: float = None
    d1_ids: tuple = None
    d2: float = None
    d2_ids: tuple = None
    delta: float = None
    davg: float = None

    hoeffding_bound: float = None
    min_value: float = -np.inf
    max_value: float = np.inf
    Rsq: float = None

    confidence_level: float = 0.9
    n_min: int = 5
    tau: float = 0.1

    # Stream attributes
    t: int = 0

    # Distance attributes
    Dsq: np.ndarray = None # distance matrix    

    def __post_init__(self):
        n = len(self.ids)
        self.Dsq = np.zeros((n, n))

    # def init_distances(self):
    #     """Initialize the distance matrix"""
    #     self.Dsq = np.linalg.norm(self.data - self.data[:, None], axis=-1)**1

    def update_diameter_coefficients(self):
        """Initialize the diameter"""
        dshape = self.Dsq.shape

        # Get the minimum distance
        d0_idflat = np.argmin(self.Dsq)
        self.d0_ids = np.unravel_index(d0_idflat, dshape)
        self.d0 = self.Dsq[self.d0_ids]

        # Get the maximum distance
        d1_idflat = np.argmax(self.Dsq)
        self.d1_ids = np.unravel_index(d1_idflat, dshape)
        self.d1 = self.Dsq[self.d1_ids]

        # Get the 2nd maximum distance
        self.Dsq[self.d1_ids] = -np.inf
        d2_idflat = np.argmax(self.Dsq)
        self.d2_ids = np.unravel_index(d2_idflat, dshape)
        self.d2 = self.Dsq[self.d2_ids]
        self.Dsq[self.d1_ids] = self.d1

        self.delta = self.d1 - self.d2

        # Get the average distance
        self.davg = np.mean(self.Dsq)


    def init_range(self):
        """Initialize the range statistic"""
        self.min_value = np.min(self.data)
        self.max_value = np.max(self.data)
        self.Rsq = (self.max_value - self.min_value)
        self.Rsq *= self.Rsq

    def update_range(self,vals):
        """Update the range statistic"""
        self.min_value = np.min([self.min_value, np.min(vals)])
        self.max_value = np.max([self.max_value, np.max(vals)])
        self.Rsq = (self.max_value - self.min_value)
        self.Rsq *= self.Rsq

    def update_hoeffding(self):
        """Update the Hoeffding bound on the error of the diameter statistics"""
        self.hoeffding_bound = np.sqrt(self.Rsq * np.log(1 / self.confidence_level) / (2 * self.t))

    def update(self, t:int, idxs:np.ndarray, vals:np.ndarray):
        """Update the cluster with multiple observations"""
        # Check input
        assert len(idxs) == len(vals)
        assert len(idxs) > 0
        assert t > self.t

        # Cut to ids and values that are in this cluster
        tmp = np.isin(idxs, self.ids)
        idxs = idxs[tmp]
        vals = vals[tmp]

        # Take last observation if duplicates exist
        idxs, unique_idx = np.unique(idxs, return_index=True)
        vals = np.array(vals)[unique_idx]

        # Get the local indices of the updated observations through idx
        # TODO: test
        update_ids = [np.where(self.ids == idx)[0][0] for idx in idxs]

        # Update data matrix (add zero times if necessary)
        n,m = self.data.shape
        new_cols = np.zeros((n, t - self.t))
        new_cols[update_ids, -1] = vals
        self.data = np.hstack([self.data, new_cols])

        # Update distance matrix
        new_vals = new_cols[:, -1]
        new_diffs = new_vals - new_vals[:, None]
        Dsq += new_diffs**2

        # Update diameter statistics
        self.update_range(vals)
        self.update_hoeffding()
        self.update_diameter_coefficients()


        # TODO: check if t statistic is correct according to hoeffding bound
        self.t = t

    def check_split(self):
        """Test if the cluster should be split, if so, do it."""

        if self.t <= self.n_min:
            return False
        
        # Check if the Hoeffding bound is violated
        e = self.hoeffding_bound
        if ( self.delta > e ) or ( self.tau > e ):
            if ( (self.d1 - self.d0) * abs((self.d1 - self.davg) - (self.davg - self.d0)) ) > e:
                self.split()
                return True

        return False
    
    def split(self):
        """Split the cluster using d1_idx as pivots"""

        logging.info(f"Splitting cluster with {self.t} observations and pivot indices {self.ids[self.d1_ids]}")

        # Get the pivot indices
        x1, y1 = self.d1_ids

        # Make new clusters based on minimum distance to pivot
        c1_ids = np.argmin(self.Dsq[x1], axis=0)
        c2_ids = np.argmin(self.Dsq[y1], axis=0)

        # Get the new cluster data
        c1_data = self.data[c1_ids]
        c2_data = self.data[c2_ids]

        # Get the new cluster ids
        c1_ids = self.ids[c1_ids]
        c2_ids = self.ids[c2_ids]

        # Create new clusters
        c1 = Cluster(ids=c1_ids, data=c1_data)
        c2 = Cluster(ids=c2_ids, data=c2_data)

        # Set the new clusters as children
        self.children = [c1, c2]

        # Set the current cluster as parent
        c1.parent = self
        c2.parent = self

        # Set the current cluster as inactive
        self.is_active = False

    




