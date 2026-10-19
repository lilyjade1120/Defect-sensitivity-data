from time import perf_counter

import torch

from qimpy import rc, log
from qimpy.io import log_config


def find_neighbors_all_pairs(
    RT: torch.Tensor, pos_i: torch.Tensor, pos_j: torch.Tensor, cutoff: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return indices into pos_i, pos_j that are within CUTOFF of each other, 
    in minimum image convention based on RT, and corresponding pos_i - pos_j.
    (Reference O(N^2) implementation that examines all pairs.)"""
    dx = (pos_i[:, None] - pos_j) @ torch.linalg.inv(RT)  # fractional
    dx -= torch.floor(dx + 0.5)  # minimum image convention
    displacements = dx @ RT
    distances = displacements.norm(dim=-1)
    index_i, index_j = torch.logical_and(distances > 0, distances < cutoff).nonzero().T
    return index_i, index_j, displacements[index_i, index_j]


class FindNeighbors:
    
    def __init__(
        self, RT: torch.Tensor, cutoff: float, bin_size_target: float = 10.0
    ) -> None:
        """Initialize given cell dimensions and cutoff."""
        self.RT = RT
        self.invRT = torch.linalg.inv(RT)
        self.cutoff = cutoff
        
        # Set up bins:
        Nbins = torch.ceil(RT.norm(dim=1) / bin_size_target).to(torch.int)
        if torch.all(Nbins <= 3):
            self.Nbins = None
            return  # unlikely to speed up with binning
        else:
            self.Nbins = Nbins
            log.info(f"Using {Nbins.tolist()} grid for spatial lookup")

        # Initialize bin adjacency:
        device = RT.device
        dx_bins_1D = [torch.tensor([0.0, 1/Ni], device=device) for Ni in Nbins]
        dx_bins = torch.stack(torch.meshgrid(*dx_bins_1D, indexing="ij"), dim=-1).flatten(0, 2)
        i_max = torch.ceil(
            cutoff * torch.linalg.inv(RT / Nbins[:, None]).norm(dim=0)
        ).to(torch.int) + 1
        i_grids_1D = [torch.arange(-i_max_i, i_max_i+1, device=device) for i_max_i in i_max]
        i_grids = torch.stack(torch.meshgrid(*i_grids_1D, indexing="ij"), dim=-1).flatten(0, 2)
        x_bin_edges = (i_grids / Nbins)[:, None] + dx_bins
        distances = ((x_bin_edges[..., None, :] - dx_bins) @ RT).norm(dim=-1)
        min_distances = distances.flatten(1, 2).min(dim=1)[0]
        self.neighbor_offsets = i_grids[min_distances <= cutoff]
        
    def __call__(
        self, pos_i: torch.Tensor, pos_j: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return indices into pos_i, pos_j that are within self.cutoff of each other, 
        in minimum image convention based on self.RT, and corresponding pos_i - pos_j.
        O(N) implementation using an O(1) spatial look-up."""
        if self.Nbins is None:
            return find_neighbors_all_pairs(self.RT, pos_i, pos_j, self.cutoff)

        # Convert to fractional coordinates, wrapped to [0, 1):
        bins_i, x_i_list, indices_i = self.map_positions(pos_i)
        bins_j, x_j_list, indices_j = self.map_positions(pos_j)
        
        # Create inverse lookup for neighbors:
        device = bins_j.device
        j_bin_index_map = torch.full(self.Nbins.tolist(), -1, device=device)
        j_bin_index_map[*bins_j.T] = torch.arange(len(bins_j), device=device)
        
        # Find neighbors by bin pairs:
        selected_i = []
        selected_j = []
        selected_displacements = []
        for bin_i, x_i, index_i in zip(bins_i, x_i_list, indices_i):
            # Find occupied neighboring bins:
            bin_neighbors = (bin_i + self.neighbor_offsets) % self.Nbins
            j_bin_indices = j_bin_index_map[*bin_neighbors.T]
            j_bin_indices = j_bin_indices[j_bin_indices >= 0]
            
            # Loop over bin pairs:
            for j_bin_index in j_bin_indices.tolist():
                x_j = x_j_list[j_bin_index]
                index_j = indices_j[j_bin_index]
                dx = x_i[:, None] - x_j
                dx -= torch.floor(dx + 0.5)  # minimum image convention
                displacements = dx @ self.RT
                distances = displacements.norm(dim=-1)
                local_i, local_j = torch.logical_and(
                    distances > 0, distances < self.cutoff
                ).nonzero().T
                selected_i.append(index_i[local_i])  # collect global i indices
                selected_j.append(index_j[local_j])  # collect global j indices
                selected_displacements.append(displacements[local_i, local_j])
        
        indices_i = torch.cat(selected_i)
        indices_j = torch.cat(selected_j)
        displacements = torch.cat(selected_displacements)
        sort_index = (indices_i * len(pos_j) + indices_j).argsort()  # sort by i, then j
        return indices_i[sort_index], indices_j[sort_index], displacements[sort_index]
    
    def map_positions(
        self, pos: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        """Map pos into bins, returning bin coordinates (N_occ_binsx3 int),
        fractional coordinates of atoms within each bin (list of N_in_binx3),
        and the corresponding indices into the `pos`."""
        x = pos @ self.invRT  # convert to fractional coordinates
        x -= torch.floor(x)  # ... and wrap to [0, 1)
        
        # Map to bins:
        i_bin_all = torch.floor(x * self.Nbins).to(torch.int) % self.Nbins
        bin_coords, bin_index, bin_counts = i_bin_all.unique(
            dim=0, return_inverse=True, return_counts=True
        )
        
        # Split positions by bins:
        bin_index_sorted, sort_index = bin_index.sort(dim=0)
        indices = sort_index.split(bin_counts.tolist())
        x_split = [x[index] for index in indices]
        return bin_coords, x_split, indices


if __name__ == "__main__":

    def main():
        rc.init()
        log_config()

        torch.random.manual_seed(0)
        N = 7000
        CUTOFF = 3.1
        RT = torch.diag(torch.tensor([50., 40., 45.]))
        pos_i, pos_j = torch.rand((2, N, 3)) @ RT
        RT[2, 2] *= 1.7  # test having vacuum spacing leading to empty bins
        time0 = perf_counter()
        index_i, index_j, displacements = FindNeighbors(RT, CUTOFF)(pos_i, pos_j)
        time1 = perf_counter()
        index_i_ref, index_j_ref, displacements_ref = find_neighbors_all_pairs(
            RT, pos_i, pos_j, CUTOFF
        )
        time2 = perf_counter()
        log.info(f"{torch.allclose(index_i, index_i_ref) = }")
        log.info(f"{torch.allclose(index_j, index_j_ref) = }")
        log.info(f"{torch.allclose(displacements, displacements_ref) = }")
        log.info(f"Binned time:    {time1 - time0:.2} s")
        log.info(f"All-pairs time: {time2 - time1:.2} s")

    main()

    
