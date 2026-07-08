import torch


class LatticeQuantization:
    """Finite-codebook lattice quantizer used by the compression mechanisms.

    The input vector is divided into blocks of size ``lattice_dim``. Each block is
    scaled into the unit ball, quantized to the nearest normalized lattice
    codeword, and then rescaled back to the original range.
    """

    def __init__(self, args, gen_mat, our_round=True):
        self.args = args
        self.our_round = our_round
        self.dim = args.lattice_dim
        self.gen_mat = gen_mat
        self.num_codewords = int(getattr(args, "num_codewords", round(2 ** (args.lattice_dim * args.R))))
        self.overloading = args.num_overloading
        self.lattice_grid_radius = int(getattr(args, "lattice_grid_radius", 100))
        self.verbose = bool(getattr(args, "quantizer_verbose", False))
        self._cached_codewords = None
        self._cached_codewords_key = None
        self.last_num_removed = 0

    @staticmethod
    def divide_into_blocks(input_tensor, dim=2):
        flat = input_tensor.reshape(-1)
        pad_with = (dim - len(flat) % dim) % dim

        if pad_with > 0:
            flat = torch.cat((
                flat,
                torch.zeros(pad_with, dtype=flat.dtype, device=flat.device),
            ))

        return flat.view(dim, -1), pad_with

    @staticmethod
    def combine_blocks(blocks, pad_with, original_shape):
        flat = blocks.reshape(-1)
        if pad_with > 0:
            flat = flat[:-pad_with]
        return flat.view(original_shape)

    def _remove_overloaded_distances(self, distances):
        sorted_distances, _ = torch.sort(distances, descending=True)
        remaining = sorted_distances.clone()
        counter = 0

        if self.overloading == -1:
            variance_threshold = 0.003
            max_counter = 10

            while (
                remaining.numel() > 1
                and torch.var(remaining, unbiased=False) > variance_threshold
                and counter < max_counter
            ):
                remaining = remaining[1:]
                counter += 1

        elif self.overloading == 0:
            pass

        else:
            max_counter = int(float(self.overloading) * len(sorted_distances) / 100.0)
            while remaining.numel() > 1 and counter < max_counter:
                remaining = remaining[1:]
                counter += 1

        self.last_num_removed = counter
        if self.verbose:
            print("overload-filtered blocks: {}".format(counter))

        return remaining

    def scale_points_to_fit_circle(self, points, for_grid=True, target_count=23, should_print=False, radius=1):
        del radius  # The support radius is normalized to one in this implementation.

        if not for_grid:
            points = points.T
            target_count = 10**15

        if for_grid and not points.requires_grad:
            points = torch.unique(points, dim=0)

        distances = torch.linalg.norm(points, dim=1)

        if distances.numel() == 0:
            threshold = torch.tensor(1.0, dtype=points.dtype, device=points.device)
            return points, threshold

        distances_for_scale = self._remove_overloaded_distances(distances) if not for_grid else distances

        if target_count < len(distances_for_scale):
            threshold = distances_for_scale.topk(target_count, largest=False).values[-1]
        else:
            threshold = distances_for_scale.max()

        threshold = torch.clamp(threshold, min=1e-12)

        if should_print or self.verbose:
            print("scaling threshold: {}".format(threshold))

        return points / threshold, threshold

    def _make_integer_grid(self, device):
        d = int(self.args.lattice_dim)
        r = int(self.lattice_grid_radius)

        # High-dimensional grids grow exponentially. The default radius is kept
        # small for dimensions such as E8.
        if d >= 8 and r > 2:
            print(
                "Warning: lattice_grid_radius={} is too large for dimension {}. "
                "Using radius=2 instead.".format(r, d)
            )
            r = 2

        ranges = [torch.arange(-r, r + 1, device=device) for _ in range(d)]

        try:
            mesh = torch.meshgrid(*ranges, indexing="ij")
        except TypeError:
            mesh = torch.meshgrid(*ranges)

        return torch.stack(mesh, dim=-1).reshape(-1, d)

    def _codeword_cache_key(self):
        if getattr(self.gen_mat, "requires_grad", False):
            return None

        return (
            str(self.gen_mat.device),
            str(self.gen_mat.dtype),
            self.args.lattice_dim,
            self.num_codewords,
            self.lattice_grid_radius,
            float(self.gen_mat.detach().sum().cpu()),
        )

    def _get_codewords(self):
        cache_key = self._codeword_cache_key()

        if cache_key is not None and self._cached_codewords_key == cache_key:
            return self._cached_codewords

        device = self.gen_mat.device
        grid = self._make_integer_grid(device)

        # Codewords are generated from integer coefficients. The convention used
        # here is row-coefficients times G^T.
        transformed_points = torch.matmul(grid.float(), self.gen_mat.T)

        codewords, _ = self.scale_points_to_fit_circle(
            transformed_points,
            for_grid=True,
            target_count=self.num_codewords,
        )

        distances = torch.linalg.norm(codewords, dim=1)
        codewords = codewords[distances <= 1]

        if cache_key is not None:
            self._cached_codewords_key = cache_key
            self._cached_codewords = codewords

        return codewords

    def __call__(self, input, shouldPrint=False, shouldReturnBack=False, gettingAlph=None):
        original_shape = input.shape
        vec, pad_with = self.divide_into_blocks(input, self.args.lattice_dim)

        dither = 0

        if gettingAlph is not None:
            scaling_factor_vec = gettingAlph
            scaled_points_vec = (vec + dither) / scaling_factor_vec
        else:
            scaled_points_vec, scaling_factor_vec = self.scale_points_to_fit_circle(
                vec + dither,
                for_grid=False,
                target_count=10**15,
                should_print=shouldPrint,
            )

        scaled_points_vec = scaled_points_vec.T
        codewords = self._get_codewords().to(scaled_points_vec.device)

        if shouldPrint or self.verbose:
            print("codewords after filtering: {}".format(len(codewords)))

        distances = torch.cdist(
            scaled_points_vec.T.to(torch.float32),
            codewords.to(torch.float32),
        )

        if distances.size(1) == 0:
            return codewords, vec

        assignments = distances.argmin(dim=1)
        reconstructed_points = codewords[assignments].T

        output = ((reconstructed_points * scaling_factor_vec) - dither).to(torch.float32)

        if shouldReturnBack:
            reconstructed_tensor = self.combine_blocks(output, pad_with, original_shape)
            return reconstructed_tensor, input

        return output, vec.to(torch.float32)
