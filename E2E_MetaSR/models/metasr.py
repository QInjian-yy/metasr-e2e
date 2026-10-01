"""Meta-RDN-B and reduced-depth variants, with checkpointing and tiled Meta-Upscale."""

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from vendor.official.metardn import MetaRDN


def official_args(scale=32, rgb_range=1):
    return SimpleNamespace(scale=[scale], G0=64, RDNkSize=3,
                           RDNconfig="B", n_colors=3, rgb_range=rgb_range)


class MemoryEfficientMetaRDN(MetaRDN):
    def __init__(self, scale=32, lr_chunk_size=256, checkpoint_rdb=True, rgb_range=1,
                 rdn_blocks=16):
        if isinstance(scale, bool) or not isinstance(scale, int) or scale < 1:
            raise ValueError("This baseline supports positive integer scales only")
        if not isinstance(lr_chunk_size, int) or lr_chunk_size < 1:
            raise ValueError("lr_chunk_size must be a positive integer")
        if isinstance(rdn_blocks, bool) or not isinstance(rdn_blocks, int) or not 1 <= rdn_blocks <= 16:
            raise ValueError("rdn_blocks must be an integer from 1 to 16")
        super().__init__(official_args(scale, rgb_range))
        if rdn_blocks != self.D:
            self.D = rdn_blocks
            self.RDBs = self.RDBs[:rdn_blocks]
            self.GFF[0] = nn.Conv2d(rdn_blocks * 64, 64, 1)
        self.set_scale(0)
        self.lr_chunk_size = lr_chunk_size
        self.checkpoint_rdb = checkpoint_rdb

    def extract_features(self, x):
        # Same operations and residual as official MetaRDN.forward, before P2W.
        x = self.sub_mean(x)
        f__1 = self.SFENet1(x)
        x = self.SFENet2(f__1)
        RDBs_out = []
        for rdb in self.RDBs:
            if self.checkpoint_rdb and torch.is_grad_enabled():
                x = checkpoint(rdb, x, use_reentrant=False)
            else:
                x = rdb(x)
            RDBs_out.append(x)
        x = self.GFF(torch.cat(RDBs_out, 1))
        x += f__1
        return x

    def position_table(self, device):
        # Official input_matrix_wpn: [1/r, y/r-floor(y/r), x/r-floor(x/r)].
        # At integer r the same r*r positions recur at every LR location.
        offsets = torch.arange(self.scale, device=device, dtype=torch.float32) / self.scale
        dy, dx = torch.meshgrid(offsets, offsets, indexing="ij")
        return torch.stack((torch.full_like(dy, 1.0 / self.scale), dy, dx), -1).reshape(-1, 3)

    def weight_table(self, feature):
        positions = self.position_table(feature.device).to(self.P2W.meta_block[0].weight.dtype)
        return self.P2W(positions).reshape(self.scale ** 2, 576, 3)

    def _tiles(self, feature, box):
        """Yield (top,left,tile) for a box, retaining at most lr_chunk_size LR patches.

        One unfold per call. Boundary tiles select only the requested offsets;
        no full-resolution position matrix, repeated feature or weight tensor.
        """
        if feature.ndim != 4 or feature.shape[1] != 64:
            raise ValueError("Expected features [B,64,H,W]")
        top, left, height, width = box
        h, w = feature.shape[-2:]
        r = self.scale
        if (min(top, left) < 0 or min(height, width) < 1
                or top + height > h * r or left + width > w * r):
            raise ValueError("HR crop lies outside the feature's scaled extent")
        patches = F.unfold(feature, kernel_size=3, padding=1)  # [B,576,H*W]
        weights = self.weight_table(feature)  # [r*r,576,3], recomputed every call/update
        bottom, right = top + height, left + width
        for ly in range(top // r, (bottom - 1) // r + 1):
            y0, y1 = max(top, ly * r), min(bottom, (ly + 1) * r)
            dy = torch.arange(y0 - ly * r, y1 - ly * r, device=feature.device)
            x0 = left
            while x0 < right:
                lx = x0 // r
                if x0 % r or right - x0 < r:
                    x1 = min(right, (lx + 1) * r)
                else:
                    x1 = x0 + min(self.lr_chunk_size, (right - x0) // r) * r
                lx_end = (x1 - 1) // r + 1
                dx0 = x0 % r
                dx1 = (x1 - 1) % r + 1
                dx = torch.arange(dx0, dx1, device=feature.device)
                offset_ids = (dy[:, None] * r + dx[None, :]).reshape(-1)
                kernel = weights.index_select(0, offset_ids).permute(1, 0, 2).reshape(576, -1)
                indices = torch.arange(lx, lx_end, device=feature.device) + ly * w
                selected = patches.index_select(2, indices).transpose(1, 2)
                values = torch.matmul(selected, kernel)  # [B,L,dy*dx*3]
                tile = values.reshape(feature.shape[0], lx_end - lx, len(dy), len(dx), 3)
                tile = tile.permute(0, 4, 2, 1, 3).reshape(feature.shape[0], 3, len(dy), -1)
                yield y0, x0, self.add_mean(tile)
                x0 = x1

    def decode_crop(self, feature, box):
        """Differentiable contiguous HR crop; box = (top,left,height,width)."""
        rows, row, row_top = [], [], None
        for top, _, tile in self._tiles(feature, box):
            if row_top is not None and top != row_top:
                rows.append(torch.cat(row, dim=3))
                row = []
            row.append(tile)
            row_top = top
        rows.append(torch.cat(row, dim=3))
        return torch.cat(rows, dim=2)

    def iter_full_sr(self, feature):
        """Inference iterator. Consume each GPU tile immediately; do not collect it.

        The context belongs inside the generator, so even a grad-enabled caller
        cannot accidentally retain a complete 8K decoder graph.
        """
        with torch.no_grad():
            h, w = feature.shape[-2:]
            yield from self._tiles(feature, (0, 0, h * self.scale, w * self.scale))

    def forward(self, lr, box):
        # Full reconstruction is explicitly exposed only through iter_full_sr.
        return self.decode_crop(self.extract_features(lr), box)
