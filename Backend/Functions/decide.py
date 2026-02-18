import numpy as np
import hashlib


class AdaptiveLSBCore:
    def __init__(self, max_bits=2, block_rows=128, seed_key="default", exclude_positions=None):
        self.max_bits    = max_bits
        self.blue_bonus  = 5
        self.block_rows  = block_rows
        self.seed_key    = seed_key
        self.last_bit_index = 0

        self._base_seed = hashlib.sha256(seed_key.encode()).digest()

        if exclude_positions is None:
            self._excl = None
        else:
            self._excl = np.asarray(exclude_positions, dtype=np.int64)

    def _zone(self, block):
        r = block[:, :, 0].astype(np.uint16) & 0xFC
        g = block[:, :, 1].astype(np.uint16) & 0xFC
        b = block[:, :, 2].astype(np.uint16) & 0xFC
        z = np.full((block.shape[0], block.shape[1]), 25, dtype=np.uint8)
        z[(g > r) & (g > b)] = 15
        z[(g > 100) & (b > 100)] = 20
        return z

    def _variance_fast(self, ch):
        c  = (ch & 0xFC).astype(np.float32)
        cp = np.pad(c, 1, mode='edge')
        h, w = c.shape
        s  = np.zeros((h, w), dtype=np.float32)
        s2 = np.zeros((h, w), dtype=np.float32)
        for dr in range(3):
            for dc in range(3):
                p   = cp[dr:dr+h, dc:dc+w]
                s  += p
                s2 += p * p
        mean = s * (1.0 / 9.0)
        var  = s2 * (1.0 / 9.0) - mean * mean
        var[var < 0] = 0
        return var

    def _grad_fast(self, ch):
        c  = (ch & 0xFC).astype(np.int16)
        cp = np.pad(c, 1, mode='edge')
        h, w = c.shape
        gx = cp[1:h+1, 2:w+2].astype(np.int32) - cp[1:h+1, 0:w].astype(np.int32)
        gy = cp[2:h+2, 1:w+1].astype(np.int32) - cp[0:h,   1:w+1].astype(np.int32)
        return (gx * gx + gy * gy).astype(np.float32)

    def _score(self, zone, var, grad, channel_index):
        s  = zone.astype(np.int16)
        s += np.where(var < 225,  5,
             np.where(var < 900,  20,
             np.where(var < 2500, 30, 25))).astype(np.int16)
        s += np.where(grad < 100,  5,
             np.where(grad < 900,  15,
             np.where(grad < 3600, 25, 20))).astype(np.int16)
        if channel_index == 2:
            s += self.blue_bonus
        return s.astype(np.uint8)

    def _block_seed_int(self, rs, c):
        raw = hashlib.sha256(
            self._base_seed + rs.to_bytes(4, 'big') + c.to_bytes(1, 'big')
        ).digest()
        return int.from_bytes(raw[:8], 'big')

    def _candidate_indices(self, score, rs, cols, n_ch, c, shuffle=True):
        local_idx = np.flatnonzero((score >= 40).ravel())

        if self._excl is not None and local_idx.size > 0:
            global_idxs = rs * cols * n_ch + c + local_idx * n_ch
            local_idx   = local_idx[~np.isin(global_idxs, self._excl)]

        if local_idx.size == 0:
            return np.empty(0, dtype=np.uint8), np.empty(0, dtype=np.int64)

        if shuffle:
            rng = np.random.default_rng(self._block_seed_int(rs, c))
            rng.shuffle(local_idx)

        score_flat  = score.ravel()
        bits_per    = np.where(score_flat[local_idx] >= 60, 2, 1).astype(np.uint8)
        global_idxs = (rs * cols * n_ch + c + local_idx * n_ch).astype(np.int64)

        return bits_per, global_idxs

    def encode(self, img, bit_array):
        rows, cols, n_ch = img.shape
        flat      = img.reshape(-1)
        bit_array = np.asarray(bit_array, dtype=np.uint8)
        bit_len   = len(bit_array)
        bit_idx   = 0

        for rs in range(0, rows, self.block_rows):
            re    = min(rs + self.block_rows, rows)
            block = img[rs:re]
            zone  = self._zone(block)

            for c in range(n_ch):
                bits_per, global_idxs = self._candidate_indices(
                    self._score(zone,
                                self._variance_fast(block[:, :, c]),
                                self._grad_fast(block[:, :, c]),
                                c),
                    rs, cols, n_ch, c
                )
                if global_idxs.size == 0:
                    continue

                cum    = np.cumsum(bits_per)
                remain = bit_len - bit_idx
                n_use  = int(np.searchsorted(cum, remain, side='right'))
                if n_use == 0:
                    self.last_bit_index = bit_idx
                    return img

                gidx   = global_idxs[:n_use]
                bper   = bits_per[:n_use]
                cum_u  = cum[:n_use]
                starts = (np.concatenate([[0], cum_u[:-1]]) + bit_idx).astype(np.int64)

                m1 = bper == 1
                if m1.any():
                    p = gidx[m1]
                    flat[p] = (flat[p] & np.uint8(0xFE)) | bit_array[starts[m1]]

                m2 = bper == 2
                if m2.any():
                    p  = gidx[m2]
                    s2 = starts[m2]
                    flat[p] = (flat[p] & np.uint8(0xFC)) | (bit_array[s2] << 1) | bit_array[s2 + 1]

                bit_idx += int(cum_u[-1])
                if bit_idx >= bit_len:
                    self.last_bit_index = bit_idx
                    return img

        self.last_bit_index = bit_idx
        return img

    def decode(self, img, payload_bit_length):
        rows, cols, n_ch = img.shape
        flat      = img.reshape(-1)
        extracted = np.zeros(payload_bit_length, dtype=np.uint8)
        bit_idx   = 0

        for rs in range(0, rows, self.block_rows):
            re    = min(rs + self.block_rows, rows)
            block = img[rs:re]
            zone  = self._zone(block)

            for c in range(n_ch):
                bits_per, global_idxs = self._candidate_indices(
                    self._score(zone,
                                self._variance_fast(block[:, :, c]),
                                self._grad_fast(block[:, :, c]),
                                c),
                    rs, cols, n_ch, c
                )
                if global_idxs.size == 0:
                    continue

                cum    = np.cumsum(bits_per)
                remain = payload_bit_length - bit_idx
                n_use  = int(np.searchsorted(cum, remain, side='right'))
                if n_use == 0:
                    return extracted

                gidx   = global_idxs[:n_use]
                bper   = bits_per[:n_use]
                cum_u  = cum[:n_use]
                vals   = flat[gidx]
                starts = (np.concatenate([[0], cum_u[:-1]]) + bit_idx).astype(np.int64)

                m1 = bper == 1
                if m1.any():
                    extracted[starts[m1]] = vals[m1] & np.uint8(1)

                m2 = bper == 2
                if m2.any():
                    s2 = starts[m2]
                    extracted[s2]     = (vals[m2] >> 1) & np.uint8(1)
                    extracted[s2 + 1] = vals[m2] & np.uint8(1)

                bit_idx += int(cum_u[-1])
                if bit_idx >= payload_bit_length:
                    return extracted

        return extracted

    def capacity(self, img):
        rows, cols, n_ch = img.shape
        total = 0
        for rs in range(0, rows, self.block_rows):
            re    = min(rs + self.block_rows, rows)
            block = img[rs:re]
            zone  = self._zone(block)
            for c in range(n_ch):
                bits_per, global_idxs = self._candidate_indices(
                    self._score(zone,
                                self._variance_fast(block[:, :, c]),
                                self._grad_fast(block[:, :, c]),
                                c),
                    rs, cols, n_ch, c, shuffle=False
                )
                total += int(bits_per.sum()) if bits_per.size > 0 else 0
        return total