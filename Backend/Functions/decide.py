import numpy as np
from scipy import ndimage as nd
import hashlib

class AdaptiveLSBCore:
    def __init__(self, max_bits=2, block_rows=128, seed_key="default"):
        self.max_bits = max_bits
        self.blue_bonus = 5
        self.block_rows = block_rows

        seed_hash = hashlib.sha256(seed_key.encode()).digest()
        seed_int = int.from_bytes(seed_hash[:8], "big")
        self.rng = np.random.default_rng(seed_int)

    def _zone(self, block):
        r = block[:, :, 0]
        g = block[:, :, 1]
        b = block[:, :, 2]

        z = np.full((block.shape[0], block.shape[1]), 25, dtype=np.uint8)
        z[(g > r) & (g > b)] = 15
        z[(g > 100) & (b > 100)] = 20
        return z

    def _variance(self, ch):
        ch = ch >> self.max_bits
        m = nd.uniform_filter(ch, 3, mode="nearest")
        s = nd.uniform_filter(ch * ch, 3, mode="nearest")
        v = s - m * m
        v[v < 0] = 0
        return v

    def _grad(self, ch):
        ch = ch >> self.max_bits
        gx = nd.sobel(ch, 1, mode="nearest")
        gy = nd.sobel(ch, 0, mode="nearest")
        return gx * gx + gy * gy

    def _score(self, zone, var, grad, channel_index):
        score = zone.copy()

        score += np.where(var < 225, 5,
                 np.where(var < 900, 20,
                 np.where(var < 2500, 30, 25))).astype(np.uint8)

        score += np.where(grad < 100, 5,
                 np.where(grad < 900, 15,
                 np.where(grad < 3600, 25, 20))).astype(np.uint8)

        if channel_index == 2:
            score += self.blue_bonus

        return score

    def encode(self, img, bit_array):
        rows, cols, ch = img.shape
        flat = img.reshape(-1)

        bit_len = len(bit_array)
        bit_idx = 0

        for rs in range(0, rows, self.block_rows):
            re = min(rs + self.block_rows, rows)
            block = img[rs:re]
            zone = self._zone(block)

            for c in range(ch):
                channel = block[:, :, c]

                var = self._variance(channel)
                grad = self._grad(channel)
                score = self._score(zone, var, grad, c)

                mask = score >= 40
                local_idx = np.flatnonzero(mask.ravel())
                if local_idx.size > 0:
                    self.rng.shuffle(local_idx)

                bits_per = np.where(score.ravel()[local_idx] >= 60, 2, 1)
                global_idx = ((rs * cols * ch) + c) + (local_idx * ch)

                for idx, b in zip(global_idx, bits_per):
                    if bit_idx >= bit_len:
                        self.last_bit_index = bit_idx
                        return img

                    remaining = bit_len - bit_idx
                    use = min(b, remaining)

                    if use == 1:
                        flat[idx] &= 0b11111110
                        flat[idx] |= bit_array[bit_idx]
                        bit_idx += 1
                    else:
                        flat[idx] &= 0b11111100
                        flat[idx] |= (bit_array[bit_idx] << 1) | bit_array[bit_idx + 1]
                        bit_idx += 2

                del var
                del grad
                del score
                del mask

            del block
            del zone

        self.last_bit_index = bit_idx
        return img

    def decode(self, img, payload_bit_length):
        rows, cols, ch = img.shape
        flat = img.reshape(-1)

        extracted = np.zeros(payload_bit_length, dtype=np.uint8)
        bit_idx = 0

        for rs in range(0, rows, self.block_rows):
            re = min(rs + self.block_rows, rows)
            block = img[rs:re]
            zone = self._zone(block)

            for c in range(ch):
                channel = block[:, :, c]

                var = self._variance(channel)
                grad = self._grad(channel)
                score = self._score(zone, var, grad, c)

                mask = score >= 40
                local_idx = np.flatnonzero(mask.ravel())
                if local_idx.size == 0:
                    continue
                if local_idx.size > 0:
                    self.rng.shuffle(local_idx)


                bits_per = np.where(score.ravel()[local_idx] >= 60, 2, 1)
                global_idx = ((rs * cols * ch) + c) + (local_idx * ch)

                for idx, b in zip(global_idx, bits_per):
                    if bit_idx >= payload_bit_length:
                        return extracted

                    val = flat[idx]
                    remaining = payload_bit_length - bit_idx
                    use = min(b, remaining)

                    if use == 1:
                        extracted[bit_idx] = val & 1
                        bit_idx += 1
                    else:
                        extracted[bit_idx] = (val >> 1) & 1
                        extracted[bit_idx + 1] = val & 1
                        bit_idx += 2

                del var
                del grad
                del score
                del mask

            del block
            del zone

        return extracted