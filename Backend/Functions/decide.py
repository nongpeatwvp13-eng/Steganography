import numpy as np
from scipy import ndimage as nd

class AdaptiveLSBDecider:
    def __init__(self, max_bits_per_channel: int = 2):
        self.max_bits = max_bits_per_channel
        self.blue_channel_bonus = 5

    def _complexity(self, channel):
        f = channel.astype(np.float32, copy=False)
        mean = nd.uniform_filter(f, size=3, mode="nearest")
        sq_mean = nd.uniform_filter(f * f, size=3, mode="nearest")
        var = sq_mean - mean * mean
        var[var < 0] = 0
        return np.sqrt(var, dtype=np.float32)

    def _gradient(self, channel):
        f = channel.astype(np.float32, copy=False)
        gx = nd.sobel(f, axis=1, mode="nearest")
        gy = nd.sobel(f, axis=0, mode="nearest")
        return np.sqrt(gx * gx + gy * gy, dtype=np.float32)

    def _zone_map(self, img):
        r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        z = np.zeros((r.shape[0], r.shape[1]), dtype=np.uint8)
        green = (g > r) & (g > b)
        z[green] = 1
        cyan = (g > 100) & (b > 100)
        z[cyan] = 2
        other = ~(green | cyan)
        z[other] = 3
        return z

    def generate_embedding_map(self, img):
        rows, cols, ch = img.shape
        z = self._zone_map(img)
        embedding_map = np.zeros((rows, cols, ch), dtype=np.uint8)

        for c in range(ch):
            channel = img[:,:,c]
            comp = self._complexity(channel)
            grad = self._gradient(channel)
            score = np.zeros((rows, cols), dtype=np.uint16)

            score[z == 0] += 15
            score[z == 2] += 20
            score[z == 3] += 25

            score[comp < 15] += 5
            score[(comp >= 15) & (comp < 30)] += 20
            score[(comp >= 30) & (comp < 50)] += 30
            score[comp >= 50] += 25

            score[grad < 10] += 5
            score[(grad >= 10) & (grad < 30)] += 15
            score[(grad >= 30) & (grad < 60)] += 25
            score[grad >= 60] += 20

            if c == 2:
                score += self.blue_channel_bonus

            embedding_map[:,:,c][score >= 60] = self.max_bits
            embedding_map[:,:,c][(score >= 40) & (score < 60)] = 1

        return embedding_map


_global_map = None

def initialize_embedding_map(img_array, max_bits=2):
    global _global_map
    decider = AdaptiveLSBDecider(max_bits)
    _global_map = decider.generate_embedding_map(img_array)
    return _global_map

def get_embedding_bits(i, j, channel):
    global _global_map
    if _global_map is None:
        raise RuntimeError("Map not initialized")
    return int(_global_map[i, j, channel])

def get_embedding_map_stats():
    global _global_map
    if _global_map is None:
        return {"error": "No map initialized"}
    return {
        "total_capacity_bits": int(_global_map.sum()),
        "average_bits_per_channel": float(_global_map.mean())
    }

def cleanup():
    global _global_map
    _global_map = None