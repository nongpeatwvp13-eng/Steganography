from PIL import Image
import numpy as np
from typing import Tuple, Dict, Optional
from scipy import ndimage as nd

class AdaptiveLSBDecider:
    def __init__(self, max_bits_per_channel: int = 2):
        self.max_bits = max_bits_per_channel
        
        self.thresholds = {
            'complexity_low': 15,
            'complexity_medium': 30,
            'complexity_high': 50,
            'saturation_low': 0.3,
            'saturation_high': 0.7,
            'value_dark': 0.3,
            'value_bright': 0.7,
            'gradient_low': 10,
            'gradient_medium': 30,
            'gradient_high': 60,
        }
        
        self.blue_channel_bonus = 5
        self._stats_cache = None
    
    def _calculate_complexity_channel(self, channel_data: np.ndarray) -> np.ndarray:
        img_float = channel_data.astype(np.float32, copy=False)

        mean = nd.uniform_filter(img_float, size=3, mode='nearest')
        sq_mean = nd.uniform_filter(img_float * img_float, size=3, mode='nearest')

        variance = sq_mean - mean * mean
        variance[variance < 0] = 0

        std = np.sqrt(variance, dtype=np.float32)

        return std

    
    def _calculate_gradient_channel(self, channel_data: np.ndarray) -> np.ndarray:
        img_float = channel_data.astype(np.float32, copy=False)
        gx = nd.sobel(img_float, axis=1, mode='nearest')
        gy = nd.sobel(img_float, axis=0, mode='nearest')
        gradient = np.sqrt(gx*gx + gy*gy, dtype=np.float32)
        
        return gradient
    
    def _get_color_zone_map(self, img_array: np.ndarray) -> np.ndarray:
        r = img_array[:, :, 0]
        g = img_array[:, :, 1]
        b = img_array[:, :, 2]

        zone_map = np.zeros((r.shape[0], r.shape[1]), dtype=np.uint8)

        green_mask = (g > r) & (g > b)
        zone_map[green_mask] = 1

        cyan_mask = (g > 100) & (b > 100)
        zone_map[cyan_mask] = 2

        other_mask = ~(green_mask | cyan_mask)
        zone_map[other_mask] = 3

        return zone_map

    
    def generate_embedding_map(self, img_array: np.ndarray) -> np.ndarray:
        rows, cols, channels = img_array.shape
        
        print("  - Creating color zone map...")
        zone_map = self._get_color_zone_map(img_array)
        
        print("  - Computing embedding map...")
        embedding_map = np.zeros_like(img_array, dtype=np.uint8)

        total_score = np.zeros((rows, cols), dtype=np.uint8)
        
        for c in range(channels): 
            total_score.fill(0)
            complexity = self._calculate_complexity_channel(img_array[:, :, c])
            gradient = self._calculate_gradient_channel(img_array[:, :, c])

            total_score[zone_map == 0] += 15
            total_score[zone_map == 2] += 20
            total_score[zone_map == 3] += 25

            total_score[complexity < 15] += 5
            total_score[(complexity >= 15) & (complexity < 30)] += 20
            total_score[(complexity >= 30) & (complexity < 50)] += 30
            total_score[complexity >= 50] += 25

            total_score[gradient < 10] += 5
            total_score[(gradient >= 10) & (gradient < 30)] += 15
            total_score[(gradient >= 30) & (gradient < 60)] += 25
            total_score[gradient >= 60] += 20

            if c == 2:
                total_score += self.blue_channel_bonus

            embedding_map[:, :, c][total_score >= 60] = self.max_bits
            embedding_map[:, :, c][(total_score >= 40) & (total_score < 60)] = 1
        
        print(f"Embedding map generated!")
        print(f"   Average bits per channel: {embedding_map.mean():.3f}")
        print(f"   Total capacity: {embedding_map.sum()} bits")
        
        return embedding_map
    
    def get_stats(self) -> Dict:
        if self._stats_cache is None:
            return {'error': 'No stats available'}
        return self._stats_cache

_global_decider = None
_global_embedding_map = None

def initialize_embedding_map(img_array: np.ndarray, max_bits: int = 2) -> np.ndarray:
    global _global_decider, _global_embedding_map
    
    rows, cols, _ = img_array.shape

    if rows > 2160: 
        analysis_limit = 2000
        print(f"High-res Image Detected: Analyzing only first {analysis_limit} rows to save RAM/Time")
        
        _global_embedding_map = np.zeros_like(img_array[:,:,0:3], dtype=np.uint8)
        
        _global_decider = AdaptiveLSBDecider(max_bits_per_channel=max_bits)
        partial_map = _global_decider.generate_embedding_map(img_array[:analysis_limit, :, :])
        
        _global_embedding_map[:analysis_limit, :, :] = partial_map
    else:
        _global_decider = AdaptiveLSBDecider(max_bits_per_channel=max_bits)
        _global_embedding_map = _global_decider.generate_embedding_map(img_array)
    
    return _global_embedding_map

def get_embedding_bits(i: int, j: int, channel: int) -> int:
    global _global_embedding_map
    
    if _global_embedding_map is None:
        raise RuntimeError("Must call initialize_embedding_map() first!")
    
    return int(_global_embedding_map[i, j, channel])

def get_embedding_map_stats() -> Dict:
    global _global_decider
    
    if _global_decider is None:
        return {'error': 'No map initialized'}
    
    return _global_decider.get_stats()

def cleanup():
    global _global_decider, _global_embedding_map
    _global_decider = None
    _global_embedding_map = None

def get_embedding_bits_legacy(img_array: np.ndarray, i: int, j: int, 
    channel: int, max_bits: int = 2) -> int:
    decider = AdaptiveLSBDecider(max_bits_per_channel=max_bits)
    return 1