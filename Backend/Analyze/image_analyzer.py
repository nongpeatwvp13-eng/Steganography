import math
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import logging
from skimage.metrics import structural_similarity as ssim

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(funcName)s: %(message)s')
logger = logging.getLogger(__name__)

FAST_DIM = 768
SSIM_DIM = 512
BIT_PLANE_DIM = 512
LSB_DIM = 512

BLOCK_SIZE = 512


def load_image_safe(
    path: str,
    max_dim: Optional[int] = None,
    as_float: bool = False,
    grayscale: bool = False
) -> Optional[np.ndarray]:
    try:
        flags = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR

        if max_dim and max_dim <= 1024:
            flags = cv2.IMREAD_REDUCED_COLOR_4 if not grayscale else cv2.IMREAD_GRAYSCALE
        elif max_dim and max_dim <= 2048:
            flags = cv2.IMREAD_REDUCED_COLOR_2 if not grayscale else cv2.IMREAD_GRAYSCALE

        img = cv2.imread(str(path), flags)
        if img is None:
            logger.error(f"Failed to load image: {path}")
            return None

        if not grayscale and len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if max_dim and max(img.shape[:2]) > max_dim:
            h, w = img.shape[:2]
            scale = max_dim / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        if as_float:
            img = img.astype(np.float32) / 255.0

        return img

    except Exception as e:
        logger.error(f"Error loading image {path}: {e}")
        return None


def _iter_blocks(arr1: np.ndarray, arr2: np.ndarray):
    for r in range(0, arr1.shape[0], BLOCK_SIZE):
        re = min(r + BLOCK_SIZE, arr1.shape[0])
        for c in range(0, arr1.shape[1], BLOCK_SIZE):
            ce = min(c + BLOCK_SIZE, arr1.shape[1])
            yield arr1[r:re, c:ce], arr2[r:re, c:ce]


def _iter_blocks_region(arr1: np.ndarray, arr2: np.ndarray,
                        r0: int, r1: int, c0: int, c1: int):
    for r in range(r0, r1, BLOCK_SIZE):
        re = min(r + BLOCK_SIZE, r1)
        for c in range(c0, c1, BLOCK_SIZE):
            ce = min(c + BLOCK_SIZE, c1)
            yield arr1[r:re, c:ce], arr2[r:re, c:ce]


def _diff_i16(b1: np.ndarray, b2: np.ndarray) -> np.ndarray:
    return b1.astype(np.int16) - b2.astype(np.int16)


def _chunked_mse_u8(arr1: np.ndarray, arr2: np.ndarray) -> float:
    acc   = np.int64(0)
    count = 0
    for b1, b2 in _iter_blocks(arr1, arr2):
        d    = _diff_i16(b1, b2)
        acc += np.sum(d * d, dtype=np.int64)
        count += d.size
    return float(acc) / count if count > 0 else 0.0


def calculate_psnr(arr1: np.ndarray, arr2: np.ndarray) -> float:
    try:
        mse = _chunked_mse_u8(arr1, arr2)
        if mse < 1e-10:
            return 100.0
        return float(10.0 * math.log10(255.0 * 255.0 / mse))
    except Exception as e:
        logger.error(f"PSNR calculation error: {e}")
        return 0.0


def calculate_mse(arr1: np.ndarray, arr2: np.ndarray) -> float:
    try:
        return _chunked_mse_u8(arr1, arr2)
    except Exception as e:
        logger.error(f"MSE calculation error: {e}")
        return 0.0


def calculate_ssim(arr1: np.ndarray, arr2: np.ndarray) -> float:
    try:
        f1 = arr1.astype(np.float32) / 255.0
        f2 = arr2.astype(np.float32) / 255.0
        if len(f1.shape) == 3:
            val = ssim(f1, f2, channel_axis=2, data_range=1.0)
        else:
            val = ssim(f1, f2, data_range=1.0)
        return float(val)
    except Exception as e:
        logger.error(f"SSIM calculation error: {e}")
        return 0.0


def _uniformity_from_means(block_means: list) -> float:
    if not block_means:
        return 1.0
    mean = sum(block_means) / len(block_means)
    if mean == 0:
        return 1.0
    variance = sum((x - mean) ** 2 for x in block_means) / len(block_means)
    cv = math.sqrt(variance) / (mean + 1e-10)
    return 1.0 / (1.0 + cv)


def analyze_pixel_differences(arr1: np.ndarray, arr2: np.ndarray) -> Dict[str, Any]:
    try:
        sum_diff = np.int64(0)
        sum_sq   = np.int64(0)
        max_diff = 0
        changed  = 0
        count    = 0
        block_medians = []

        for b1, b2 in _iter_blocks(arr1, arr2):
            d  = _diff_i16(b1, b2)
            ad = np.abs(d)
            sum_diff += np.sum(ad, dtype=np.int64)
            sum_sq   += np.sum(d * d, dtype=np.int64)
            bmax = int(ad.max())
            if bmax > max_diff:
                max_diff = bmax
            changed += int(np.sum(ad > 0))
            block_medians.append(float(np.median(ad)))
            count += d.size

        mean_diff   = float(sum_diff) / count if count > 0 else 0.0
        variance    = float(sum_sq) / count - mean_diff ** 2
        std_diff    = math.sqrt(max(variance, 0.0))
        median_diff = float(np.median(block_medians)) if block_medians else 0.0

        return {
            'mean_diff':            mean_diff / 255.0,
            'max_diff':             max_diff  / 255.0,
            'std_diff':             std_diff  / 255.0,
            'median_diff':          median_diff / 255.0,
            'changed_pixels_ratio': float(changed) / count if count > 0 else 0.0
        }
    except Exception as e:
        logger.error(f"Pixel difference analysis error: {e}")
        return {'mean_diff': 0.0, 'max_diff': 0.0, 'std_diff': 0.0,
                'median_diff': 0.0, 'changed_pixels_ratio': 0.0}


def _region_stats(arr1: np.ndarray, arr2: np.ndarray,
                  r0: int, r1: int, c0: int, c1: int) -> Tuple[float, float]:
    sum_d       = np.int64(0)
    count       = 0
    block_means = []

    for b1, b2 in _iter_blocks_region(arr1, arr2, r0, r1, c0, c1):
        ad       = np.abs(_diff_i16(b1, b2))
        bsum     = int(np.sum(ad, dtype=np.int64))
        sum_d   += bsum
        count   += ad.size
        block_means.append(bsum / ad.size)

    region_mean = (float(sum_d) / count / 255.0) if count > 0 else 0.0
    uniformity  = _uniformity_from_means(block_means)
    return region_mean, uniformity


def analyze_spatial_distribution(arr1: np.ndarray, arr2: np.ndarray) -> Dict[str, Any]:
    try:
        h, w  = arr1.shape[:2]
        mid_h = h // 2
        mid_w = w // 2
        q_h   = h // 4
        q_w   = w // 4

        regions = {
            'top':    (0,     mid_h, 0,     w),
            'bottom': (mid_h, h,     0,     w),
            'left':   (0,     h,     0,     mid_w),
            'right':  (0,     h,     mid_w, w),
            'center': (q_h,   3*q_h, q_w,   3*q_w),
            'global': (0,     h,     0,     w),
        }

        out = {name: _region_stats(arr1, arr2, *coords)
               for name, coords in regions.items()}

        edge_parts = [
            np.abs(_diff_i16(arr1[0, :],  arr2[0, :])).ravel(),
            np.abs(_diff_i16(arr1[-1, :], arr2[-1, :])).ravel(),
            np.abs(_diff_i16(arr1[:, 0],  arr2[:, 0])).ravel(),
            np.abs(_diff_i16(arr1[:, -1], arr2[:, -1])).ravel(),
        ]
        edge_mean = float(np.mean(np.concatenate(edge_parts))) / 255.0

        return {
            'top_mean':          out['top'][0],
            'bottom_mean':       out['bottom'][0],
            'left_mean':         out['left'][0],
            'right_mean':        out['right'][0],
            'center_mean':       out['center'][0],
            'edge_mean':         edge_mean,
            'top_uniformity':    out['top'][1],
            'bottom_uniformity': out['bottom'][1],
            'left_uniformity':   out['left'][1],
            'right_uniformity':  out['right'][1],
            'center_uniformity': out['center'][1],
            'global_uniformity': out['global'][1],
        }
    except Exception as e:
        logger.error(f"Spatial distribution analysis error: {e}")
        return {
            'top_mean': 0.0, 'bottom_mean': 0.0, 'left_mean': 0.0,
            'right_mean': 0.0, 'center_mean': 0.0, 'edge_mean': 0.0,
            'top_uniformity': 1.0, 'bottom_uniformity': 1.0,
            'left_uniformity': 1.0, 'right_uniformity': 1.0,
            'center_uniformity': 1.0, 'global_uniformity': 1.0
        }


def analyze_lsb_changes(arr1: np.ndarray, arr2: np.ndarray) -> Dict[str, Any]:
    try:
        lsb_changes = np.int64(0)
        total       = 0
        one  = np.uint8(1)

        for b1, b2 in _iter_blocks(arr1, arr2):
            lsb_changes += np.sum(np.bitwise_xor(b1 & one, b2 & one), dtype=np.int64)
            total += b1.size

        ratio = float(lsb_changes) / total if total > 0 else 0.0

        return {
            'lsb_change_ratio':  ratio,
            'lsb_changes_count': int(lsb_changes),
            'bit_plane_changes': [ratio]
        }
    except Exception as e:
        logger.error(f"LSB analysis error: {e}")
        return {'lsb_change_ratio': 0.0, 'lsb_changes_count': 0, 'bit_plane_changes': [0.0]}


def analyze_histogram_changes(arr1: np.ndarray, arr2: np.ndarray) -> Dict[str, Any]:
    try:
        if len(arr1.shape) == 3:
            n_ch = arr1.shape[2]
            chs1 = [arr1[:, :, i] for i in range(n_ch)]
            chs2 = [arr2[:, :, i] for i in range(n_ch)]
        else:
            chs1 = [arr1]
            chs2 = [arr2]

        channel_diffs = []
        eo_per        = []
        es_per        = []

        for c1, c2 in zip(chs1, chs2):
            h1 = np.bincount(c1.ravel(), minlength=256).astype(np.float32)
            h2 = np.bincount(c2.ravel(), minlength=256).astype(np.float32)
            h1 /= h1.sum() + 1e-10
            h2 /= h2.sum() + 1e-10
            channel_diffs.append(float(np.sum(np.abs(h1 - h2))))

            hh1 = h1[h1 > 0]
            hh2 = h2[h2 > 0]
            eo_per.append(float(-np.sum(hh1 * np.log2(hh1))))
            es_per.append(float(-np.sum(hh2 * np.log2(hh2))))

        return {
            'histogram_difference':         float(np.mean(channel_diffs)),
            'channel_differences':          channel_diffs,
            'entropy_original':             float(np.mean(eo_per)),
            'entropy_stego':                float(np.mean(es_per)),
            'entropy_original_per_channel': eo_per,
            'entropy_stego_per_channel':    es_per,
        }
    except Exception as e:
        logger.error(f"Histogram analysis error: {e}")
        return {
            'histogram_difference': 0.0, 'channel_differences': [0.0],
            'entropy_original': 0.0, 'entropy_stego': 0.0,
            'entropy_original_per_channel': [0.0], 'entropy_stego_per_channel': [0.0],
        }


def fast_correlation(arr1: np.ndarray, arr2: np.ndarray) -> float:
    try:
        sum_a = np.int64(0)
        sum_b = np.int64(0)
        count = 0

        for b1, b2 in _iter_blocks(arr1, arr2):
            sum_a += np.sum(b1, dtype=np.int64)
            sum_b += np.sum(b2, dtype=np.int64)
            count += b1.size

        mean_a = float(sum_a) / count
        mean_b = float(sum_b) / count

        num   = np.float64(0.0)
        den_a = np.float64(0.0)
        den_b = np.float64(0.0)

        for b1, b2 in _iter_blocks(arr1, arr2):
            a  = b1.astype(np.float64).ravel() - mean_a
            b  = b2.astype(np.float64).ravel() - mean_b
            num   += np.sum(a * b)
            den_a += np.sum(a * a)
            den_b += np.sum(b * b)

        return float(num / (math.sqrt(den_a) * math.sqrt(den_b) + 1e-10))
    except Exception as e:
        logger.error(f"Correlation calculation error: {e}")
        return 0.0


def analyze_correlation(arr1: np.ndarray, arr2: np.ndarray) -> Dict[str, Any]:
    try:
        if len(arr1.shape) == 3:
            correlations = [fast_correlation(arr1[:, :, i], arr2[:, :, i])
                            for i in range(arr1.shape[2])]
            overall_corr = float(np.mean(correlations))
        else:
            overall_corr = fast_correlation(arr1, arr2)
            correlations = [overall_corr]

        return {'overall_correlation': overall_corr, 'channel_correlations': correlations}
    except Exception as e:
        logger.error(f"Correlation analysis error: {e}")
        return {'overall_correlation': 0.0, 'channel_correlations': [0.0]}


def analyze_noise_characteristics(arr1: np.ndarray, arr2: np.ndarray) -> Dict[str, Any]:
    try:
        sum_d  = np.int64(0)
        sum_sq = np.int64(0)
        count  = 0

        for b1, b2 in _iter_blocks(arr1, arr2):
            d      = _diff_i16(b1, b2).ravel()
            sum_d += np.sum(d, dtype=np.int64)
            sum_sq += np.sum(d.astype(np.int32) * d.astype(np.int32), dtype=np.int64)
            count += d.size

        mean     = float(sum_d) / count if count > 0 else 0.0
        variance = float(sum_sq) / count - mean ** 2
        std      = math.sqrt(max(variance, 0.0))

        sum_skew = np.float64(0.0)
        sum_kurt = np.float64(0.0)

        for b1, b2 in _iter_blocks(arr1, arr2):
            d  = _diff_i16(b1, b2).ravel().astype(np.float64)
            z  = (d - mean) / (std + 1e-10)
            z2 = z * z
            sum_skew += np.sum(z2 * z)
            sum_kurt += np.sum(z2 * z2)

        skewness     = float(sum_skew) / count if count > 0 else 0.0
        kurtosis_val = float(sum_kurt) / count - 3.0 if count > 0 else 0.0

        return {
            'noise_mean':     mean / 255.0,
            'noise_std':      std  / 255.0,
            'noise_skewness': skewness,
            'noise_kurtosis': kurtosis_val
        }
    except Exception as e:
        logger.error(f"Noise characteristics analysis error: {e}")
        return {'noise_mean': 0.0, 'noise_std': 0.0,
                'noise_skewness': 0.0, 'noise_kurtosis': 0.0}


def _chunked_std_u8(arr: np.ndarray) -> float:
    sum_v  = np.int64(0)
    sum_sq = np.int64(0)
    count  = 0
    for r in range(0, arr.shape[0], BLOCK_SIZE):
        re = min(r + BLOCK_SIZE, arr.shape[0])
        for c in range(0, arr.shape[1], BLOCK_SIZE):
            ce = min(c + BLOCK_SIZE, arr.shape[1])
            b  = arr[r:re, c:ce].ravel().astype(np.int32)
            sum_v  += np.sum(b, dtype=np.int64)
            sum_sq += np.sum(b * b, dtype=np.int64)
            count  += b.size
    mean = float(sum_v) / count if count > 0 else 0.0
    var  = float(sum_sq) / count - mean ** 2
    return math.sqrt(max(var, 0.0)) / 255.0


def comprehensive_analysis(original_path: str, stego_path: str) -> Dict[str, Any]:
    try:
        orig  = load_image_safe(original_path, as_float=False, grayscale=False)
        stego = load_image_safe(stego_path,    as_float=False, grayscale=False)

        if orig is None or stego is None:
            logger.error(f"Image loading failed - orig: {orig is None}, stego: {stego is None}")
            return {"error": "Failed to load images"}

        logger.info(f"Original shape: {orig.shape}")
        logger.info(f"Stego shape: {stego.shape}")

        if orig.shape != stego.shape:
            logger.warning("Shape mismatch! Resizing stego to match original")
            stego = cv2.resize(stego, (orig.shape[1], orig.shape[0]),
                               interpolation=cv2.INTER_AREA)

        if len(orig.shape) == 3:
            orig_gray  = cv2.cvtColor(orig,  cv2.COLOR_RGB2GRAY)
            stego_gray = cv2.cvtColor(stego, cv2.COLOR_RGB2GRAY)
        else:
            orig_gray  = orig
            stego_gray = stego

        h, w = orig.shape[:2]
        ssim_scale = SSIM_DIM / max(h, w)
        ssim_h, ssim_w = int(h * ssim_scale), int(w * ssim_scale)
        orig_ssim  = cv2.resize(orig,  (ssim_w, ssim_h), interpolation=cv2.INTER_AREA)
        stego_ssim = cv2.resize(stego, (ssim_w, ssim_h), interpolation=cv2.INTER_AREA)

        pixel_diff   = analyze_pixel_differences(orig, stego)
        spatial_dist = analyze_spatial_distribution(orig, stego)
        lsb_data     = analyze_lsb_changes(orig, stego)
        hist_data    = analyze_histogram_changes(orig_gray, stego_gray)
        corr_data    = analyze_correlation(orig, stego)
        noise_data   = analyze_noise_characteristics(orig, stego)

        total_pixels   = orig.size
        changed_pixels = int(pixel_diff['changed_pixels_ratio'] * total_pixels)
        n_ch           = orig.shape[2] if len(orig.shape) == 3 else 1

        eo = hist_data['entropy_original_per_channel']
        es = hist_data['entropy_stego_per_channel']

        def _entropy_block(i):
            eo_v = eo[i] if i < len(eo) else hist_data['entropy_original']
            es_v = es[i] if i < len(es) else hist_data['entropy_stego']
            return {
                'original_entropy':   eo_v,
                'stego_entropy':      es_v,
                'entropy_difference': es_v - eo_v,
                'entropy_increase':   ((es_v - eo_v) / eo_v * 100) if eo_v > 0 else 0.0
            }

        orig_std = _chunked_std_u8(orig)
        snr = (20.0 * math.log10(orig_std / (noise_data['noise_std'] + 1e-10))
               if noise_data['noise_std'] > 0 else 100.0)

        results = {
            'psnr': calculate_psnr(orig, stego),
            'mse':  calculate_mse(orig, stego),
            'ssim': calculate_ssim(orig_ssim, stego_ssim),

            'pixel_differences': {
                'overall': {
                    'percent_changed': pixel_diff['changed_pixels_ratio'] * 100,
                    'max_difference':  pixel_diff['max_diff'],
                    'mean_difference': pixel_diff['mean_diff'],
                    'std_difference':  pixel_diff['std_diff']
                }
            },

            'spatial_distribution': {
                'global': {
                    'percent_changed':  pixel_diff['changed_pixels_ratio'] * 100,
                    'changed_pixels':   changed_pixels,
                    'uniformity_score': spatial_dist['global_uniformity']
                },
                'top_region': {
                    'percent_changed':  spatial_dist['top_mean'] * 100,
                    'changed_pixels':   int(spatial_dist['top_mean'] * total_pixels / 2),
                    'uniformity_score': spatial_dist['top_uniformity']
                },
                'bottom_region': {
                    'percent_changed':  spatial_dist['bottom_mean'] * 100,
                    'changed_pixels':   int(spatial_dist['bottom_mean'] * total_pixels / 2),
                    'uniformity_score': spatial_dist['bottom_uniformity']
                },
                'left_region': {
                    'percent_changed':  spatial_dist['left_mean'] * 100,
                    'changed_pixels':   int(spatial_dist['left_mean'] * total_pixels / 2),
                    'uniformity_score': spatial_dist['left_uniformity']
                },
                'right_region': {
                    'percent_changed':  spatial_dist['right_mean'] * 100,
                    'changed_pixels':   int(spatial_dist['right_mean'] * total_pixels / 2),
                    'uniformity_score': spatial_dist['right_uniformity']
                },
                'center_region': {
                    'percent_changed':  spatial_dist['center_mean'] * 100,
                    'changed_pixels':   int(spatial_dist['center_mean'] * total_pixels / 4),
                    'uniformity_score': spatial_dist['center_uniformity']
                }
            },

            'histogram_statistics': {
                'red': {
                    'chi_square':       0.0,
                    'correlation':      corr_data['channel_correlations'][0] if len(corr_data['channel_correlations']) > 0 else 1.0,
                    'kl_divergence':    hist_data['channel_differences'][0] if len(hist_data['channel_differences']) > 0 else 0.0,
                    'entropy_original': eo[0] if len(eo) > 0 else hist_data['entropy_original'],
                    'entropy_stego':    es[0] if len(es) > 0 else hist_data['entropy_stego']
                },
                'green': {
                    'chi_square':       0.0,
                    'correlation':      corr_data['channel_correlations'][1] if len(corr_data['channel_correlations']) > 1 else 1.0,
                    'kl_divergence':    hist_data['channel_differences'][1] if len(hist_data['channel_differences']) > 1 else 0.0,
                    'entropy_original': eo[1] if len(eo) > 1 else hist_data['entropy_original'],
                    'entropy_stego':    es[1] if len(es) > 1 else hist_data['entropy_stego']
                },
                'blue': {
                    'chi_square':       0.0,
                    'correlation':      corr_data['channel_correlations'][2] if len(corr_data['channel_correlations']) > 2 else 1.0,
                    'kl_divergence':    hist_data['channel_differences'][2] if len(hist_data['channel_differences']) > 2 else 0.0,
                    'entropy_original': eo[2] if len(eo) > 2 else hist_data['entropy_original'],
                    'entropy_stego':    es[2] if len(es) > 2 else hist_data['entropy_stego']
                }
            },

            'entropy_analysis': {
                'red':   _entropy_block(0),
                'green': _entropy_block(1),
                'blue':  _entropy_block(2),
            },

            'lsb_analysis': {
                'red': {
                    'changes':          lsb_data['lsb_changes_count'] // 3 if n_ch == 3 else lsb_data['lsb_changes_count'],
                    'percent_changed':  lsb_data['lsb_change_ratio'] * 100,
                    'entropy':          lsb_data['bit_plane_changes'][0],
                    'randomness_score': lsb_data['bit_plane_changes'][0],
                    'ones_ratio':       0.5
                },
                'green': {
                    'changes':          lsb_data['lsb_changes_count'] // 3 if n_ch == 3 else lsb_data['lsb_changes_count'],
                    'percent_changed':  lsb_data['lsb_change_ratio'] * 100,
                    'entropy':          lsb_data['bit_plane_changes'][0],
                    'randomness_score': lsb_data['bit_plane_changes'][0],
                    'ones_ratio':       0.5
                },
                'blue': {
                    'changes':          lsb_data['lsb_changes_count'] // 3 if n_ch == 3 else lsb_data['lsb_changes_count'],
                    'percent_changed':  lsb_data['lsb_change_ratio'] * 100,
                    'entropy':          lsb_data['bit_plane_changes'][0],
                    'randomness_score': lsb_data['bit_plane_changes'][0],
                    'ones_ratio':       0.5
                }
            },

            'noise_analysis': {
                'snr':      snr,
                'std_noise': noise_data['noise_std']
            },

            'correlation_analysis': corr_data
        }

        return results

    except Exception as e:
        logger.error(f"Comprehensive analysis error: {e}")
        return {"error": str(e)}


def calculate_quality_score(results: Dict[str, Any]) -> float:
    try:
        weights = {
            'psnr':                 0.25,
            'ssim':                 0.25,
            'lsb_change_ratio':     0.15,
            'histogram_difference': 0.15,
            'overall_correlation':  0.10,
            'changed_pixels_ratio': 0.10
        }

        psnr         = results.get('psnr', 0)
        ssim_val     = results.get('ssim', 0)
        lsb_ratio    = results.get('lsb_analysis', {}).get('red', {}).get('percent_changed', 0) / 100
        hist_diff    = results.get('histogram_statistics', {}).get('red', {}).get('kl_divergence', 0)
        correlation  = results.get('correlation_analysis', {}).get('overall_correlation', 0)
        pixel_change = results.get('pixel_differences', {}).get('overall', {}).get('percent_changed', 0) / 100

        psnr_score  = min(psnr / 50.0, 1.0)
        ssim_score  = ssim_val
        lsb_score   = 1.0 - min(lsb_ratio * 10, 1.0)
        hist_score  = 1.0 - min(hist_diff / 2.0, 1.0)
        corr_score  = correlation
        pixel_score = 1.0 - min(pixel_change * 10, 1.0)

        quality_score = (
            weights['psnr']                 * psnr_score  +
            weights['ssim']                 * ssim_score  +
            weights['lsb_change_ratio']     * lsb_score   +
            weights['histogram_difference'] * hist_score  +
            weights['overall_correlation']  * corr_score  +
            weights['changed_pixels_ratio'] * pixel_score
        )

        return float(quality_score * 100)

    except Exception as e:
        logger.error(f"Quality score calculation error: {e}")
        return 0.0


def generate_analysis_report(results: Dict[str, Any]) -> str:
    if "error" in results:
        return f"Analysis Error: {results['error']}"

    report = []
    report.append("=" * 60)
    report.append("STEGANOGRAPHY QUALITY ANALYSIS REPORT")
    report.append("=" * 60)

    report.append("\nBASIC METRICS:")
    report.append(f"  PSNR: {results.get('psnr', 0):.2f} dB")
    report.append(f"  MSE: {results.get('mse', 0):.6f}")
    report.append(f"  SSIM: {results.get('ssim', 0):.4f}")

    report.append("\nPIXEL ANALYSIS:")
    pa = results.get('pixel_differences', {}).get('overall', {})
    report.append(f"  Mean Difference: {pa.get('mean_difference', 0):.6f}")
    report.append(f"  Max Difference: {pa.get('max_difference', 0):.6f}")
    report.append(f"  Std Difference: {pa.get('std_difference', 0):.6f}")
    report.append(f"  Changed Pixels Ratio: {pa.get('percent_changed', 0):.2f}%")

    report.append("\nLSB ANALYSIS:")
    lsb = results.get('lsb_analysis', {}).get('red', {})
    report.append(f"  LSB Change Ratio: {lsb.get('percent_changed', 0):.2f}%")
    report.append(f"  Total LSB Changes: {lsb.get('changes', 0):,}")

    report.append("\nHISTOGRAM ANALYSIS:")
    ha = results.get('histogram_statistics', {}).get('red', {})
    report.append(f"  KL Divergence: {ha.get('kl_divergence', 0):.4f}")
    report.append(f"  Original Entropy: {ha.get('entropy_original', 0):.4f}")
    report.append(f"  Stego Entropy: {ha.get('entropy_stego', 0):.4f}")

    report.append("\nCORRELATION ANALYSIS:")
    ca = results.get('correlation_analysis', {})
    report.append(f"  Overall Correlation: {ca.get('overall_correlation', 0):.6f}")

    report.append("\nNOISE CHARACTERISTICS:")
    na = results.get('noise_analysis', {})
    report.append(f"  SNR: {na.get('snr', 0):.2f} dB")
    report.append(f"  Noise Std: {na.get('std_noise', 0):.6f}")

    quality_score = calculate_quality_score(results)
    report.append("\n" + "=" * 60)
    report.append(f"OVERALL QUALITY SCORE: {quality_score:.2f}/100")
    report.append("=" * 60)

    return "\n".join(report)