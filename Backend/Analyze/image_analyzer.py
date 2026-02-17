import numpy as np  
import cv2  
from pathlib import Path  
from typing import Dict, Any, Tuple, Optional  
import logging  
from scipy import stats  
from skimage.metrics import structural_similarity as ssim  
  
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(funcName)s: %(message)s')
logger = logging.getLogger(__name__)  
  
FAST_DIM = 768  
SSIM_DIM = 512  
BIT_PLANE_DIM = 512  
LSB_DIM = 512  
  
  
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
  
  
def calculate_psnr(arr1: np.ndarray, arr2: np.ndarray) -> float:  
    try:  
        mse = np.mean((arr1 - arr2) ** 2)  
        if mse < 1e-10:  
            return 100.0  
        max_pixel = 1.0 if arr1.dtype == np.float32 or arr1.dtype == np.float64 else 255.0  
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))  
        return float(psnr)  
    except Exception as e:  
        logger.error(f"PSNR calculation error: {e}")  
        return 0.0  
  
  
def calculate_mse(arr1: np.ndarray, arr2: np.ndarray) -> float:  
    try:  
        mse = np.mean((arr1 - arr2) ** 2)  
        return float(mse)  
    except Exception as e:  
        logger.error(f"MSE calculation error: {e}")  
        return 0.0  
  
  
def calculate_ssim(arr1: np.ndarray, arr2: np.ndarray) -> float:  
    try:  
        if len(arr1.shape) == 3:  
            ssim_value = ssim(arr1, arr2, channel_axis=2, data_range=1.0)  
        else:  
            ssim_value = ssim(arr1, arr2, data_range=1.0)  
        return float(ssim_value)  
    except Exception as e:  
        logger.error(f"SSIM calculation error: {e}")  
        return 0.0  
  
  
def calculate_uniformity_score(diff_region: np.ndarray) -> float:
    try:
        blocks_h = max(1, diff_region.shape[0] // 8)
        blocks_w = max(1, diff_region.shape[1] // 8)
        
        block_means = []
        for i in range(8):
            for j in range(8):
                start_h = i * blocks_h
                end_h = min((i + 1) * blocks_h, diff_region.shape[0])
                start_w = j * blocks_w
                end_w = min((j + 1) * blocks_w, diff_region.shape[1])
                
                block = diff_region[start_h:end_h, start_w:end_w]
                if block.size > 0:
                    block_means.append(np.mean(block))
        
        if len(block_means) == 0:
            return 1.0
        
        variance = np.var(block_means)
        mean = np.mean(block_means)
        
        if mean == 0:
            return 1.0
        
        cv = np.sqrt(variance) / (mean + 1e-10)
        uniformity = 1.0 / (1.0 + cv)
        
        return float(uniformity)
        
    except Exception as e:
        logger.error(f"Uniformity calculation error: {e}")
        return 1.0


def analyze_pixel_differences(arr1: np.ndarray, arr2: np.ndarray) -> Dict[str, Any]:  
    try:  
        diff = np.abs(arr1 - arr2)  
          
        return {  
            'mean_diff': float(np.mean(diff)),  
            'max_diff': float(np.max(diff)),  
            'std_diff': float(np.std(diff)),  
            'median_diff': float(np.median(diff)),  
            'changed_pixels_ratio': float(np.sum(diff > 0.001) / diff.size)  
        }  
    except Exception as e:  
        logger.error(f"Pixel difference analysis error: {e}")  
        return {  
            'mean_diff': 0.0,  
            'max_diff': 0.0,  
            'std_diff': 0.0,  
            'median_diff': 0.0,  
            'changed_pixels_ratio': 0.0  
        }  
  
  
def analyze_spatial_distribution(arr1: np.ndarray, arr2: np.ndarray) -> Dict[str, Any]:  
    try:  
        diff = np.abs(arr1 - arr2)  
          
        if len(diff.shape) == 3:  
            diff_gray = np.mean(diff, axis=2)  
        else:  
            diff_gray = diff  
          
        h, w = diff_gray.shape  
        top_half = diff_gray[:h//2, :]  
        bottom_half = diff_gray[h//2:, :]  
        left_half = diff_gray[:, :w//2]  
        right_half = diff_gray[:, w//2:]
        center = diff_gray[h//4:3*h//4, w//4:3*w//4]
          
        return {  
            'top_mean': float(np.mean(top_half)),  
            'bottom_mean': float(np.mean(bottom_half)),  
            'left_mean': float(np.mean(left_half)),  
            'right_mean': float(np.mean(right_half)),  
            'center_mean': float(np.mean(center)),  
            'edge_mean': float(np.mean(np.concatenate([  
                diff_gray[0, :], diff_gray[-1, :],  
                diff_gray[:, 0], diff_gray[:, -1]  
            ]))),
            'top_uniformity': calculate_uniformity_score(top_half),
            'bottom_uniformity': calculate_uniformity_score(bottom_half),
            'left_uniformity': calculate_uniformity_score(left_half),
            'right_uniformity': calculate_uniformity_score(right_half),
            'center_uniformity': calculate_uniformity_score(center),
            'global_uniformity': calculate_uniformity_score(diff_gray)
        }  
    except Exception as e:  
        logger.error(f"Spatial distribution analysis error: {e}")  
        return {  
            'top_mean': 0.0,  
            'bottom_mean': 0.0,  
            'left_mean': 0.0,  
            'right_mean': 0.0,  
            'center_mean': 0.0,  
            'edge_mean': 0.0,
            'top_uniformity': 1.0,
            'bottom_uniformity': 1.0,
            'left_uniformity': 1.0,
            'right_uniformity': 1.0,
            'center_uniformity': 1.0,
            'global_uniformity': 1.0
        }  
  
  
def analyze_lsb_changes(arr1: np.ndarray, arr2: np.ndarray) -> Dict[str, Any]:  
    try:  
        img1_uint = (arr1 * 255).astype(np.uint8) if arr1.dtype == np.float32 or arr1.dtype == np.float64 else arr1  
        img2_uint = (arr2 * 255).astype(np.uint8) if arr2.dtype == np.float32 or arr2.dtype == np.float64 else arr2  
          
        lsb_changes = np.sum(np.bitwise_xor(img1_uint & 1, img2_uint & 1))  
        total_pixels = img1_uint.size  
        lsb_change_ratio = lsb_changes / total_pixels  
          
        bit_planes_diff = []  
        for bit in range(8):  
            plane1 = (img1_uint >> bit) & 1  
            plane2 = (img2_uint >> bit) & 1  
            diff_ratio = np.sum(plane1 != plane2) / total_pixels  
            bit_planes_diff.append(float(diff_ratio))  
          
        return {  
            'lsb_change_ratio': float(lsb_change_ratio),  
            'lsb_changes_count': int(lsb_changes),  
            'bit_plane_changes': bit_planes_diff  
        }  
    except Exception as e:  
        logger.error(f"LSB analysis error: {e}")  
        return {  
            'lsb_change_ratio': 0.0,  
            'lsb_changes_count': 0,  
            'bit_plane_changes': [0.0] * 8  
        }  
  
  
def fast_entropy(arr: np.ndarray) -> float:  
    try:  
        img_uint = (arr * 255).astype(np.uint8) if arr.dtype == np.float32 or arr.dtype == np.float64 else arr  
        histogram = np.bincount(img_uint.ravel(), minlength=256)  
        histogram = histogram / (histogram.sum() + 1e-10)  
        histogram = histogram[histogram > 0]  
        entropy = -np.sum(histogram * np.log2(histogram))  
        return float(entropy)  
    except Exception as e:  
        logger.error(f"Entropy calculation error: {e}")  
        return 0.0  
  
  
def analyze_histogram_changes(arr1: np.ndarray, arr2: np.ndarray) -> Dict[str, Any]:  
    try:  
        img1_uint = (arr1 * 255).astype(np.uint8) if arr1.dtype == np.float32 or arr1.dtype == np.float64 else arr1  
        img2_uint = (arr2 * 255).astype(np.uint8) if arr2.dtype == np.float32 or arr2.dtype == np.float64 else arr2  
          
        if len(img1_uint.shape) == 3:  
            channels = cv2.split(img1_uint)  
            channels2 = cv2.split(img2_uint)  
        else:  
            channels = [img1_uint]  
            channels2 = [img2_uint]  
          
        channel_diffs = []  
        for c1, c2 in zip(channels, channels2):  
            hist1 = np.bincount(c1.ravel(), minlength=256)  
            hist2 = np.bincount(c2.ravel(), minlength=256)  
            hist1 = hist1 / (hist1.sum() + 1e-10)  
            hist2 = hist2 / (hist2.sum() + 1e-10)  
            diff = np.sum(np.abs(hist1 - hist2))  
            channel_diffs.append(float(diff))  
          
        entropy_orig = fast_entropy(arr1)  
        entropy_stego = fast_entropy(arr2)  
          
        return {  
            'histogram_difference': float(np.mean(channel_diffs)),  
            'channel_differences': channel_diffs,  
            'entropy_original': entropy_orig,  
            'entropy_stego': entropy_stego  
        }  
    except Exception as e:  
        logger.error(f"Histogram analysis error: {e}")  
        return {  
            'histogram_difference': 0.0,  
            'channel_differences': [0.0],  
            'entropy_original': 0.0,  
            'entropy_stego': 0.0  
        }  
  
  
def fast_correlation(arr1: np.ndarray, arr2: np.ndarray) -> float:  
    try:  
        a = arr1.ravel() - arr1.mean()  
        b = arr2.ravel() - arr2.mean()  
        numerator = np.sum(a * b)  
        denominator = np.sqrt(np.sum(a * a)) * np.sqrt(np.sum(b * b)) + 1e-10  
        return float(numerator / denominator)  
    except Exception as e:  
        logger.error(f"Correlation calculation error: {e}")  
        return 0.0  
  
  
def analyze_correlation(arr1: np.ndarray, arr2: np.ndarray) -> Dict[str, Any]:  
    try:  
        if len(arr1.shape) == 3:  
            correlations = []  
            for i in range(arr1.shape[2]):  
                corr = fast_correlation(arr1[:, :, i], arr2[:, :, i])  
                correlations.append(corr)  
            overall_corr = float(np.mean(correlations))  
        else:  
            overall_corr = fast_correlation(arr1, arr2)  
            correlations = [overall_corr]  
          
        return {  
            'overall_correlation': overall_corr,  
            'channel_correlations': correlations  
        }  
    except Exception as e:  
        logger.error(f"Correlation analysis error: {e}")  
        return {  
            'overall_correlation': 0.0,  
            'channel_correlations': [0.0]  
        }  
  
  
def analyze_noise_characteristics(arr1: np.ndarray, arr2: np.ndarray) -> Dict[str, Any]:  
    try:  
        diff = arr2 - arr1  
          
        mean = float(np.mean(diff))  
        std = float(np.std(diff))  
        diff_flat = diff.ravel()  
        skewness = float(stats.skew(diff_flat))  
        kurtosis_val = float(stats.kurtosis(diff_flat))  
          
        return {  
            'noise_mean': mean,  
            'noise_std': std,  
            'noise_skewness': skewness,  
            'noise_kurtosis': kurtosis_val  
        }  
    except Exception as e:  
        logger.error(f"Noise characteristics analysis error: {e}")  
        return {  
            'noise_mean': 0.0,  
            'noise_std': 0.0,  
            'noise_skewness': 0.0,  
            'noise_kurtosis': 0.0  
        }  
  
  
def comprehensive_analysis(original_path: str, stego_path: str) -> Dict[str, Any]:
    try:
        orig_fast = load_image_safe(original_path, FAST_DIM, as_float=True, grayscale=False)  
        stego_fast = load_image_safe(stego_path, FAST_DIM, as_float=True, grayscale=False)  
        
        if orig_fast is None or stego_fast is None:
            logger.error(f"Image loading failed - orig: {orig_fast is None}, stego: {stego_fast is None}")
            return {"error": "Failed to load images"}
        
        logger.info(f"Original shape: {orig_fast.shape}, range: [{orig_fast.min():.3f}, {orig_fast.max():.3f}]")
        logger.info(f"Stego shape: {stego_fast.shape}, range: [{stego_fast.min():.3f}, {stego_fast.max():.3f}]")
        
        if orig_fast.shape != stego_fast.shape:
            logger.warning(f"Shape mismatch! Resizing stego to match original")
            stego_fast = cv2.resize(
                stego_fast, 
                (orig_fast.shape[1], orig_fast.shape[0]),
                interpolation=cv2.INTER_AREA
            )
        
        h, w = orig_fast.shape[:2]
        ssim_scale = SSIM_DIM / max(h, w)
        ssim_h, ssim_w = int(h * ssim_scale), int(w * ssim_scale)
        
        orig_ssim = cv2.resize(orig_fast, (ssim_w, ssim_h), interpolation=cv2.INTER_AREA)
        stego_ssim = cv2.resize(stego_fast, (ssim_w, ssim_h), interpolation=cv2.INTER_AREA)
        
        if len(orig_fast.shape) == 3:
            orig_gray = cv2.cvtColor((orig_fast * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0  
            stego_gray = cv2.cvtColor((stego_fast * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0  
        else:
            orig_gray = orig_fast  
            stego_gray = stego_fast  
        
        orig_uint = (orig_fast * 255).astype(np.uint8)  
        stego_uint = (stego_fast * 255).astype(np.uint8)
        
        pixel_diff = analyze_pixel_differences(orig_fast, stego_fast)
        spatial_dist = analyze_spatial_distribution(orig_fast, stego_fast)
        lsb_data = analyze_lsb_changes(orig_uint, stego_uint)
        hist_data = analyze_histogram_changes(orig_gray, stego_gray)
        corr_data = analyze_correlation(orig_fast, stego_fast)
        noise_data = analyze_noise_characteristics(orig_fast, stego_fast)
        
        total_pixels = orig_fast.size
        changed_pixels = int(pixel_diff['changed_pixels_ratio'] * total_pixels)
        
        results = {
            'psnr': calculate_psnr(orig_fast, stego_fast),
            'mse': calculate_mse(orig_fast, stego_fast),
            'ssim': calculate_ssim(orig_ssim, stego_ssim),
            
            'pixel_differences': {
                'overall': {
                    'percent_changed': pixel_diff['changed_pixels_ratio'] * 100,
                    'max_difference': pixel_diff['max_diff'],
                    'mean_difference': pixel_diff['mean_diff'],
                    'std_difference': pixel_diff['std_diff']
                }
            },
            
            'spatial_distribution': {
                'global': {
                    'percent_changed': pixel_diff['changed_pixels_ratio'] * 100,
                    'changed_pixels': changed_pixels,
                    'uniformity_score': spatial_dist['global_uniformity']
                },
                'top_region': {
                    'percent_changed': spatial_dist['top_mean'] * 100,
                    'changed_pixels': int(spatial_dist['top_mean'] * total_pixels / 2),
                    'uniformity_score': spatial_dist['top_uniformity']
                },
                'bottom_region': {
                    'percent_changed': spatial_dist['bottom_mean'] * 100,
                    'changed_pixels': int(spatial_dist['bottom_mean'] * total_pixels / 2),
                    'uniformity_score': spatial_dist['bottom_uniformity']
                },
                'left_region': {
                    'percent_changed': spatial_dist['left_mean'] * 100,
                    'changed_pixels': int(spatial_dist['left_mean'] * total_pixels / 2),
                    'uniformity_score': spatial_dist['left_uniformity']
                },
                'right_region': {
                    'percent_changed': spatial_dist['right_mean'] * 100,
                    'changed_pixels': int(spatial_dist['right_mean'] * total_pixels / 2),
                    'uniformity_score': spatial_dist['right_uniformity']
                },
                'center_region': {
                    'percent_changed': spatial_dist['center_mean'] * 100,
                    'changed_pixels': int(spatial_dist['center_mean'] * total_pixels / 4),
                    'uniformity_score': spatial_dist['center_uniformity']
                }
            },
            
            'histogram_statistics': {
                'red': {
                    'chi_square': 0.0,
                    'correlation': corr_data['channel_correlations'][0] if len(corr_data['channel_correlations']) > 0 else 1.0,
                    'kl_divergence': hist_data['channel_differences'][0] if len(hist_data['channel_differences']) > 0 else 0.0,
                    'entropy_original': hist_data['entropy_original'],
                    'entropy_stego': hist_data['entropy_stego']
                },
                'green': {
                    'chi_square': 0.0,
                    'correlation': corr_data['channel_correlations'][1] if len(corr_data['channel_correlations']) > 1 else 1.0,
                    'kl_divergence': hist_data['channel_differences'][1] if len(hist_data['channel_differences']) > 1 else 0.0,
                    'entropy_original': hist_data['entropy_original'],
                    'entropy_stego': hist_data['entropy_stego']
                },
                'blue': {
                    'chi_square': 0.0,
                    'correlation': corr_data['channel_correlations'][2] if len(corr_data['channel_correlations']) > 2 else 1.0,
                    'kl_divergence': hist_data['channel_differences'][2] if len(hist_data['channel_differences']) > 2 else 0.0,
                    'entropy_original': hist_data['entropy_original'],
                    'entropy_stego': hist_data['entropy_stego']
                }
            },
            
            'entropy_analysis': {
                'red': {
                    'original_entropy': hist_data['entropy_original'],
                    'stego_entropy': hist_data['entropy_stego'],
                    'entropy_difference': hist_data['entropy_stego'] - hist_data['entropy_original'],
                    'entropy_increase': ((hist_data['entropy_stego'] - hist_data['entropy_original']) / hist_data['entropy_original'] * 100) if hist_data['entropy_original'] > 0 else 0.0
                },
                'green': {
                    'original_entropy': hist_data['entropy_original'],
                    'stego_entropy': hist_data['entropy_stego'],
                    'entropy_difference': hist_data['entropy_stego'] - hist_data['entropy_original'],
                    'entropy_increase': ((hist_data['entropy_stego'] - hist_data['entropy_original']) / hist_data['entropy_original'] * 100) if hist_data['entropy_original'] > 0 else 0.0
                },
                'blue': {
                    'original_entropy': hist_data['entropy_original'],
                    'stego_entropy': hist_data['entropy_stego'],
                    'entropy_difference': hist_data['entropy_stego'] - hist_data['entropy_original'],
                    'entropy_increase': ((hist_data['entropy_stego'] - hist_data['entropy_original']) / hist_data['entropy_original'] * 100) if hist_data['entropy_original'] > 0 else 0.0
                }
            },
            
            'lsb_analysis': {
                'red': {
                    'changes': lsb_data['lsb_changes_count'] // 3 if len(orig_fast.shape) == 3 else lsb_data['lsb_changes_count'],
                    'percent_changed': lsb_data['lsb_change_ratio'] * 100,
                    'entropy': lsb_data['bit_plane_changes'][0] if len(lsb_data['bit_plane_changes']) > 0 else 0.0,
                    'randomness_score': lsb_data['bit_plane_changes'][0] if len(lsb_data['bit_plane_changes']) > 0 else 0.0,
                    'ones_ratio': 0.5
                },
                'green': {
                    'changes': lsb_data['lsb_changes_count'] // 3 if len(orig_fast.shape) == 3 else lsb_data['lsb_changes_count'],
                    'percent_changed': lsb_data['lsb_change_ratio'] * 100,
                    'entropy': lsb_data['bit_plane_changes'][0] if len(lsb_data['bit_plane_changes']) > 0 else 0.0,
                    'randomness_score': lsb_data['bit_plane_changes'][0] if len(lsb_data['bit_plane_changes']) > 0 else 0.0,
                    'ones_ratio': 0.5
                },
                'blue': {
                    'changes': lsb_data['lsb_changes_count'] // 3 if len(orig_fast.shape) == 3 else lsb_data['lsb_changes_count'],
                    'percent_changed': lsb_data['lsb_change_ratio'] * 100,
                    'entropy': lsb_data['bit_plane_changes'][0] if len(lsb_data['bit_plane_changes']) > 0 else 0.0,
                    'randomness_score': lsb_data['bit_plane_changes'][0] if len(lsb_data['bit_plane_changes']) > 0 else 0.0,
                    'ones_ratio': 0.5
                }
            },
            
            'noise_analysis': {
                'snr': 20 * np.log10(np.std(orig_fast) / (noise_data['noise_std'] + 1e-10)) if noise_data['noise_std'] > 0 else 100.0,
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
            'psnr': 0.25,  
            'ssim': 0.25,  
            'lsb_change_ratio': 0.15,  
            'histogram_difference': 0.15,  
            'overall_correlation': 0.10,  
            'changed_pixels_ratio': 0.10  
        }  
          
        psnr = results.get('psnr', 0)
        ssim_val = results.get('ssim', 0)
        lsb_ratio = results.get('lsb_analysis', {}).get('red', {}).get('percent_changed', 0) / 100
        hist_diff = results.get('histogram_statistics', {}).get('red', {}).get('kl_divergence', 0)
        correlation = results.get('correlation_analysis', {}).get('overall_correlation', 0)
        pixel_change = results.get('pixel_differences', {}).get('overall', {}).get('percent_changed', 0) / 100
          
        psnr_score = min(psnr / 50.0, 1.0)  
        ssim_score = ssim_val  
        lsb_score = 1.0 - min(lsb_ratio * 10, 1.0)  
        hist_score = 1.0 - min(hist_diff / 2.0, 1.0)  
        corr_score = correlation  
        pixel_score = 1.0 - min(pixel_change * 10, 1.0)  
          
        quality_score = (  
            weights['psnr'] * psnr_score +  
            weights['ssim'] * ssim_score +  
            weights['lsb_change_ratio'] * lsb_score +  
            weights['histogram_difference'] * hist_score +  
            weights['overall_correlation'] * corr_score +  
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