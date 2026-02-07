import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import cv2
from scipy import stats as scipy_stats
from concurrent.futures import ThreadPoolExecutor
import io, base64, gc

MAX_IMAGE_DIM = 4000

def to_python(val):
    if isinstance(val, (np.integer, np.int64, np.int32)):
        return int(val)
    if isinstance(val, (np.floating, np.float64, np.float32)):
        return float(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    return val

def calculate_psnr(original_path, stego_path):
    try:
        img1 = Image.open(original_path)
        img2 = Image.open(stego_path)
        
        if max(img1.size) > MAX_IMAGE_DIM or max(img2.size) > MAX_IMAGE_DIM:
            img1.thumbnail((MAX_IMAGE_DIM, MAX_IMAGE_DIM), Image.Resampling.LANCZOS)
            img2.thumbnail((MAX_IMAGE_DIM, MAX_IMAGE_DIM), Image.Resampling.LANCZOS)
        
        arr1 = np.array(img1.convert('RGB'), dtype=np.float32)
        arr2 = np.array(img2.convert('RGB'), dtype=np.float32)
        img1.close()
        img2.close()
        
        mse = float(np.mean((arr1 - arr2) ** 2))
        del arr1, arr2
        gc.collect()
        
        if mse < 1e-10:
            return 100.0
        
        psnr = float(20 * np.log10(255.0 / np.sqrt(mse)))
        return round(psnr, 2)
    except Exception as e:
        print(f"PSNR error: {e}")
        return 0.0

def calculate_mse(original_path, stego_path):
    try:
        img1 = Image.open(original_path)
        img2 = Image.open(stego_path)
        
        if max(img1.size) > MAX_IMAGE_DIM or max(img2.size) > MAX_IMAGE_DIM:
            img1.thumbnail((MAX_IMAGE_DIM, MAX_IMAGE_DIM), Image.Resampling.LANCZOS)
            img2.thumbnail((MAX_IMAGE_DIM, MAX_IMAGE_DIM), Image.Resampling.LANCZOS)
        
        arr1 = np.array(img1.convert('RGB'), dtype=np.float32)
        arr2 = np.array(img2.convert('RGB'), dtype=np.float32)
        img1.close()
        img2.close()
        
        mse = float(np.mean((arr1 - arr2) ** 2))
        del arr1, arr2
        gc.collect()
        return round(mse, 6)
    except Exception as e:
        print(f"MSE error: {e}")
        return 0.0

def get_histogram_data(img_path, sample_rate=None):
    try:
        img = Image.open(img_path)
        width, height = img.size
        total_pixels = width * height
        
        if sample_rate is None:
            if total_pixels > 33554432:
                sample_rate = 128
            elif total_pixels > 16777216:
                sample_rate = 64
            elif total_pixels > 8294400:
                sample_rate = 32
            else:
                sample_rate = 16
        
        arr = np.array(img.convert('RGB'))[::sample_rate, ::sample_rate, :]
        
        result = {
            'red': np.histogram(arr[:, :, 0], bins=256, range=(0, 256))[0].tolist(),
            'green': np.histogram(arr[:, :, 1], bins=256, range=(0, 256))[0].tolist(),
            'blue': np.histogram(arr[:, :, 2], bins=256, range=(0, 256))[0].tolist()
        }
        
        img.close()
        del arr
        gc.collect()
        return result
    except Exception as e:
        print(f"Histogram error: {e}")
        return {'red': [], 'green': [], 'blue': []}

def calculate_histogram_statistics(histogram_data):
    try:
        stats = {}
        for ch, hist in histogram_data.items():
            if not hist or ch not in ['red', 'green', 'blue']:
                continue
            arr = np.array(hist)
            bins = np.arange(256)
            total = np.sum(arr)
            if total == 0:
                continue
            mean = float(np.sum(bins * arr) / total)
            var = float(np.sum(((bins - mean) ** 2) * arr) / total)
            
            stats[ch] = {
                'mean': round(mean, 2),
                'std': round(float(np.sqrt(var)), 2),
                'min': int(np.min(bins[arr > 0])) if np.any(arr > 0) else 0,
                'max': int(np.max(bins[arr > 0])) if np.any(arr > 0) else 255
            }
        return stats
    except Exception as e:
        print(f"Histogram stats error: {e}")
        return {}

def analyze_lsb_planes(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        
        if max(img.size) > 2000:
            img.thumbnail((2000, 2000), Image.Resampling.LANCZOS)
        
        arr = np.array(img)
        img.close()
        
        result = {}
        for i, ch in enumerate(['red', 'green', 'blue']):
            lsb = arr[:, :, i] & 1
            total = int(lsb.size)
            ones = int(np.sum(lsb))
            result[ch] = {
                'zeros': total - ones,
                'ones': ones,
                'ratio_ones': round(float(ones / total), 4),
                'entropy': round(float(calculate_entropy(lsb)), 4)
            }
        del arr
        gc.collect()
        return result
    except Exception as e:
        print(f"LSB analysis error: {e}")
        return {}

def compare_lsb_planes(original_path, stego_path):
    try:
        img1 = Image.open(original_path).convert('RGB')
        img2 = Image.open(stego_path).convert('RGB')
        
        if max(img1.size) > 2000:
            img1.thumbnail((2000, 2000), Image.Resampling.LANCZOS)
            img2.thumbnail((2000, 2000), Image.Resampling.LANCZOS)
        
        arr1 = np.array(img1)
        arr2 = np.array(img2)
        img1.close()
        img2.close()
        
        result = {}
        for i, ch in enumerate(['red', 'green', 'blue']):
            lsb1 = arr1[:, :, i] & 1
            lsb2 = arr2[:, :, i] & 1
            diff = (lsb1 != lsb2)
            total = int(diff.size)
            changed = int(np.sum(diff))
            
            result[ch] = {
                'bits_changed': changed,
                'percent_changed': round((changed / total) * 100, 2),
                'original_entropy': round(float(calculate_entropy(lsb1)), 4),
                'stego_entropy': round(float(calculate_entropy(lsb2)), 4)
            }
        
        del arr1, arr2
        gc.collect()
        return result
    except Exception as e:
        print(f"LSB compare error: {e}")
        return {}

def calculate_entropy(data):
    try:
        flat = data.flatten()
        _, counts = np.unique(flat, return_counts=True)
        probs = counts / len(flat)
        return float(-np.sum(probs * np.log2(probs + 1e-10)))
    except:
        return 0.0

def get_image_stats(image_path):
    try:
        img = Image.open(image_path)
        width, height = int(img.width), int(img.height)
        total_pixels = int(width * height)
        
        max_capacity_bits = int(total_pixels * 3 * 1.5)
        delimiter_bits = len('######END######') * 8
        overhead_factor = 0.5
        practical_capacity_bits = int((max_capacity_bits - delimiter_bits) * overhead_factor)
        practical_capacity_chars = int(practical_capacity_bits // 8)
        
        stats = {
            'format': str(img.format) if img.format else 'Unknown',
            'mode': str(img.mode),
            'width': width,
            'height': height,
            'total_pixels': total_pixels,
            'theoretical_max_bits': int(total_pixels * 3 * 2),
            'practical_max_bits': practical_capacity_bits,
            'max_capacity_chars': practical_capacity_chars,
            'overhead_info': 'Accounts for AES encryption + delimiter'
        }
        
        img.close()
        return stats
    except Exception as e:
        print(f"Stats error: {e}")
        return {}

def generate_lsb_visualization(original_path, stego_path, max_size=500):
    try:
        img1 = Image.open(original_path).convert('RGB')
        img2 = Image.open(stego_path).convert('RGB')
        
        if max(img1.size) > max_size:
            img1.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            img2.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        arr1 = np.array(img1)
        arr2 = np.array(img2)
        img1.close()
        img2.close()
        
        viz = {}
        for i, ch in enumerate(['red', 'green', 'blue']):
            lsb1 = arr1[:, :, i] & 1
            lsb2 = arr2[:, :, i] & 1
            
            for name, data in [('original', lsb1), ('stego', lsb2), ('diff', (lsb1 != lsb2).astype(np.uint8))]:
                img_data = (data * 255).astype(np.uint8)
                pil_img = Image.fromarray(img_data, mode='L')
                buf = io.BytesIO()
                pil_img.save(buf, format='PNG', optimize=True, compress_level=6)
                pil_img.close()
                buf.seek(0)
                viz[f'{name}_{ch}_lsb'] = base64.b64encode(buf.read()).decode('utf-8')
                buf.close()
        
        del arr1, arr2
        gc.collect()
        return viz
    except Exception as e:
        print(f"LSB viz error: {e}")
        return {}

def comprehensive_analysis(original_path, stego_path):
    try:
        analysis = {}
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                'psnr': executor.submit(calculate_psnr, original_path, stego_path),
                'mse': executor.submit(calculate_mse, original_path, stego_path),
                'histogram_original': executor.submit(get_histogram_data, original_path),
                'histogram_stego': executor.submit(get_histogram_data, stego_path),
                'lsb_comparison': executor.submit(compare_lsb_planes, original_path, stego_path),
                'lsb_visualizations': executor.submit(generate_lsb_visualization, original_path, stego_path, 500),
                'image_stats_original': executor.submit(get_image_stats, original_path)
            }
            
            for key, future in futures.items():
                try:
                    analysis[key] = future.result(timeout=15)
                except Exception as e:
                    print(f"{key} timeout: {e}")
                    analysis[key] = {}
        
        if 'histogram_original' in analysis and 'histogram_stego' in analysis:
            analysis['histogram_stats_original'] = calculate_histogram_statistics(analysis['histogram_original'])
            analysis['histogram_stats_stego'] = calculate_histogram_statistics(analysis['histogram_stego'])
        
        gc.collect()
        return analysis
    except Exception as e:
        print(f"Comprehensive analysis error: {e}")
        return {'error': str(e)}

def quick_analysis(original_path, stego_path):
    try:
        result = {
            'psnr': calculate_psnr(original_path, stego_path),
            'mse': calculate_mse(original_path, stego_path),
            'histogram_original': get_histogram_data(original_path),
            'histogram_stego': get_histogram_data(stego_path),
            'image_stats_original': get_image_stats(original_path)
        }
        gc.collect()
        return result
    except:
        return {}

def super_fast_analysis(original_path, stego_path):
    try:
        return {
            'psnr': calculate_psnr(original_path, stego_path),
            'mse': calculate_mse(original_path, stego_path),
            'image_stats_original': get_image_stats(original_path),
            'success': True
        }
    except:
        return {}

def calculate_ssim(original_path, stego_path):
    return {'red': 0, 'green': 0, 'blue': 0, 'average': 0}

def analyze_pixel_differences(original_path, stego_path):
    return {}

def calculate_image_entropy(image_path):
    return {}

def chi_square_test(original_hist, stego_hist):
    return {}

def extract_bit_planes(image_path, channel='gray'):
    return {}

def generate_bit_plane_images(image_path, max_size=1024):
    return {}

def generate_difference_heatmap(original_path, stego_path, max_size=500):
    return None
