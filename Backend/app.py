from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import io, os, tempfile, base64, sys, json
import numpy as np
from werkzeug.utils import secure_filename
from PIL import Image
import logging
import traceback
from flask.json.provider import DefaultJSONProvider

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

class CustomJSONProvider(DefaultJSONProvider):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)): return int(obj)
        if isinstance(obj, (np.floating, np.float64)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

app = Flask(__name__, static_folder=os.path.join(BASE_DIR, 'dist'), static_url_path='')
app.json = CustomJSONProvider(app)
CORS(app, resources={
    r"/api/*": {
        "origins": os.environ.get("CORS_ORIGIN", "*")
    }
})


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

Image.MAX_IMAGE_PIXELS = None
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 31536000

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Functions.Stego import encode_message, decode_message, get_image_stats
from Analyze.image_analyzer import comprehensive_analysis

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'webp', 'tiff', 'tif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def image_to_base64(img_path, max_size=(400, 400), quality=60):
    try:
        img = Image.open(img_path)
        img.thumbnail(max_size, Image.Resampling.NEAREST)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=quality, optimize=True)
        b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        img.close()
        return b64
    except Exception as e:
        logger.error(f"Base64 Error: {e}")
        return None

@app.route('/api/encode', methods=['POST'])
def encode():
    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            if 'image' not in request.files:
                return jsonify({'error': 'No image uploaded'}), 400
            image_file = request.files['image']
            if not allowed_file(image_file.filename):
                return jsonify({'error': 'Invalid file type'}), 400
            
            input_path = os.path.join(tmp_dir, f"original_{secure_filename(image_file.filename)}")
            output_path = os.path.join(tmp_dir, "encoded.png")
            image_file.save(input_path)

            stats = get_image_stats(input_path, real_capacity=True)
            
            if 'message' not in request.form or 'password' not in request.form:
                return jsonify({'error': 'Missing message or password'}), 400

            message_bytes = len(request.form['message'].encode('utf-8'))
            message_bits = message_bytes * 8 * 1.3

            
            if message_bits > stats['practical_max_bits']:
                return jsonify({
                    'error': f"Image capacity too low! Need {int(message_bits)} bits, but only have {stats['practical_max_bits']}."
                }), 400

            encode_message(input_path, request.form['message'], request.form['password'], output_path)
            return send_file(output_path, mimetype='image/png', as_attachment=True, download_name='encoded.png')
            
        except Exception as e:
            logger.exception(f"Encoding error: {e}")
            return jsonify({'error': 'Internal server error during encoding'}), 500

@app.route('/api/decode', methods=['POST'])
def decode():
    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            if 'image' not in request.files or 'password' not in request.form:
                return jsonify({'error': 'Missing required fields'}), 400
            image_file = request.files['image']
            if not allowed_file(image_file.filename):
                return jsonify({'error': 'Invalid file type'}), 400
            
            input_path = os.path.join(tmp_dir, secure_filename(image_file.filename))
            image_file.save(input_path)
            
            message = decode_message(input_path, request.form['password'])
            
            if message and message.startswith('Error:'):
                return jsonify({'success': False, 'error': message}), 400
                
            return jsonify({'success': True, 'message': message or 'No message found'})
            
        except Exception as e:
            logger.error(f"Decoding error: {e}")
            return jsonify({'success': False, 'error': 'Failed to decode message'}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze():
    with tempfile.TemporaryDirectory() as tmp_dir:
        if 'original' not in request.files or 'stego' not in request.files:
            return jsonify({'error': 'Both images required'}), 400

        paths = {}
        for key in ['original', 'stego']:
            f = request.files[key]
            if not allowed_file(f.filename): return jsonify({'error': f'Invalid {key}'}), 400
            paths[key] = os.path.join(tmp_dir, f"{key}.png")
            f.save(paths[key])

        try:            
            orig_p = paths['original']
            steg_p = paths['stego']
            logger.info("Starting comprehensive analysis...")
            analysis_results = comprehensive_analysis(orig_p, steg_p)
            
            if 'error' in analysis_results:
                return jsonify({'success': False, 'error': analysis_results['error']}), 500
            
            analysis_results['original_image'] = image_to_base64(orig_p)
            analysis_results['stego_image'] = image_to_base64(steg_p)
            analysis_results['success'] = True
            
            return jsonify(analysis_results)
            
        except Exception as e:
            logger.exception(f"Analysis error: {e}")
            return jsonify({'success': False, 'error': 'Analysis failed'}), 500

@app.route('/api/capacity', methods=['POST'])
def check_capacity():
    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            if 'image' not in request.files:
                return jsonify({'success': False, 'error': 'No image uploaded'}), 400

            is_real = request.form.get('real', 'false').lower() == 'true'
            image_file = request.files['image']
            
            input_path = os.path.join(tmp_dir, secure_filename(image_file.filename))
            image_file.save(input_path)
            
            capacity = get_image_stats(input_path, real_capacity=is_real)
            return jsonify({'success': True, 'capacity': capacity})
        except Exception as e:
            logger.error(f"Capacity check error: {e}")
            return jsonify({'success': False, 'error': "Failed to check capacity"}), 500
        
@app.route('/')
def index():
    for folder in ['dist', 'Frontend']:
        path = os.path.join(BASE_DIR, folder, 'index.html')
        if os.path.exists(path): 
            return send_file(path)
    return jsonify({'error': 'Frontend not found', 'checked_path': os.path.join(BASE_DIR, 'dist')}), 404

@app.route('/<path:path>')
def serve_static(path):
    for folder in ['dist', 'Frontend']:
        dir_path = os.path.join(BASE_DIR, folder)
        if os.path.exists(os.path.join(dir_path, path)):
            return send_from_directory(dir_path, path)
    return jsonify({'error': 'File not found'}), 404

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') != 'production'
    app.run(debug=debug, port=port, host='0.0.0.0', threaded=True)