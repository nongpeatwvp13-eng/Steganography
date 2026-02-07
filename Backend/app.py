from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import io, os, tempfile, base64, sys, json
import numpy as np
from werkzeug.utils import secure_filename
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Functions.Stego import encode_message, decode_message, get_image_stats
from Analyze.image_analyzer import comprehensive_analysis

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

app = Flask(__name__, static_folder='dist', static_url_path='')
app.json_encoder = NumpyEncoder
CORS(app, resources={r"/api/*": {"origins": "*"}})
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 31536000
app.config['JSON_SORT_KEYS'] = False

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'webp', 'tiff', 'tif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def image_to_base64(img_path, max_size=(600, 600), quality=75):
    try:
        img = Image.open(img_path)
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=quality, optimize=True)
        img.close()
        b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        buffer.close()
        return b64
    except:
        return None

@app.route('/api/encode', methods=['POST'])
def encode():
    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            if 'image' not in request.files or 'message' not in request.form or 'password' not in request.form:
                return jsonify({'error': 'Missing required fields'}), 400
            
            image_file = request.files['image']
            if image_file.filename == '' or not allowed_file(image_file.filename):
                return jsonify({'error': 'Invalid file'}), 400
            
            input_path = os.path.join(tmp_dir, f"original_{secure_filename(image_file.filename)}")
            output_path = os.path.join(tmp_dir, "encoded.png")
            image_file.save(input_path)
            
            encode_message(input_path, request.form['message'], request.form['password'], output_path)
            
            return send_file(output_path, mimetype='image/png', as_attachment=True, download_name='encoded.png')
        except ValueError as ve:
            return jsonify({'error': str(ve)}), 400
        except Exception as e:
            return jsonify({'error': f'Encoding failed: {str(e)}'}), 500

@app.route('/api/decode', methods=['POST'])
def decode():
    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            if 'image' not in request.files or 'password' not in request.form:
                return jsonify({'error': 'Missing required fields'}), 400
            
            image_file = request.files['image']
            if image_file.filename == '' or not allowed_file(image_file.filename):
                return jsonify({'error': 'Invalid file'}), 400
            
            input_path = os.path.join(tmp_dir, secure_filename(image_file.filename))
            image_file.save(input_path)
            message = decode_message(input_path, request.form['password'])
            
            if message and message.startswith('Error:'):
                return jsonify({'success': False, 'error': message}), 400
            return jsonify({'success': True, 'message': message or 'No message found'})
        except Exception as e:
            return jsonify({'success': False, 'error': f'Decoding failed: {str(e)}'}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze():
    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            if 'original' not in request.files or 'stego' not in request.files:
                return jsonify({'error': 'Both images required'}), 400
            
            original_path = os.path.join(tmp_dir, 'original.png')
            stego_path = os.path.join(tmp_dir, 'stego.png')
            request.files['original'].save(original_path)
            request.files['stego'].save(stego_path)
            
            analysis_results = comprehensive_analysis(original_path, stego_path)
            analysis_results['original_image'] = image_to_base64(original_path, max_size=(500, 500), quality=70)
            analysis_results['stego_image'] = image_to_base64(stego_path, max_size=(500, 500), quality=70)
            analysis_results['success'] = True
            return jsonify(analysis_results)
        except Exception as e:
            return jsonify({'success': False, 'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/api/capacity', methods=['POST'])
def check_capacity():
    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            if 'image' not in request.files:
                return jsonify({'error': 'No image provided'}), 400
            
            input_path = os.path.join(tmp_dir, secure_filename(request.files['image'].filename))
            request.files['image'].save(input_path)
            return jsonify({'success': True, 'capacity': get_image_stats(input_path)})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/')
def index():
    dist_path = os.path.join(os.path.dirname(__file__), 'dist', 'index.html')
    frontend_path = os.path.join(os.path.dirname(__file__), 'Frontend', 'index.html')
    
    if os.path.exists(dist_path):
        return send_file(dist_path)
    elif os.path.exists(frontend_path):
        return send_file(frontend_path)
    return jsonify({'error': 'Frontend not found. Run: npm run build'}), 404

@app.route('/<path:path>')
def serve_static(path):
    dist_dir = os.path.join(os.path.dirname(__file__), 'dist')
    if os.path.exists(os.path.join(dist_dir, path)):
        return send_from_directory(dist_dir, path)
    
    frontend_dir = os.path.join(os.path.dirname(__file__), 'Frontend')
    if os.path.exists(os.path.join(frontend_dir, path)):
        return send_from_directory(frontend_dir, path)
    
    return jsonify({'error': 'File not found'}), 404

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') != 'production'
    
    if not debug:
        print("Production mode - use gunicorn instead")
        print("gunicorn -w 4 -b 0.0.0.0:5000 --timeout 120 app:app")
    else:
        print("="*60)
        print("LSB Steganography - Development Server")
        print("="*60)
        print(f"Port: {port}")
        print("Open: http://localhost:5000")
        print("="*60)
    
    app.run(debug=debug, port=port, host='0.0.0.0', threaded=True)
