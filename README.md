# LSB Steganography

Adaptive LSB steganography with AES-256-GCM encryption. Hide secret messages inside images with minimal visual distortion.

## Features

- **Adaptive bit embedding** - Analyzes local texture (variance, gradient, color zone) to decide where and how many bits to embed per pixel, prioritizing high-complexity regions
- **AES-256-GCM encryption** - Messages are encrypted with a password-derived key (PBKDF2, 100k iterations) before embedding
- **Comprehensive image analysis** - Compare original and stego images with PSNR, MSE, SSIM, histogram analysis, LSB statistics, spatial distribution, and bit-plane visualization
- **Capacity checking** - Estimate or compute exact embedding capacity for any image

## Tech Stack

- **Backend**: Python / Flask
- **Frontend**: Vanilla JS with Vite build tooling
- **Crypto**: PyCryptodome (AES-256-GCM)
- **Image processing**: NumPy, Pillow, OpenCV, scikit-image

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+

### Install dependencies

```bash
pip install -r requirements.txt
npm install
```

### Run in development mode

```bash
bash run.sh
```

This starts the Flask backend on port 7860 and the Vite dev server on port 3000 (with API proxy to the backend).

### Build for production

```bash
npm run build
python Backend/app.py
```

The production build serves the frontend from `dist/` via Flask on port 7860.

### Docker

```bash
docker build -t steganography .
docker run -p 7860:7860 steganography
```

## Project Structure

```
Backend/
  Functions/
    AES_256.py       # AES-256-GCM encryption/decryption
    common.py        # Shared utilities (seed derivation, header positions)
    decide.py        # Adaptive LSB core (scoring, encoding, decoding)
    encode_LSB.py    # High-level encode pipeline
    decode_LSB.py    # High-level decode pipeline
    Stego.py         # Public API (encode_message, decode_message, get_image_stats)
  Analyze/
    image_analyzer.py  # Comprehensive stego analysis (PSNR, SSIM, histograms, etc.)
  app.py             # Flask application and API routes
Frontend/
  index.html         # Single-page app
  main.js            # Event handlers and API calls
  ui.js              # UI rendering and chart management
  styles.css         # Main stylesheet
  bit_plane_styles.css  # Bit-plane analysis styles
tests/
  test_aes.py              # AES encryption tests
  test_stego_roundtrip.py  # End-to-end encode/decode tests
```

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/encode` | POST | Encode a message into an image |
| `/api/decode` | POST | Decode a message from a stego image |
| `/api/capacity` | POST | Check embedding capacity of an image |
| `/api/analyze` | POST | Compare original and stego images |

## How It Works

1. **Encryption**: The plaintext message is encrypted with AES-256-GCM using a key derived from the password via PBKDF2 (random salt per encryption)
2. **Scoring**: Each pixel is scored based on local variance, gradient magnitude, and color zone. High-texture areas get higher scores, allowing 2-bit embedding; low-texture areas get 1-bit or are skipped
3. **Embedding**: Encrypted bits are embedded in LSB positions of scored pixels, with positions shuffled using a password-derived PRNG seed
4. **Header**: A 32-bit payload length is stored at deterministic positions derived from the password

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```
