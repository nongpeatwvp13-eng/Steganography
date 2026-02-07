// Main Application Logic
// Coordinates between UI and API with parallel processing

import uiManager from './ui.js';
import api from './api.js';

class SteganographyApp {
    constructor() {
        this.files = {
            encode: null,
            decode: null,
            analyzeOriginal: null,
            analyzeStego: null,
            capacity: null
        };
        
        this.init();
    }

    init() {
        // Initialize tab switching
        document.querySelectorAll('.tab').forEach(tab => {
            tab.addEventListener('click', () => {
                const tabName = tab.getAttribute('data-tab');
                uiManager.switchTab(tabName);
            });
        });

        // Initialize file uploads
        this.initFileUpload('encode-image-upload', 'encode-image-input', 'encode');
        this.initFileUpload('decode-image-upload', 'decode-image-input', 'decode');
        this.initFileUpload('capacity-image-upload', 'capacity-image-input', 'capacity');
        this.initFileUpload('analyze-original-upload', 'analyze-original-input', 'analyzeOriginal');
        this.initFileUpload('analyze-stego-upload', 'analyze-stego-input', 'analyzeStego');

        // Initialize form submissions
        document.getElementById('encode-form')?.addEventListener('submit', (e) => {
            e.preventDefault();
            this.handleEncode();
        });

        document.getElementById('decode-form')?.addEventListener('submit', (e) => {
            e.preventDefault();
            this.handleDecode();
        });

        document.getElementById('capacity-form')?.addEventListener('submit', (e) => {
            e.preventDefault();
            this.handleCapacity();
        });

        document.getElementById('analyze-form')?.addEventListener('submit', (e) => {
            e.preventDefault();
            this.handleAnalyze();
        });

        // Auto-check capacity when encoding image is uploaded
        this.autoCapacityCheck();
    }

    /**
     * Initialize file upload with drag & drop
     */
    initFileUpload(uploadId, inputId, fileKey) {
        const uploadDiv = document.getElementById(uploadId);
        const input = document.getElementById(inputId);

        if (!uploadDiv || !input) return;

        // Click to upload
        uploadDiv.addEventListener('click', () => input.click());

        // File selection
        input.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file && this.validateImage(file)) {
                this.files[fileKey] = file;
                uiManager.updateFileUpload(uploadDiv, file);
                
                // Auto-preview for analysis
                if (fileKey === 'analyzeOriginal' || fileKey === 'analyzeStego') {
                    uiManager.previewImage(file, `${fileKey}-preview`);
                }
            }
        });

        // Drag & drop
        uploadDiv.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadDiv.style.borderColor = '#0066cc';
        });

        uploadDiv.addEventListener('dragleave', () => {
            uploadDiv.style.borderColor = '#e0e0e0';
        });

        uploadDiv.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadDiv.style.borderColor = '#e0e0e0';
            
            const file = e.dataTransfer.files[0];
            if (file && this.validateImage(file)) {
                this.files[fileKey] = file;
                input.files = e.dataTransfer.files;
                uiManager.updateFileUpload(uploadDiv, file);
            }
        });
    }

    /**
     * Validate image file
     */
    validateImage(file) {
        const validTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/bmp', 'image/webp', 'image/tiff'];
        const maxSize = 100 * 1024 * 1024; // 100MB

        if (!validTypes.includes(file.type)) {
            uiManager.showAlert('encode-alert', 'Please select a valid image file (PNG, JPEG, BMP, WEBP, TIFF)', 'error');
            return false;
        }

        if (file.size > maxSize) {
            uiManager.showAlert('encode-alert', 'File size must be less than 100MB', 'error');
            return false;
        }

        return true;
    }

    /**
     * Auto-check capacity when image is uploaded for encoding
     */
    autoCapacityCheck() {
        const input = document.getElementById('encode-image-input');
        if (!input) return;

        input.addEventListener('change', async () => {
            if (this.files.encode) {
                try {
                    const result = await api.checkCapacity(this.files.encode);
                    this.displayQuickCapacity(result.capacity);
                } catch (error) {
                    console.error('Auto capacity check failed:', error);
                }
            }
        });
    }

    /**
     * Display quick capacity info
     */
    displayQuickCapacity(capacity) {
        const info = document.getElementById('capacity-info');
        if (!info) return;

        info.innerHTML = `
            <div class="alert alert-info show">
                <strong>Image Capacity:</strong> ${uiManager.formatNumber(capacity.max_capacity_chars)} characters
                (${capacity.width}×${capacity.height} pixels)
            </div>
        `;
    }

    /**
     * Handle encoding
     */
    async handleEncode() {
        uiManager.clearAll();
        uiManager.hideResult('encode-result');
        
        const message = document.getElementById('encode-message').value;
        const password = document.getElementById('encode-password').value;

        // Validation
        if (!this.files.encode) {
            uiManager.showAlert('encode-alert', 'Please select an image', 'error');
            return;
        }

        if (!message.trim()) {
            uiManager.showAlert('encode-alert', 'Please enter a message', 'error');
            return;
        }

        if (password.length < 4) {
            uiManager.showAlert('encode-alert', 'Password must be at least 4 characters', 'error');
            return;
        }

        try {
            uiManager.showLoading('encode-loading');
            uiManager.setButtonState('encode-btn', false);

            const result = await api.encode(this.files.encode, message, password);

            uiManager.hideLoading('encode-loading');
            uiManager.setButtonState('encode-btn', true);

            if (result.success) {
                // Download the encoded image
                uiManager.downloadBlob(result.blob, result.filename);
                
                uiManager.showAlert('encode-alert', '✓ Message encoded successfully! File downloaded.', 'success');
                
                // Clear form
                document.getElementById('encode-message').value = '';
                document.getElementById('encode-password').value = '';
            }

        } catch (error) {
            uiManager.hideLoading('encode-loading');
            uiManager.setButtonState('encode-btn', true);
            uiManager.showAlert('encode-alert', error.message, 'error');
        }
    }

    /**
     * Handle decoding
     */
    async handleDecode() {
        uiManager.clearAll();
        uiManager.hideResult('decode-result');
        
        const password = document.getElementById('decode-password').value;

        // Validation
        if (!this.files.decode) {
            uiManager.showAlert('decode-alert', 'Please select an image', 'error');
            return;
        }

        if (!password) {
            uiManager.showAlert('decode-alert', 'Please enter the password', 'error');
            return;
        }

        try {
            uiManager.showLoading('decode-loading');
            uiManager.setButtonState('decode-btn', false);

            const result = await api.decode(this.files.decode, password);

            uiManager.hideLoading('decode-loading');
            uiManager.setButtonState('decode-btn', true);

            if (result.success) {
                uiManager.displayMessage(result.message, 'decode-result');
                uiManager.showAlert('decode-alert', '✓ Message decoded successfully!', 'success');
            }

        } catch (error) {
            uiManager.hideLoading('decode-loading');
            uiManager.setButtonState('decode-btn', true);
            uiManager.showAlert('decode-alert', error.message, 'error');
        }
    }

    /**
     * Handle capacity check
     */
    async handleCapacity() {
        uiManager.clearAll();
        uiManager.hideResult('capacity-result');

        if (!this.files.capacity) {
            uiManager.showAlert('capacity-alert', 'Please select an image', 'error');
            return;
        }

        try {
            uiManager.showLoading('capacity-loading');
            uiManager.setButtonState('capacity-btn', false);

            const result = await api.checkCapacity(this.files.capacity);

            uiManager.hideLoading('capacity-loading');
            uiManager.setButtonState('capacity-btn', true);

            if (result.success) {
                uiManager.displayCapacity(result.capacity, 'capacity-result');
            }

        } catch (error) {
            uiManager.hideLoading('capacity-loading');
            uiManager.setButtonState('capacity-btn', true);
            uiManager.showAlert('capacity-alert', error.message, 'error');
        }
    }

    /**
     * Handle analysis - Process in parallel when possible
     */
    async handleAnalyze() {
        uiManager.clearAll();
        uiManager.hideResult('analyze-result');

        if (!this.files.analyzeOriginal || !this.files.analyzeStego) {
            uiManager.showAlert('analyze-alert', 'Please select both original and stego images', 'error');
            return;
        }

        try {
            uiManager.showLoading('analyze-loading');
            uiManager.setButtonState('analyze-btn', false);

            const result = await api.analyze(this.files.analyzeOriginal, this.files.analyzeStego);

            uiManager.hideLoading('analyze-loading');
            uiManager.setButtonState('analyze-btn', true);

            if (result.success) {
                uiManager.displayAnalysis(result, 'analyze-result');
                uiManager.showAlert('analyze-alert', '✓ Analysis completed!', 'success');
            }

        } catch (error) {
            uiManager.hideLoading('analyze-loading');
            uiManager.setButtonState('analyze-btn', true);
            uiManager.showAlert('analyze-alert', error.message, 'error');
        }
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.app = new SteganographyApp();
});
