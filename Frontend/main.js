import uiManager from './ui.js';

const API_BASE = window.location.origin;

document.addEventListener('DOMContentLoaded', () => {
    console.log('LSB Steganography App Initialized');
    initializeTabs();
    initializeFileUploads();
    initializeForms();
});

function initializeTabs() {
    document.querySelectorAll('.tab').forEach(tab => {
        tab.addEventListener('click', () => {
            const tabName = tab.getAttribute('data-tab');
            uiManager.switchTab(tabName);
        });
    });
}

function initializeFileUploads() {
    const uploads = [
        { uploadId: 'encode-image-upload', inputId: 'encode-image-input' },
        { uploadId: 'decode-image-upload', inputId: 'decode-image-input' },
        { uploadId: 'capacity-image-upload', inputId: 'capacity-image-input' },
        { uploadId: 'analyze-original-upload', inputId: 'analyze-original-input', previewId: 'analyzeOriginal-preview' },
        { uploadId: 'analyze-stego-upload', inputId: 'analyze-stego-input', previewId: 'analyzeStego-preview' }
    ];

    uploads.forEach(({ uploadId, inputId, previewId }) => {
        const uploadDiv = document.getElementById(uploadId);
        const input = document.getElementById(inputId);

        if (!uploadDiv || !input) return;

        uploadDiv.addEventListener('click', () => input.click());

        input.addEventListener('change', (e) => {
            const file = e.target.files[0];
            uiManager.updateFileUpload(uploadDiv, file);
            
            if (previewId && file) {
                uiManager.previewImage(file, previewId);
            }
        });

        uploadDiv.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadDiv.style.borderColor = '#0066cc';
        });

        uploadDiv.addEventListener('dragleave', () => {
            uploadDiv.style.borderColor = '';
        });

        uploadDiv.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadDiv.style.borderColor = '';
            
            const file = e.dataTransfer.files[0];
            if (file && file.type.startsWith('image/')) {
                input.files = e.dataTransfer.files;
                uiManager.updateFileUpload(uploadDiv, file);
                
                if (previewId) {
                    uiManager.previewImage(file, previewId);
                }
            }
        });
    });
}

function initializeForms() {
    document.getElementById('encode-form').addEventListener('submit', handleEncode);
    document.getElementById('decode-form').addEventListener('submit', handleDecode);
    document.getElementById('capacity-form').addEventListener('submit', handleCapacity);
    document.getElementById('analyze-form').addEventListener('submit', handleAnalyze);
}

async function handleEncode(e) {
    e.preventDefault();
    
    const imageInput = document.getElementById('encode-image-input');
    const message = document.getElementById('encode-message').value;
    const password = document.getElementById('encode-password').value;
    
    if (!imageInput.files[0]) {
        uiManager.showAlert('encode-alert', 'Please select an image', 'error');
        return;
    }
    
    if (password.length < 4) {
        uiManager.showAlert('encode-alert', 'Password must be at least 4 characters', 'error');
        return;
    }
    
    uiManager.toggleLoading('encode-loading', true);
    uiManager.toggleResult('encode-result', false);
    
    try {
        const formData = new FormData();
        formData.append('image', imageInput.files[0]);
        formData.append('message', message);
        formData.append('password', password);
        
        const response = await fetch(`${API_BASE}/api/encode`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            let errorMessage = 'Encoding failed';
            try {
                const error = await response.json();
                errorMessage = error.error || errorMessage;
            } catch {
                errorMessage = `HTTP ${response.status}: ${response.statusText}`;
            }
            throw new Error(errorMessage);
        }
        
        const blob = await response.blob();
        uiManager.downloadBlob(blob, 'encoded.png');
        
        uiManager.showAlert('encode-alert', 'Message encoded successfully! Image downloaded.', 'success');
        
        document.getElementById('encode-form').reset();
        uiManager.updateFileUpload(document.getElementById('encode-image-upload'), null);
        
    } catch (error) {
        console.error('Encode error:', error);
        uiManager.showAlert('encode-alert', error.message, 'error');
    } finally {
        uiManager.toggleLoading('encode-loading', false);
    }
}

async function handleDecode(e) {
    e.preventDefault();
    
    const imageInput = document.getElementById('decode-image-input');
    const password = document.getElementById('decode-password').value;
    
    if (!imageInput.files[0]) {
        uiManager.showAlert('decode-alert', 'Please select an image', 'error');
        return;
    }
    
    uiManager.toggleLoading('decode-loading', true);
    uiManager.toggleResult('decode-result', false);
    
    try {
        const formData = new FormData();
        formData.append('image', imageInput.files[0]);
        formData.append('password', password);
        
        const response = await fetch(`${API_BASE}/api/decode`, {
            method: 'POST',
            body: formData
        });
        
        let data;
        try {
            data = await response.json();
        } catch {
            throw new Error(`Invalid response from server (HTTP ${response.status})`);
        }
        
        if (!response.ok || !data.success) {
            throw new Error(data.error || 'Decoding failed');
        }
        
        uiManager.displayMessage(data.message, 'decode-result');
        uiManager.showAlert('decode-alert', 'Message decoded successfully', 'success');
        
    } catch (error) {
        console.error('Decode error:', error);
        uiManager.showAlert('decode-alert', error.message, 'error');
    } finally {
        uiManager.toggleLoading('decode-loading', false);
    }
}

async function handleCapacity(e) {
    e.preventDefault();
    
    const imageInput = document.getElementById('capacity-image-input');
    
    if (!imageInput.files[0]) {
        uiManager.showAlert('capacity-alert', 'Please select an image', 'error');
        return;
    }
    
    uiManager.toggleLoading('capacity-loading', true);
    uiManager.toggleResult('capacity-result', false);
    
    try {
        const formData = new FormData();
        formData.append('image', imageInput.files[0]);
        
        const response = await fetch(`${API_BASE}/api/capacity`, {
            method: 'POST',
            body: formData
        });
        
        let data;
        try {
            data = await response.json();
        } catch {
            throw new Error(`Invalid response from server (HTTP ${response.status})`);
        }
        
        if (!response.ok || !data.success) {
            throw new Error(data.error || 'Capacity check failed');
        }
        
        uiManager.displayCapacity(data.capacity, 'capacity-result');
        uiManager.showAlert('capacity-alert', 'Capacity calculated successfully', 'success');
        
    } catch (error) {
        console.error('Capacity error:', error);
        uiManager.showAlert('capacity-alert', error.message, 'error');
    } finally {
        uiManager.toggleLoading('capacity-loading', false);
    }
}

async function handleAnalyze(e) {
    e.preventDefault();
    
    const originalInput = document.getElementById('analyze-original-input');
    const stegoInput = document.getElementById('analyze-stego-input');
    
    if (!originalInput.files[0] || !stegoInput.files[0]) {
        uiManager.showAlert('analyze-alert', 'Please select both original and stego images', 'error');
        return;
    }
    
    uiManager.toggleLoading('analyze-loading', true);
    uiManager.toggleResult('analyze-result', false);

    document.getElementById('analyze-result').innerHTML = '<div id="histogram-charts"></div>';
    
    try {
        const formData = new FormData();
        formData.append('original', originalInput.files[0]);
        formData.append('stego', stegoInput.files[0]);
        
        console.log('Starting analysis...');
        
        const response = await fetch(`${API_BASE}/api/analyze`, {
            method: 'POST',
            body: formData
        });
        
        let data;
        try {
            data = await response.json();
        } catch {
            throw new Error(`Invalid response from server (HTTP ${response.status})`);
        }
        
        if (!response.ok || !data.success) {
            throw new Error(data.error || 'Analysis failed');
        }
        
        console.log('Analysis complete!', data);
        
        uiManager.displayAnalysis(data, 'analyze-result');
        uiManager.showAlert('analyze-alert', 'Analysis complete! Scroll down to see results.', 'success');
        
    } catch (error) {
        console.error('Analysis error:', error);
        uiManager.showAlert('analyze-alert', error.message, 'error');
    } finally {
        uiManager.toggleLoading('analyze-loading', false);
    }
}

window.app = {
    uiManager,
    handleEncode,
    handleDecode,
    handleCapacity,
    handleAnalyze
};

console.log('All handlers initialized');