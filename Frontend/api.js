// API Client for Steganography App
// Handles all backend communication with parallel processing support

const API_BASE = '/api';

class SteganographyAPI {
    constructor() {
        this.pendingRequests = new Map();
    }

    /**
     * Encode message into image
     */
    async encode(imageFile, message, password) {
        const formData = new FormData();
        formData.append('image', imageFile);
        formData.append('message', message);
        formData.append('password', password);

        try {
            const response = await fetch(`${API_BASE}/encode`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || 'Encoding failed');
            }

            // Return blob for download
            const blob = await response.blob();
            return {
                success: true,
                blob: blob,
                filename: 'encoded_message.png'
            };
        } catch (error) {
            throw new Error(`Encoding failed: ${error.message}`);
        }
    }

    /**
     * Decode message from image
     */
    async decode(imageFile, password) {
        const formData = new FormData();
        formData.append('image', imageFile);
        formData.append('password', password);

        try {
            const response = await fetch(`${API_BASE}/decode`, {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (!response.ok || !data.success) {
                throw new Error(data.error || 'Decoding failed');
            }

            return {
                success: true,
                message: data.message
            };
        } catch (error) {
            throw new Error(`Decoding failed: ${error.message}`);
        }
    }

    /**
     * Check image capacity
     */
    async checkCapacity(imageFile) {
        const formData = new FormData();
        formData.append('image', imageFile);

        try {
            const response = await fetch(`${API_BASE}/capacity`, {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (!response.ok || !data.success) {
                throw new Error(data.error || 'Capacity check failed');
            }

            return {
                success: true,
                capacity: data.capacity
            };
        } catch (error) {
            throw new Error(`Capacity check failed: ${error.message}`);
        }
    }

    /**
     * Analyze images (original vs stego)
     */
    async analyze(originalFile, stegoFile) {
        const formData = new FormData();
        formData.append('original', originalFile);
        formData.append('stego', stegoFile);

        try {
            const response = await fetch(`${API_BASE}/analyze`, {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (!response.ok || !data.success) {
                throw new Error(data.error || 'Analysis failed');
            }

            return {
                success: true,
                ...data
            };
        } catch (error) {
            throw new Error(`Analysis failed: ${error.message}`);
        }
    }

    /**
     * Batch process multiple operations in parallel
     * Example: Upload image and check capacity simultaneously
     */
    async batchProcess(operations) {
        const promises = operations.map(op => {
            switch(op.type) {
                case 'encode':
                    return this.encode(op.image, op.message, op.password);
                case 'decode':
                    return this.decode(op.image, op.password);
                case 'capacity':
                    return this.checkCapacity(op.image);
                case 'analyze':
                    return this.analyze(op.original, op.stego);
                default:
                    return Promise.reject(new Error(`Unknown operation: ${op.type}`));
            }
        });

        try {
            const results = await Promise.all(promises);
            return {
                success: true,
                results: results
            };
        } catch (error) {
            throw new Error(`Batch processing failed: ${error.message}`);
        }
    }

    /**
     * Cancel pending request
     */
    cancelRequest(requestId) {
        const controller = this.pendingRequests.get(requestId);
        if (controller) {
            controller.abort();
            this.pendingRequests.delete(requestId);
        }
    }

    /**
     * Cancel all pending requests
     */
    cancelAllRequests() {
        this.pendingRequests.forEach(controller => controller.abort());
        this.pendingRequests.clear();
    }
}

// Create singleton instance
const api = new SteganographyAPI();

// Export for use in other modules
export default api;
