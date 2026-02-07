// UI Utility Functions
// Handles all UI updates, notifications, and visual feedback

import Chart from 'chart.js/auto';

class UIManager {
    constructor() {
        this.activeTab = 'encode';
    }

    /**
     * Show loading indicator
     */
    showLoading(elementId) {
        const element = document.getElementById(elementId);
        if (element) {
            element.classList.add('show');
        }
    }

    /**
     * Hide loading indicator
     */
    hideLoading(elementId) {
        const element = document.getElementById(elementId);
        if (element) {
            element.classList.remove('show');
        }
    }

    /**
     * Show alert message
     */
    showAlert(elementId, message, type = 'info') {
        const element = document.getElementById(elementId);
        if (element) {
            element.textContent = message;
            element.className = `alert alert-${type} show`;
            
            // Auto-hide after 5 seconds
            setTimeout(() => {
                this.hideAlert(elementId);
            }, 5000);
        }
    }

    /**
     * Hide alert message
     */
    hideAlert(elementId) {
        const element = document.getElementById(elementId);
        if (element) {
            element.classList.remove('show');
        }
    }

    /**
     * Show result section
     */
    showResult(elementId) {
        const element = document.getElementById(elementId);
        if (element) {
            element.classList.add('show');
        }
    }

    /**
     * Hide result section
     */
    hideResult(elementId) {
        const element = document.getElementById(elementId);
        if (element) {
            element.classList.remove('show');
        }
    }

    /**
     * Update file upload UI
     */
    updateFileUpload(uploadElement, file) {
        if (!uploadElement) return;

        const textElement = uploadElement.querySelector('.file-upload-text');
        if (textElement && file) {
            textElement.textContent = `Selected: ${file.name} (${this.formatFileSize(file.size)})`;
            uploadElement.classList.add('has-file');
        } else if (textElement) {
            textElement.textContent = 'Click to select image or drag and drop';
            uploadElement.classList.remove('has-file');
        }
    }

    /**
     * Format file size
     */
    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
    }

    /**
     * Format number with commas
     */
    formatNumber(num) {
        return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
    }

    /**
     * Display capacity information
     */
    displayCapacity(capacity, containerId) {
        const container = document.getElementById(containerId);
        if (!container) return;

        const html = `
            <div class="stats-grid">
                <div class="stat-card">
                    <h4>Image Dimensions</h4>
                    <div class="value">${capacity.width} × ${capacity.height}</div>
                </div>
                <div class="stat-card">
                    <h4>Total Pixels</h4>
                    <div class="value">${this.formatNumber(capacity.total_pixels)}</div>
                </div>
                <div class="stat-card">
                    <h4>Max Capacity</h4>
                    <div class="value">${this.formatNumber(capacity.max_capacity_chars)}<span class="unit">chars</span></div>
                </div>
                <div class="stat-card">
                    <h4>Practical Capacity</h4>
                    <div class="value">${this.formatNumber(capacity.practical_max_bits)}<span class="unit">bits</span></div>
                </div>
            </div>
            <div class="alert alert-info show" style="margin-top: 20px;">
                <strong>Note:</strong> ${capacity.overhead_info || 'Practical capacity accounts for encryption overhead'}
            </div>
        `;

        container.innerHTML = html;
        this.showResult(containerId);
    }

    /**
     * Display decoded message
     */
    displayMessage(message, containerId) {
        const container = document.getElementById(containerId);
        if (!container) return;

        const html = `
            <div class="result-text">${this.escapeHtml(message)}</div>
            <div class="flex gap-10 mt-20">
                <button class="btn btn-secondary" onclick="uiManager.copyToClipboard('${this.escapeHtml(message)}')">
                    📋 Copy Message
                </button>
            </div>
        `;

        container.innerHTML = html;
        this.showResult(containerId);
    }

    /**
     * Copy text to clipboard
     */
    async copyToClipboard(text) {
        try {
            await navigator.clipboard.writeText(text);
            alert('Message copied to clipboard!');
        } catch (err) {
            console.error('Failed to copy:', err);
        }
    }

    /**
     * Escape HTML to prevent XSS
     */
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    /**
     * Download blob as file
     */
    downloadBlob(blob, filename) {
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    /**
     * Preview image from file
     */
    previewImage(file, containerId) {
        const container = document.getElementById(containerId);
        if (!container || !file) return;

        const reader = new FileReader();
        reader.onload = (e) => {
            container.innerHTML = `<img src="${e.target.result}" alt="Preview">`;
        };
        reader.readAsDataURL(file);
    }

    /**
     * Switch tabs
     */
    switchTab(tabName) {
        // Hide all tabs
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.remove('active');
        });
        document.querySelectorAll('.tab').forEach(tab => {
            tab.classList.remove('active');
        });

        // Show selected tab
        const content = document.getElementById(`${tabName}-tab`);
        const tab = document.querySelector(`[data-tab="${tabName}"]`);
        
        if (content) content.classList.add('active');
        if (tab) tab.classList.add('active');

        this.activeTab = tabName;
    }

    /**
     * Display analysis results with charts
     */
    displayAnalysis(analysis, containerId) {
        const container = document.getElementById(containerId);
        if (!container) return;

        let html = `
            <div class="stats-grid">
                <div class="stat-card">
                    <h4>PSNR (Peak Signal-to-Noise Ratio)</h4>
                    <div class="value">${analysis.psnr || 0}<span class="unit">dB</span></div>
                    <p style="font-size: 0.85em; margin-top: 8px; color: #666;">
                        ${analysis.psnr > 40 ? 'Excellent - Nearly imperceptible' : 
                          analysis.psnr > 30 ? 'Good - Minor differences' : 'Noticeable differences'}
                    </p>
                </div>
                <div class="stat-card">
                    <h4>MSE (Mean Squared Error)</h4>
                    <div class="value">${(analysis.mse || 0).toFixed(4)}</div>
                    <p style="font-size: 0.85em; margin-top: 8px; color: #666;">
                        ${analysis.mse < 1 ? 'Excellent quality' : 
                          analysis.mse < 5 ? 'Good quality' : 'Quality degraded'}
                    </p>
                </div>
            </div>

            <div class="image-grid mt-20">
                <div class="image-item">
                    <h4>Original Image</h4>
                    <img src="data:image/jpeg;base64,${analysis.original_image}" alt="Original">
                </div>
                <div class="image-item">
                    <h4>Stego Image</h4>
                    <img src="data:image/jpeg;base64,${analysis.stego_image}" alt="Stego">
                </div>
            </div>
        `;

        // Add LSB comparison if available
        if (analysis.lsb_comparison) {
            html += `
                <div class="section mt-20">
                    <h3>LSB Bit Plane Analysis</h3>
                    <div class="stats-grid">
                        ${Object.entries(analysis.lsb_comparison).map(([channel, data]) => `
                            <div class="stat-card">
                                <h4>${channel.toUpperCase()} Channel</h4>
                                <div class="value">${data.bits_changed || 0}<span class="unit">bits changed</span></div>
                                <p style="font-size: 0.85em; margin-top: 8px;">
                                    ${(data.percent_changed || 0).toFixed(2)}% modified
                                </p>
                            </div>
                        `).join('')}
                    </div>
                </div>
            `;
        }

        container.innerHTML = html;
        this.showResult(containerId);

        // Render histograms if available
        if (analysis.histogram_original && analysis.histogram_stego) {
            this.renderHistogramComparison(analysis, 'histogram-charts');
        }
    }

    /**
     * Render histogram comparison using Chart.js
     */
    renderHistogramComparison(analysis, containerId) {
        const container = document.getElementById(containerId);
        if (!container) return;

        const channels = ['red', 'green', 'blue'];
        const colors = ['rgba(255, 99, 132, 0.5)', 'rgba(75, 192, 192, 0.5)', 'rgba(54, 162, 235, 0.5)'];

        channels.forEach((channel, index) => {
            const canvasId = `chart-${channel}`;
            const chartDiv = document.createElement('div');
            chartDiv.className = 'chart-container';
            chartDiv.innerHTML = `<canvas id="${canvasId}"></canvas>`;
            container.appendChild(chartDiv);

            const ctx = document.getElementById(canvasId).getContext('2d');
            new Chart(ctx, {
                type: 'line',
                data: {
                    labels: Array.from({length: 256}, (_, i) => i),
                    datasets: [
                        {
                            label: `Original ${channel.toUpperCase()}`,
                            data: analysis.histogram_original[channel],
                            borderColor: colors[index],
                            backgroundColor: colors[index],
                            borderWidth: 1,
                            pointRadius: 0
                        },
                        {
                            label: `Stego ${channel.toUpperCase()}`,
                            data: analysis.histogram_stego[channel],
                            borderColor: colors[index].replace('0.5', '0.8'),
                            backgroundColor: 'transparent',
                            borderWidth: 1,
                            borderDash: [5, 5],
                            pointRadius: 0
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        title: {
                            display: true,
                            text: `${channel.toUpperCase()} Channel Histogram`
                        }
                    },
                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: 'Pixel Value'
                            }
                        },
                        y: {
                            title: {
                                display: true,
                                text: 'Frequency'
                            }
                        }
                    }
                }
            });
        });
    }

    /**
     * Enable/disable button
     */
    setButtonState(buttonId, enabled) {
        const button = document.getElementById(buttonId);
        if (button) {
            button.disabled = !enabled;
        }
    }

    /**
     * Clear all results and alerts
     */
    clearAll() {
        document.querySelectorAll('.result').forEach(el => el.classList.remove('show'));
        document.querySelectorAll('.alert').forEach(el => el.classList.remove('show'));
    }
}

// Create singleton instance
const uiManager = new UIManager();

// Export for ES modules
export default uiManager;
