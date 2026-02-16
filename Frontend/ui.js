class UIManager {
    constructor() {
        this.activeTab = 'encode';
        this.charts = {};
    }

    toggleLoading(id, show) {
        const el = document.getElementById(id);
        if (el) el.classList.toggle('show', show);
    }

    showAlert(id, message, type = 'info') {
        const el = document.getElementById(id);
        if (!el) return;
        el.textContent = message;
        el.className = `alert alert-${type} show`;
        setTimeout(() => el.classList.remove('show'), 5000);
    }

    toggleResult(id, show) {
        const el = document.getElementById(id);
        if (el) el.classList.toggle('show', show);
    }

    updateFileUpload(uploadEl, file) {
        if (!uploadEl) return;
        const textEl = uploadEl.querySelector('.file-upload-text');
        if (!textEl) return;
        
        if (file) {
            textEl.textContent = `${file.name} (${this.formatBytes(file.size)})`;
            uploadEl.classList.add('has-file');
        } else {
            textEl.textContent = 'Click to select or drag and drop';
            uploadEl.classList.remove('has-file');
        }
    }

    formatBytes(bytes) {
        if (!bytes) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return `${(bytes / Math.pow(k, i)).toFixed(1)} ${sizes[i]}`;
    }

    formatNumber(num) {
        if (num === undefined || num === null) return '0';
        return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ',');
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    switchTab(tabName) {
        document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
        document.querySelectorAll('.tab').forEach(el => el.classList.remove('active'));

        const content = document.getElementById(`${tabName}-tab`);
        const tab = document.querySelector(`[data-tab="${tabName}"]`);
        
        if (content) content.classList.add('active');
        if (tab) tab.classList.add('active');

        this.activeTab = tabName;
    }

    displayAnalysis(analysis, containerId) {
        const container = document.getElementById(containerId);
        if (!container) return;

        let html = '';
        html += this._renderQualityMetrics(analysis);
        html += this._renderImageComparison(analysis);
        html += this._renderPixelDifferences(analysis);
        html += this._renderSpatialDistribution(analysis);
        html += this._renderHistogramStats(analysis);
        html += this._renderEntropyAnalysis(analysis);
        html += this._renderLSBAnalysis(analysis);
        html += this._renderAdvancedAnalysis(analysis);
        html += `<div id="histogram-charts"></div>`;
        html += this._renderFullBitPlaneAnalysis(analysis);

        container.innerHTML = html;
        this.toggleResult(containerId, true);

        setTimeout(() => {
            if (analysis.histogram_original && analysis.histogram_stego) {
                this.renderHistograms(analysis);
            }
        }, 200);
    }

    _renderQualityMetrics(a) {
        return `
            <div class="analysis-section">
                <h2>Image Quality Metrics</h2>
                <div class="stats-grid">
                    ${this.renderMetricCard('PSNR', a.psnr?.toFixed(2), 'dB', this.getQualityLabel(a.psnr, 'psnr'))}
                    ${this.renderMetricCard('MSE', a.mse?.toFixed(4), '', this.getQualityLabel(a.mse, 'mse'))}
                    ${this.renderMetricCard('SSIM', a.ssim?.toFixed(4), '', this.getQualityLabel(a.ssim, 'ssim'))}
                </div>
            </div>`;
    }

    _renderImageComparison(a) {
        if (!a.original_image || !a.stego_image) return '';
        return `
            <div class="analysis-section">
                <h2>Image Comparison</h2>
                <div class="image-grid">
                    <div class="image-item">
                        <h4>Original Image</h4>
                        <img src="data:image/png;base64,${a.original_image}" alt="Original" style="max-width: 100%; border: 1px solid #ddd;">
                    </div>
                    <div class="image-item">
                        <h4>Stego Image</h4>
                        <img src="data:image/png;base64,${a.stego_image}" alt="Stego" style="max-width: 100%; border: 1px solid #ddd;">
                    </div>
                </div>
            </div>`;
    }

    _renderPixelDifferences(a) {
        const pd = a.pixel_differences?.overall;
        if (!pd) return '';
        return `
            <div class="analysis-section">
                <h2>Pixel Differences Analysis</h2>
                <div class="stats-grid">
                    ${this.renderMetricCard('Changed Pixels', pd.percent_changed?.toFixed(2), '%')}
                    ${this.renderMetricCard('Max Difference', pd.max_difference, '')}
                    ${this.renderMetricCard('Mean Difference', pd.mean_difference?.toFixed(4), '')}
                    ${this.renderMetricCard('Std Deviation', pd.std_difference?.toFixed(4), '')}
                </div>
            </div>`;
    }

    _renderSpatialDistribution(a) {
        if (!a.spatial_distribution) return '';
        let cards = Object.entries(a.spatial_distribution).map(([region, data]) => {
            const regionName = region.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
            return `
                <div class="stat-card">
                    <h4>${regionName}</h4>
                    <div class="value">${data.percent_changed.toFixed(2)}<span class="unit">%</span></div>
                    <div class="mini-stat" style="margin-top: 10px;">
                        <span class="label">Changes</span>
                        <span class="value">${this.formatNumber(data.changed_pixels)}</span>
                    </div>
                    <div class="mini-stat">
                        <span class="label">Uniformity</span>
                        <span class="value">${data.uniformity_score.toFixed(3)}</span>
                    </div>
                </div>`;
        }).join('');
        
        return `
            <div class="analysis-section">
                <h2>Spatial Distribution</h2>
                <p style="color: #666; font-size: 0.9rem; margin-bottom: 15px;">Analysis of change concentration across image regions (higher uniformity = more evenly distributed changes).</p>
                <div class="stats-grid">${cards}</div>
            </div>`;
    }

    _renderHistogramStats(a) {
        if (!a.histogram_statistics) return '';
        let cards = Object.entries(a.histogram_statistics).map(([ch, s]) => `
            <div class="channel-card">
                <h3>${ch.toUpperCase()} Channel Statistics</h3>
                <div class="stats-grid-small">
                    <div class="mini-stat"><span class="label">Chi-Square</span> <span class="value">${s.chi_square?.toFixed(2)}</span></div>
                    <div class="mini-stat"><span class="label">Correlation</span> <span class="value">${s.correlation?.toFixed(6)}</span></div>
                    <div class="mini-stat"><span class="label">KL Divergence</span> <span class="value">${s.kl_divergence?.toFixed(6)}</span></div>
                    <div class="mini-stat"><span class="label">Entropy (Orig)</span> <span class="value">${s.entropy_original?.toFixed(4)}</span></div>
                    <div class="mini-stat"><span class="label">Entropy (Stego)</span> <span class="value">${s.entropy_stego?.toFixed(4)}</span></div>
                </div>
            </div>`).join('');
        return `<div class="analysis-section"><h2>Histogram Statistical Analysis</h2><div class="channel-analysis">${cards}</div></div>`;
    }

    _renderEntropyAnalysis(a) {
        if (!a.entropy_analysis) return '';
        let cards = Object.entries(a.entropy_analysis).map(([ch, data]) => `
            <div class="stat-card">
                <h4>${ch.toUpperCase()} Channel</h4>
                <div class="mini-stat"><span class="label">Original Entropy</span> <span class="value">${data.original_entropy?.toFixed(4)}</span></div>
                <div class="mini-stat"><span class="label">Stego Entropy</span> <span class="value">${data.stego_entropy?.toFixed(4)}</span></div>
                <div class="mini-stat"><span class="label">Difference</span> <span class="value">${data.entropy_difference?.toFixed(6)}</span></div>
                <div class="mini-stat"><span class="label">% Increase</span> <span class="value">${data.entropy_increase?.toFixed(4)}%</span></div>
            </div>`).join('');
        return `<div class="analysis-section"><h2>🔬 Entropy Analysis</h2><div class="stats-grid">${cards}</div></div>`;
    }

    _renderLSBAnalysis(a) {
        if (!a.lsb_analysis) return '';
        let cards = Object.entries(a.lsb_analysis).map(([ch, bit]) => `
            <div class="channel-card">
                <h3>${ch.toUpperCase()} Channel LSB (Bit 0)</h3>
                <div class="stats-grid-small">
                    <div class="mini-stat"><span class="label">Changes</span> <span class="value">${this.formatNumber(bit.changes)}</span></div>
                    <div class="mini-stat"><span class="label">Percent</span> <span class="value">${bit.percent_changed?.toFixed(2)}%</span></div>
                    <div class="mini-stat"><span class="label">Entropy</span> <span class="value">${bit.entropy?.toFixed(4)}</span></div>
                    <div class="mini-stat"><span class="label">Randomness</span> <span class="value">${bit.randomness_score?.toFixed(4)}</span></div>
                    <div class="mini-stat"><span class="label">Ones Ratio</span> <span class="value">${bit.ones_ratio?.toFixed(4)}</span></div>
                </div>
            </div>`).join('');
        return `<div class="analysis-section"><h2>LSB Analysis (Least Significant Bit)</h2><div class="channel-analysis">${cards}</div></div>`;
    }

    _renderAdvancedAnalysis(a) {
        if (!a.noise_analysis && !a.correlation_analysis) return '';
        let metrics = '';
        
        if (a.noise_analysis) {
            metrics += this.renderMetricCard('SNR', a.noise_analysis.snr?.toFixed(2), 'dB');
            metrics += this.renderMetricCard('Noise Std', a.noise_analysis.std_noise?.toFixed(6), '');
        }
        
        if (a.correlation_analysis) {
            const channels = Object.values(a.correlation_analysis);
            const avg = channels.reduce((sum, c) => sum + (c.avg_corr_change || 0), 0) / channels.length;
            metrics += this.renderMetricCard('Avg Corr Change', avg.toFixed(6));
        }
        
        return `<div class="analysis-section"><h2>Advanced Analysis</h2><div class="stats-grid">${metrics}</div></div>`;
    }

    _renderFullBitPlaneAnalysis(a) {
        if (!a.bit_plane_analysis) return '';
        
        let html = `
            <div class="analysis-section">
                <h2>Complete Bit Plane Analysis</h2>
                <p style="color: #666; margin-bottom: 20px;">
                    Visualization of all 8 bit planes (Bit 0 = LSB, Bit 7 = MSB). 
                    White pixels = 1, Black pixels = 0. LSB planes typically show more randomness when data is hidden.
                </p>`;
        
        ['red', 'green', 'blue'].forEach(channel => {
            const bits = a.bit_plane_analysis[channel];
            if (!bits) return;
            
            html += `
                <div class="channel-card" style="margin-bottom: 30px;">
                    <h3>${channel.toUpperCase()} Channel - All 8 Bit Planes</h3>
                    <div class="bit-planes-grid">`;
            
            for (let i = 0; i < 8; i++) {
                const b = bits[`bit${i}`];
                if (b && b.image) {
                    const bitLabel = i === 0 ? 'LSB' : i === 7 ? 'MSB' : '';
                    html += `
                        <div class="bit-plane-card">
                            <div class="bit-plane-header">
                                <h4>Bit ${i} ${bitLabel}</h4>
                            </div>
                            <div class="bit-plane-image">
                                <img src="data:image/png;base64,${b.image}" 
                                     alt="Bit ${i}" 
                                     style="width: 100%; image-rendering: pixelated; border: 1px solid #ddd;">
                            </div>
                            <div class="bit-plane-stats">
                                <div class="mini-stat">
                                    <span class="label">Changes</span>
                                    <span class="value">${this.formatNumber(b.changes || 0)}</span>
                                </div>
                                <div class="mini-stat">
                                    <span class="label">Changed</span>
                                    <span class="value">${(b.percent_changed || 0).toFixed(2)}%</span>
                                </div>
                                <div class="mini-stat">
                                    <span class="label">Entropy</span>
                                    <span class="value">${(b.entropy || 0).toFixed(4)}</span>
                                </div>
                                <div class="mini-stat">
                                    <span class="label">Chi-Sq</span>
                                    <span class="value">${(b.chi_square || 0).toFixed(2)}</span>
                                </div>
                                <div class="mini-stat">
                                    <span class="label">1s Ratio</span>
                                    <span class="value">${(b.ones_ratio || 0).toFixed(4)}</span>
                                </div>
                            </div>
                        </div>`;
                }
            }
            
            html += `
                    </div>
                </div>`;
        });
        
        return html + `</div>`;
    }

    getQualityLabel(value, metric) {
        if (value === undefined || value === null) return '';
        const thresholds = {
            psnr: { excellent: 40, good: 30 },
            mse: { excellent: 1, good: 5, lowIsBetter: true },
            ssim: { excellent: 0.95, good: 0.90 }
        };
        const t = thresholds[metric];
        if (!t) return '';
        
        if (t.lowIsBetter) {
            if (value < t.excellent) return 'Excellent';
            if (value < t.good) return 'Good';
        } else {
            if (value > t.excellent) return 'Excellent';
            if (value > t.good) return 'Good';
        }
        return 'Poor';
    }

    renderMetricCard(label, value, unit = '', quality = '') {
        return `
            <div class="stat-card">
                <h4>${label}</h4>
                <div class="value">${value || '0'}<span class="unit">${unit}</span></div>
                ${quality ? `<p class="quality-indicator">${quality}</p>` : ''}
            </div>`;
    }

    renderHistograms(analysis) {
        const container = document.getElementById('histogram-charts');
        if (!container) return;
        
        Object.values(this.charts).forEach(chart => {
            try { chart.destroy(); } catch(e) {}
        });
        this.charts = {};
        
        container.innerHTML = '<h2>Histogram Comparison</h2>';

        const colors = {
            red: { original: 'rgb(220, 38, 38)', stego: 'rgba(220, 38, 38, 0.5)' },
            green: { original: 'rgb(22, 163, 74)', stego: 'rgba(22, 163, 74, 0.5)' },
            blue: { original: 'rgb(37, 99, 235)', stego: 'rgba(37, 99, 235, 0.5)' }
        };

        ['red', 'green', 'blue'].forEach(channel => {
            if (!analysis.histogram_original[channel]) return;
            
            const canvasId = `chart-${channel}`;
            const wrapper = document.createElement('div');
            wrapper.className = 'chart-container';
            wrapper.style.height = '350px';
            wrapper.style.marginBottom = '30px';
            wrapper.innerHTML = `<canvas id="${canvasId}"></canvas>`;
            container.appendChild(wrapper);

            const ctx = document.getElementById(canvasId).getContext('2d');
            this.charts[channel] = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: Array.from({length: 256}, (_, i) => i),
                    datasets: [
                        { 
                            label: 'Original', 
                            data: analysis.histogram_original[channel], 
                            borderColor: colors[channel].original, 
                            pointRadius: 0, 
                            fill: false, 
                            tension: 0.1,
                            borderWidth: 2
                        },
                        { 
                            label: 'Stego', 
                            data: analysis.histogram_stego[channel], 
                            borderColor: colors[channel].stego, 
                            borderDash: [5, 5], 
                            pointRadius: 0, 
                            fill: false, 
                            tension: 0.1,
                            borderWidth: 2
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { 
                        title: { 
                            display: true, 
                            text: `${channel.toUpperCase()} Channel Histogram`, 
                            font: { size: 16, weight: 'bold' }
                        },
                        legend: { 
                            position: 'top',
                            labels: { font: { size: 12 } }
                        }
                    },
                    scales: {
                        x: { 
                            title: { display: true, text: 'Pixel Intensity (0-255)' }
                        },
                        y: { 
                            beginAtZero: true,
                            title: { display: true, text: 'Frequency' }
                        }
                    }
                }
            });
        });
    }

    displayCapacity(capacity, containerId) {
        const container = document.getElementById(containerId);
        if (!container) return;
        container.innerHTML = `
            <div class="stats-grid">
                <div class="stat-card"><h4>Dimensions</h4><div class="value">${capacity.width} × ${capacity.height}</div></div>
                <div class="stat-card"><h4>Total Pixels</h4><div class="value">${this.formatNumber(capacity.total_pixels)}</div></div>
                <div class="stat-card"><h4>Max Capacity</h4><div class="value">${this.formatNumber(capacity.max_capacity_chars)}<span class="unit">chars</span></div></div>
                <div class="stat-card"><h4>Practical Bits</h4><div class="value">${this.formatNumber(capacity.practical_max_bits)}<span class="unit">bits</span></div></div>
            </div>`;
        this.toggleResult(containerId, true);
    }

    displayMessage(message, containerId) {
        const container = document.getElementById(containerId);
        if (!container) return;
        container.innerHTML = `
            <div class="result-text">${this.escapeHtml(message)}</div>
            <div class="flex gap-10 mt-20">
                <button class="btn btn-secondary" onclick="navigator.clipboard.writeText(\`${this.escapeHtml(message)}\`).then(() => uiManager.showAlert('decode-alert', 'Copied to clipboard', 'success'))">
                    Copy to Clipboard
                </button>
            </div>`;
        this.toggleResult(containerId, true);
    }

    downloadBlob(blob, filename) {
        if (!blob) return;
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename || 'stego_image.png';
        document.body.appendChild(a);
        a.click();
        setTimeout(() => {
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }, 100);
    }

    clearAll() {
        document.querySelectorAll('.result').forEach(el => el.classList.remove('show'));
        document.querySelectorAll('.alert').forEach(el => el.classList.remove('show'));
        Object.values(this.charts).forEach(chart => {
            try { chart.destroy(); } catch(e) {}
        });
        this.charts = {};
    }

    setButtonState(buttonId, enabled) {
        const btn = document.getElementById(buttonId);
        if (btn) {
            btn.disabled = !enabled;
            btn.style.opacity = enabled ? '1' : '0.5';
            btn.style.cursor = enabled ? 'pointer' : 'not-allowed';
        }
    }
}

const uiManager = new UIManager();
export default uiManager;