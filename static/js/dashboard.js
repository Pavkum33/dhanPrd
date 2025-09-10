// Dashboard JavaScript - Real-time scanner interface

class DhanScanner {
    constructor() {
        this.socket = null;
        this.isConnected = false;
        this.scannerRunning = false;
        this.alerts = [];
        this.watchlist = JSON.parse(localStorage.getItem('watchlist') || '[]');
        this.settings = JSON.parse(localStorage.getItem('settings') || '{}');
        this.uptime = 0;
        this.breakoutChart = null;
        
        this.init();
    }

    init() {
        this.connectWebSocket();
        this.setupEventListeners();
        this.initChart();
        this.loadSettings();
        this.startUptimeCounter();
        this.checkMarketStatus();
        this.initMultiScan();
        
        // Request notification permission
        if ('Notification' in window && Notification.permission === 'default') {
            Notification.requestPermission();
        }
    }

    connectWebSocket() {
        this.socket = io.connect(window.location.origin);
        
        this.socket.on('connect', () => {
            this.isConnected = true;
            this.updateConnectionStatus(true);
            console.log('Connected to server');
        });

        this.socket.on('disconnect', () => {
            this.isConnected = false;
            this.updateConnectionStatus(false);
            console.log('Disconnected from server');
        });

        this.socket.on('scanner_data', (data) => {
            this.updateScannerTable(data);
        });

        this.socket.on('alert', (alert) => {
            this.handleNewAlert(alert);
        });

        this.socket.on('stats', (stats) => {
            this.updateStats(stats);
        });

        this.socket.on('historical_progress', (progress) => {
            this.updateProgress(progress);
        });
        
        // Multi-scan WebSocket events
        this.socket.on('multi_scan_progress', (progress) => {
            this.updateMultiScanProgress(progress.percent, progress.message);
        });
        
        this.socket.on('cpr_results', (data) => {
            this.updateCprResults(data.results);
            this.addActivityLog(`CPR scan: ${data.results.length} narrow CPR stocks found`);
        });
        
        this.socket.on('pivot_results', (data) => {
            this.updatePivotResults(data.results);
            this.addActivityLog(`Pivot scan: ${data.results.length} stocks near pivot`);
        });
        
        this.socket.on('breakout_results', (data) => {
            this.updateMetric('breakoutDetected', data.count);
            this.addActivityLog(`Breakout scan: ${data.count} breakout stocks detected`);
        });
        
        this.socket.on('scan_status_update', (data) => {
            this.updateScannerStatus(data.scanner + 'ScanStatus', data.status, data.message);
        });
        
        this.socket.on('system_status', (data) => {
            if (data.cache_status) this.updateMetric('cacheStatus', data.cache_status);
            if (data.symbols_count) this.updateMetric('symbolsCount', data.symbols_count);
        });
        
        // Live market test WebSocket events
        this.socket.on('live_market_test', (data) => {
            this.handleLiveMarketTestEvent(data);
        });
    }

    setupEventListeners() {
        // Tab navigation
        document.querySelectorAll('.nav-item').forEach(item => {
            item.addEventListener('click', (e) => {
                e.preventDefault();
                this.switchTab(item.dataset.tab);
            });
        });

        // Scanner controls
        document.getElementById('startScanner').addEventListener('click', () => this.startScanner());
        document.getElementById('stopScanner').addEventListener('click', () => this.stopScanner());
        document.getElementById('refreshData').addEventListener('click', () => this.refreshData());
        document.getElementById('fetchHistorical').addEventListener('click', () => this.fetchHistoricalData());
        document.getElementById('testWebSocket').addEventListener('click', () => this.testLiveMarketWebSocket());

        // Filters
        document.getElementById('applyFilters').addEventListener('click', () => this.applyFilters());
        document.getElementById('volumeFactor').addEventListener('input', (e) => {
            e.target.nextElementSibling.textContent = e.target.value + 'x';
        });

        // Alerts
        document.getElementById('clearAlerts').addEventListener('click', () => this.clearAlerts());
        document.getElementById('exportAlerts').addEventListener('click', () => this.exportAlerts());

        // Settings
        document.getElementById('saveSettings').addEventListener('click', () => this.saveSettings());

        // Multi-Scan Controls
        document.getElementById('runAllScans')?.addEventListener('click', () => this.runAllScans());
        document.getElementById('refreshLevels')?.addEventListener('click', () => this.refreshLevels());
        document.getElementById('autoRefresh')?.addEventListener('change', (e) => this.toggleAutoRefresh(e.target.checked));
        
        // Individual Scanner Controls
        document.getElementById('runCprScan')?.addEventListener('click', () => this.runCprScan());
        document.getElementById('runPivotScan')?.addEventListener('click', () => this.runPivotScan());
        document.getElementById('runBreakoutScan')?.addEventListener('click', () => this.runBreakoutScan());

        // Modal
        document.querySelector('.close').addEventListener('click', () => this.closeModal());
        window.addEventListener('click', (e) => {
            if (e.target.classList.contains('modal')) {
                this.closeModal();
            }
        });
    }

    switchTab(tabName) {
        // Update nav items
        document.querySelectorAll('.nav-item').forEach(item => {
            item.classList.remove('active');
        });
        document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');

        // Update content
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.remove('active');
        });
        document.getElementById(`${tabName}-tab`).classList.add('active');
    }

    startScanner() {
        if (!this.scannerRunning) {
            this.socket.emit('start_scanner');
            this.scannerRunning = true;
            document.getElementById('startScanner').disabled = true;
            document.getElementById('stopScanner').disabled = false;
            this.showNotification('Scanner Started', 'F&O scanner is now running');
        }
    }

    stopScanner() {
        if (this.scannerRunning) {
            this.socket.emit('stop_scanner');
            this.scannerRunning = false;
            document.getElementById('startScanner').disabled = false;
            document.getElementById('stopScanner').disabled = true;
            this.showNotification('Scanner Stopped', 'F&O scanner has been stopped');
        }
    }

    refreshData() {
        this.socket.emit('refresh_data');
    }

    async fetchHistoricalData() {
        const btn = document.getElementById('fetchHistorical');
        const originalText = btn.textContent;
        const progressPanel = document.getElementById('progressPanel');
        
        try {
            btn.disabled = true;
            btn.textContent = 'Starting...';
            
            // Show progress panel
            progressPanel.style.display = 'block';
            this.resetProgress();
            
            // Get configuration from UI controls
            const config = {
                fetch_days: parseInt(document.getElementById('fetchDays').value),
                lookback_period: parseInt(document.getElementById('lookback').value),
                volume_factor: parseFloat(document.getElementById('volumeFactor').value),
                price_threshold: parseInt(document.getElementById('minPrice').value) || 50
            };
            
            const response = await fetch('/api/historical/fetch', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(config)
            });
            
            const result = await response.json();
            
            if (response.ok) {
                this.showNotification('Historical Fetch Started', 'Fetching historical data for F&O analysis...');
                this.addProgressLog('Started historical data fetch...', 'info');
            } else {
                this.showNotification('Error', result.error || 'Failed to start historical fetch');
                this.addProgressLog('Error: ' + (result.error || 'Failed to start'), 'error');
                progressPanel.style.display = 'none';
            }
        } catch (error) {
            console.error('Historical fetch error:', error);
            this.showNotification('Error', 'Network error occurred');
            this.addProgressLog('Network error: ' + error.message, 'error');
        } finally {
            if (!progressPanel.style.display || progressPanel.style.display === 'none') {
                btn.disabled = false;
                btn.textContent = originalText;
            }
        }
    }

    resetProgress() {
        document.getElementById('progressBar').style.width = '0%';
        document.getElementById('progressText').textContent = '0%';
        document.getElementById('progressMessage').textContent = 'Starting...';
        document.getElementById('successCount').textContent = '0';
        document.getElementById('failCount').textContent = '0';
        document.getElementById('breakoutCount').textContent = '0';
        document.getElementById('progressLog').innerHTML = '';
    }

    updateProgress(progress) {
        const progressBar = document.getElementById('progressBar');
        const progressText = document.getElementById('progressText');
        const progressMessage = document.getElementById('progressMessage');
        const btn = document.getElementById('fetchHistorical');
        
        // Update progress bar
        if (progress.total > 0) {
            progressBar.style.width = progress.progress_percent + '%';
            progressText.textContent = progress.progress_percent + '%';
        }
        
        // Update message
        progressMessage.textContent = progress.message;
        
        // Update button text
        if (progress.step === 'fetching' || progress.step === 'analyzing') {
            btn.textContent = `Processing... (${progress.current}/${progress.total})`;
        }
        
        // Update stats if available
        if (progress.data) {
            if (progress.data.successful !== undefined) {
                document.getElementById('successCount').textContent = progress.data.successful;
            }
            if (progress.data.failed !== undefined) {
                document.getElementById('failCount').textContent = progress.data.failed;
            }
            if (progress.data.breakouts_found !== undefined) {
                document.getElementById('breakoutCount').textContent = progress.data.breakouts_found;
            }
        }
        
        // Add to log
        let logType = 'info';
        if (progress.step === 'error' || progress.step === 'error_symbol') {
            logType = 'error';
        } else if (progress.step === 'completed_symbol') {
            logType = 'success';
        }
        
        this.addProgressLog(progress.message, logType);
        
        // Handle completion
        if (progress.step === 'summary' || progress.step === 'error') {
            setTimeout(() => {
                btn.disabled = false;
                btn.textContent = 'Fetch Historical';
                if (progress.step === 'summary') {
                    this.showNotification('Analysis Complete', 
                        `Processed ${progress.data.successful} securities, found ${progress.data.breakouts_found} breakouts`);
                }
            }, 2000);
        }
    }

    addProgressLog(message, type = 'info') {
        const log = document.getElementById('progressLog');
        const entry = document.createElement('div');
        entry.className = `log-entry ${type}`;
        entry.innerHTML = `<span class="timestamp">${new Date().toLocaleTimeString()}</span> ${message}`;
        log.appendChild(entry);
        log.scrollTop = log.scrollHeight;
    }


    applyFilters() {
        const filters = {
            minPrice: document.getElementById('minPrice').value,
            maxPrice: document.getElementById('maxPrice').value,
            volumeFactor: document.getElementById('volumeFactor').value,
            lookback: document.getElementById('lookback').value
        };
        
        this.socket.emit('update_filters', filters);
        this.showNotification('Filters Applied', 'Scanner filters have been updated');
    }

    testLiveMarketWebSocket() {
        // Test live market WebSocket functionality with RELIANCE
        const btn = document.getElementById('testWebSocket');
        btn.disabled = true;
        btn.textContent = 'Testing...';
        
        this.addActivityLog('🚀 Starting live market WebSocket test for RELIANCE...', 'info');
        
        // Call the API endpoint
        fetch('/api/websocket/live-market-test', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                symbol: 'RELIANCE',
                duration: 20  // 20 second test
            })
        })
        .then(response => response.json())
        .then(data => {
            console.log('Live Market Test Response:', data);
            this.addActivityLog(`📡 ${data.message || 'Test started'}`, 'success');
            this.addActivityLog(`📊 WebSocket events: ${data.websocket_events || 'live_market_test'}`, 'info');
            this.addActivityLog(`⏱️ Duration: ${data.duration || '20 seconds'}`, 'info');
            
            // Re-enable button after 25 seconds
            setTimeout(() => {
                btn.disabled = false;
                btn.textContent = 'Live Market Test';
                this.addActivityLog('✅ Live market test completed!', 'success');
            }, 25000);
        })
        .catch(error => {
            console.error('Live Market Test Error:', error);
            this.addActivityLog(`❌ Test failed: ${error.message}`, 'error');
            btn.disabled = false;
            btn.textContent = 'Live Market Test';
        });
    }

    updateScannerTable(data) {
        const tbody = document.getElementById('scannerData');
        tbody.innerHTML = '';
        
        data.forEach(item => {
            const row = document.createElement('tr');
            const changeClass = item.change >= 0 ? 'positive' : 'negative';
            const signalClass = item.signal === 'BREAKOUT' ? 'positive' : '';
            
            row.innerHTML = `
                <td><strong>${item.symbol}</strong></td>
                <td>${item.ltp.toFixed(2)}</td>
                <td class="${changeClass}">${item.change >= 0 ? '+' : ''}${item.change.toFixed(2)}%</td>
                <td>${this.formatVolume(item.volume)}</td>
                <td>${item.resistance ? item.resistance.toFixed(2) : '-'}</td>
                <td>${item.ema8 ? item.ema8.toFixed(2) : '-'}/${item.ema13 ? item.ema13.toFixed(2) : '-'}</td>
                <td class="${signalClass}">${item.signal || '-'}</td>
                <td>
                    <button class="btn btn-sm" onclick="scanner.addToWatchlist('${item.symbol}')">⭐</button>
                    <button class="btn btn-sm" onclick="scanner.showDetails('${item.symbol}')">📊</button>
                </td>
            `;
            tbody.appendChild(row);
        });
    }

    handleNewAlert(alert) {
        this.alerts.unshift(alert);
        this.updateAlertCount();
        this.addToAlertFeed(alert);
        this.updateAlertsList();
        
        // Show notification
        if (this.settings.enableNotifications) {
            this.showNotification('Breakout Alert!', `${alert.symbol}: ${alert.message}`);
        }
        
        // Play sound
        if (this.settings.enableSound) {
            this.playAlertSound();
        }
        
        // Update chart
        this.updateBreakoutChart(alert);
    }

    updateAlertCount() {
        document.getElementById('alertCount').textContent = this.alerts.length;
        document.getElementById('totalAlerts').textContent = this.alerts.length;
    }

    addToAlertFeed(alert) {
        const feed = document.getElementById('alertFeed');
        const alertItem = document.createElement('div');
        alertItem.className = 'alert-item breakout';
        alertItem.innerHTML = `
            <div><strong>${alert.symbol}</strong></div>
            <div>${alert.message}</div>
            <div class="alert-time">${new Date(alert.timestamp).toLocaleTimeString()}</div>
        `;
        
        feed.insertBefore(alertItem, feed.firstChild);
        
        // Keep only last 10 alerts in feed
        while (feed.children.length > 10) {
            feed.removeChild(feed.lastChild);
        }
    }

    updateAlertsList() {
        const list = document.getElementById('alertsList');
        list.innerHTML = '';
        
        this.alerts.forEach(alert => {
            const card = document.createElement('div');
            card.className = 'alert-card';
            card.innerHTML = `
                <h4>${alert.symbol}</h4>
                <p>${alert.message}</p>
                <small>${new Date(alert.timestamp).toLocaleString()}</small>
            `;
            list.appendChild(card);
        });
    }

    clearAlerts() {
        this.alerts = [];
        this.updateAlertCount();
        this.updateAlertsList();
        document.getElementById('alertFeed').innerHTML = '';
    }

    handleLiveMarketTestEvent(data) {
        // Handle live market test WebSocket events
        console.log('🚀 Live Market Test Event:', data);
        
        // Add to activity log with special formatting
        const timestamp = new Date(data.timestamp).toLocaleTimeString();
        const step = data.step || 'unknown';
        const message = data.message || 'No message';
        
        // Color coding based on step
        let logClass = 'info';
        if (step === 'error' || step === 'data_error' || step === 'symbol_error') {
            logClass = 'error';
        } else if (step === 'data_success' || step === 'completed') {
            logClass = 'success';
        } else if (step === 'live_update') {
            logClass = 'update';
        }
        
        this.addActivityLog(`[${step.toUpperCase()}] ${message}`, logClass);
        
        // Show detailed data if available
        if (data.data) {
            const dataStr = JSON.stringify(data.data, null, 2);
            console.log('📊 Live Market Data:', data.data);
            
            // Add data details to log for key steps
            if (step === 'data_success') {
                this.addActivityLog(`📊 ${data.data.symbol}: ₹${data.data.latest_close} (${data.data.data_points} days, Vol: ${data.data.latest_volume})`, 'data');
            } else if (step === 'live_update') {
                this.addActivityLog(`📈 Update #${data.data.update_number}: ₹${data.data.simulated_price} (was ₹${data.data.original_close})`, 'update');
            }
        }
        
        // Show summary for completion
        if (step === 'completed' && data.summary) {
            const summary = data.summary;
            this.addActivityLog(`✅ Test Summary: ${summary.total_updates} updates in ${summary.duration}s - WebSocket: ${summary.websocket_status}`, 'success');
        }
        
        // Update progress bar if needed
        if (step === 'live_update' && data.data) {
            const updateNum = data.data.update_number || 0;
            const progressPercent = Math.min((updateNum / 10) * 100, 100); // Max 10 updates
            this.updateProgressBar('Live Market Test', progressPercent, `Update ${updateNum}/10`);
        }
    }

    exportAlerts() {
        const csv = this.convertToCSV(this.alerts);
        this.downloadCSV(csv, 'alerts_export.csv');
    }

    convertToCSV(data) {
        if (data.length === 0) return '';
        
        const headers = Object.keys(data[0]);
        const csvHeaders = headers.join(',');
        const csvRows = data.map(row => 
            headers.map(header => JSON.stringify(row[header] || '')).join(',')
        );
        
        return csvHeaders + '\n' + csvRows.join('\n');
    }

    downloadCSV(csv, filename) {
        const blob = new Blob([csv], { type: 'text/csv' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        a.click();
        window.URL.revokeObjectURL(url);
    }

    addToWatchlist(symbol) {
        if (!this.watchlist.includes(symbol)) {
            this.watchlist.push(symbol);
            localStorage.setItem('watchlist', JSON.stringify(this.watchlist));
            this.updateWatchlistDisplay();
            this.showNotification('Added to Watchlist', `${symbol} has been added to your watchlist`);
        }
    }

    updateWatchlistDisplay() {
        const grid = document.getElementById('watchlistGrid');
        grid.innerHTML = '';
        
        this.watchlist.forEach(symbol => {
            const card = document.createElement('div');
            card.className = 'watchlist-card';
            card.innerHTML = `
                <h4>${symbol}</h4>
                <div>Loading...</div>
                <button class="btn btn-sm btn-danger" onclick="scanner.removeFromWatchlist('${symbol}')">Remove</button>
            `;
            grid.appendChild(card);
        });
    }

    removeFromWatchlist(symbol) {
        this.watchlist = this.watchlist.filter(s => s !== symbol);
        localStorage.setItem('watchlist', JSON.stringify(this.watchlist));
        this.updateWatchlistDisplay();
    }

    showDetails(symbol) {
        const modal = document.getElementById('symbolModal');
        const modalSymbol = document.getElementById('modalSymbol');
        const modalBody = document.getElementById('modalBody');
        
        modalSymbol.textContent = symbol;
        modalBody.innerHTML = '<p>Loading details...</p>';
        modal.style.display = 'block';
        
        // Request details from server
        this.socket.emit('get_symbol_details', symbol, (details) => {
            modalBody.innerHTML = `
                <div>
                    <p><strong>Last Price:</strong> ${details.ltp}</p>
                    <p><strong>Volume:</strong> ${details.volume}</p>
                    <p><strong>Resistance:</strong> ${details.resistance}</p>
                    <p><strong>EMA8:</strong> ${details.ema8}</p>
                    <p><strong>EMA13:</strong> ${details.ema13}</p>
                </div>
            `;
        });
    }

    closeModal() {
        document.getElementById('symbolModal').style.display = 'none';
    }

    updateConnectionStatus(connected) {
        const status = document.getElementById('connectionStatus');
        const dot = status.querySelector('.status-dot');
        const text = status.querySelector('.status-text');
        
        if (connected) {
            dot.classList.add('connected');
            text.textContent = 'Connected';
        } else {
            dot.classList.remove('connected');
            text.textContent = 'Disconnected';
        }
    }

    checkMarketStatus() {
        const now = new Date();
        const hours = now.getHours();
        const minutes = now.getMinutes();
        const day = now.getDay();
        
        const marketStatus = document.getElementById('marketStatus');
        const dot = marketStatus.querySelector('.market-dot');
        const text = marketStatus.querySelector('.market-text');
        
        // Check if market is open (9:15 AM to 3:30 PM IST, Mon-Fri)
        const isWeekday = day >= 1 && day <= 5;
        const isMarketHours = (hours === 9 && minutes >= 15) || 
                              (hours > 9 && hours < 15) || 
                              (hours === 15 && minutes <= 30);
        
        if (isWeekday && isMarketHours) {
            dot.classList.add('open');
            text.textContent = 'Market Open';
        } else {
            dot.classList.remove('open');
            text.textContent = 'Market Closed';
        }
        
        // Check again every minute
        setTimeout(() => this.checkMarketStatus(), 60000);
    }

    updateStats(stats) {
        document.getElementById('activeSymbols').textContent = stats.activeSymbols || 0;
    }

    startUptimeCounter() {
        setInterval(() => {
            this.uptime++;
            const hours = Math.floor(this.uptime / 3600);
            const minutes = Math.floor((this.uptime % 3600) / 60);
            const seconds = this.uptime % 60;
            
            document.getElementById('uptime').textContent = 
                `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
        }, 1000);
    }

    loadSettings() {
        const defaults = {
            enableNotifications: true,
            enableSound: true,
            batchSize: 200,
            histDays: 60,
            emaShort: 8,
            emaLong: 13
        };
        
        this.settings = { ...defaults, ...this.settings };
        
        // Apply to UI
        document.getElementById('enableNotifications').checked = this.settings.enableNotifications;
        document.getElementById('enableSound').checked = this.settings.enableSound;
        document.getElementById('batchSize').value = this.settings.batchSize;
        document.getElementById('histDays').value = this.settings.histDays;
        document.getElementById('emaShort').value = this.settings.emaShort;
        document.getElementById('emaLong').value = this.settings.emaLong;
    }

    saveSettings() {
        this.settings = {
            enableNotifications: document.getElementById('enableNotifications').checked,
            enableSound: document.getElementById('enableSound').checked,
            batchSize: parseInt(document.getElementById('batchSize').value),
            histDays: parseInt(document.getElementById('histDays').value),
            emaShort: parseInt(document.getElementById('emaShort').value),
            emaLong: parseInt(document.getElementById('emaLong').value)
        };
        
        localStorage.setItem('settings', JSON.stringify(this.settings));
        this.socket.emit('update_settings', this.settings);
        this.showNotification('Settings Saved', 'Your settings have been saved successfully');
    }

    showNotification(title, body) {
        if ('Notification' in window && Notification.permission === 'granted' && this.settings.enableNotifications) {
            new Notification(title, {
                body: body,
                icon: '/static/icon.png',
                badge: '/static/badge.png'
            });
        }
    }

    playAlertSound() {
        const audio = new Audio('/static/alert.mp3');
        audio.play().catch(e => console.log('Could not play alert sound:', e));
    }

    formatVolume(volume) {
        if (volume >= 1000000) {
            return (volume / 1000000).toFixed(2) + 'M';
        } else if (volume >= 1000) {
            return (volume / 1000).toFixed(2) + 'K';
        }
        return volume.toString();
    }

    initChart() {
        const ctx = document.getElementById('breakoutChart').getContext('2d');
        this.breakoutChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Breakouts',
                    data: [],
                    borderColor: '#00ff88',
                    backgroundColor: 'rgba(0, 255, 136, 0.1)',
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    x: {
                        display: false
                    },
                    y: {
                        beginAtZero: true,
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        ticks: {
                            color: '#a0a0a0'
                        }
                    }
                }
            }
        });
    }

    updateBreakoutChart(alert) {
        const chart = this.breakoutChart;
        const time = new Date(alert.timestamp).toLocaleTimeString();
        
        chart.data.labels.push(time);
        chart.data.datasets[0].data.push(1);
        
        // Keep only last 20 points
        if (chart.data.labels.length > 20) {
            chart.data.labels.shift();
            chart.data.datasets[0].data.shift();
        }
        
        chart.update();
    }

    // Multi-Scan Initialization
    initMultiScan() {
        // Initialize system status
        this.updateSystemStatus();
        this.addActivityLog('Multi-scan system initialized');
        this.updateScanSummary();
        
        // Update system uptime in the multi-scan card
        const systemUptimeElement = document.getElementById('systemUptime');
        if (systemUptimeElement) {
            // Share uptime with the scanner sidebar
            const mainUptime = document.getElementById('uptime');
            if (mainUptime) {
                systemUptimeElement.textContent = mainUptime.textContent;
            }
        }
    }
    
    updateSystemStatus() {
        // Update cache status
        this.checkCacheStatus();
        
        // Update symbols count
        this.updateSymbolsCount();
    }
    
    async checkCacheStatus() {
        try {
            const response = await fetch('/api/cache/status');
            const data = await response.json();
            
            const cacheBackend = data.current_backend || 'SQLite';
            this.updateMetric('cacheStatus', cacheBackend);
            
        } catch (error) {
            console.error('Cache status error:', error);
            this.updateMetric('cacheStatus', 'Error');
        }
    }
    
    async updateSymbolsCount() {
        try {
            const response = await fetch('/api/debug/instruments');
            const data = await response.json();
            
            const count = data.active_futures?.length || 0;
            this.updateMetric('symbolsCount', count);
            
        } catch (error) {
            console.error('Symbols count error:', error);
            this.updateMetric('symbolsCount', '0');
        }
    }

    // Multi-Scan Methods
    async runAllScans() {
        console.log('Running all scans...');
        this.showMultiScanProgress(true);
        this.updateMultiScanProgress(0, 'Starting all scans...');
        
        let completedScans = 0;
        const totalScans = 3;
        
        const updateProgress = () => {
            completedScans++;
            const percent = Math.round((completedScans / totalScans) * 100);
            this.updateMultiScanProgress(percent, `Completed ${completedScans}/${totalScans} scans...`);
        };
        
        try {
            // Run scans in parallel with individual progress tracking
            const promises = [
                this.runCprScan().then(() => updateProgress()),
                this.runPivotScan().then(() => updateProgress()),
                this.runBreakoutScan().then(() => updateProgress())
            ];
            
            // Use Promise.allSettled to handle individual failures gracefully
            const results = await Promise.allSettled(promises);
            
            // Check results
            const successful = results.filter(r => r.status === 'fulfilled').length;
            const failed = results.filter(r => r.status === 'rejected').length;
            
            if (failed > 0) {
                this.updateMultiScanProgress(100, `Completed with ${failed} failures`);
                this.addActivityLog(`Multi-scan completed: ${successful} successful, ${failed} failed`);
            } else {
                this.updateMultiScanProgress(100, 'All scans completed successfully');
                this.addActivityLog('All scans completed successfully');
            }
            
            this.updateScanSummary();
            
        } catch (error) {
            console.error('Error running all scans:', error);
            this.addActivityLog('Critical error in multi-scan: ' + error.message);
        } finally {
            setTimeout(() => this.showMultiScanProgress(false), 2000);
        }
    }
    
    async runCprScan() {
        console.log('Running CPR scan...');
        this.updateScannerStatus('cprScanStatus', 'scanning', 'Scanning...');
        
        try {
            const response = await fetch('/api/levels/narrow-cpr-railway?month=' + this.getCurrentMonth());
            const data = await response.json();
            
            this.updateCprResults(data);
            this.updateScannerStatus('cprScanStatus', 'active', 'Complete');
            this.updateMetric('cprLastUpdate', this.getCurrentTime());
            
        } catch (error) {
            console.error('CPR scan error:', error);
            this.updateScannerStatus('cprScanStatus', 'error', 'Error');
            this.addActivityLog('CPR scan failed: ' + error.message);
        }
    }
    
    async runPivotScan() {
        console.log('Running Pivot scan...');
        this.updateScannerStatus('pivotScanStatus', 'scanning', 'Scanning...');
        
        try {
            // Get current prices from scanner data (if available)
            const currentPrices = this.getCurrentPricesFromTable();
            const symbols = Object.keys(currentPrices);
            
            const response = await fetch('/api/levels/near-pivot', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    current_prices: currentPrices,
                    symbols: symbols,
                    month: this.getCurrentMonth()
                })
            });
            
            const data = await response.json();
            this.updatePivotResults(data);
            this.updateScannerStatus('pivotScanStatus', 'active', 'Complete');
            this.updateMetric('pivotLastUpdate', this.getCurrentTime());
            
        } catch (error) {
            console.error('Pivot scan error:', error);
            this.updateScannerStatus('pivotScanStatus', 'error', 'Error');
            this.addActivityLog('Pivot scan failed: ' + error.message);
        }
    }
    
    async runBreakoutScan() {
        console.log('Running Breakout scan...');
        this.updateScannerStatus('breakoutScanStatus', 'scanning', 'Scanning...');
        
        try {
            // Use existing historical data functionality
            await this.fetchHistoricalData();
            
            // Get breakout results from scanner table
            const breakoutCount = this.getBreakoutCountFromTable();
            
            this.updateMetric('breakoutDetected', breakoutCount);
            this.updateScannerStatus('breakoutScanStatus', 'active', 'Complete');
            this.updateMetric('breakoutLastUpdate', this.getCurrentTime());
            
        } catch (error) {
            console.error('Breakout scan error:', error);
            this.updateScannerStatus('breakoutScanStatus', 'error', 'Error');
            this.addActivityLog('Breakout scan failed: ' + error.message);
        }
    }
    
    async refreshLevels() {
        console.log('Refreshing monthly levels...');
        this.addActivityLog('Refreshing monthly levels...');
        
        try {
            const response = await fetch('/api/levels/calculate', { method: 'POST' });
            const data = await response.json();
            
            if (data.status === 'running') {
                this.addActivityLog('Monthly level calculation started');
                
                // Poll for completion
                this.pollForLevelsCompletion();
            }
            
        } catch (error) {
            console.error('Error refreshing levels:', error);
            this.addActivityLog('Failed to refresh levels: ' + error.message);
        }
    }
    
    toggleAutoRefresh(enabled) {
        if (enabled) {
            this.autoRefreshInterval = setInterval(() => {
                this.runAllScans();
            }, 30000); // 30 seconds
            this.addActivityLog('Auto refresh enabled (30s)');
        } else {
            if (this.autoRefreshInterval) {
                clearInterval(this.autoRefreshInterval);
                this.autoRefreshInterval = null;
            }
            this.addActivityLog('Auto refresh disabled');
        }
    }
    
    // Helper Methods
    updateScannerStatus(statusId, className, text) {
        const statusElement = document.getElementById(statusId);
        if (statusElement) {
            statusElement.className = 'status-badge ' + className;
            statusElement.textContent = text;
        }
    }
    
    updateMetric(metricId, value) {
        const element = document.getElementById(metricId);
        if (element) {
            element.textContent = value;
        }
    }
    
    updateCprResults(data) {
        const detected = data.length || 0;
        this.updateMetric('cprDetected', detected);
        
        const resultsList = document.getElementById('cprResultsList');
        const resultsCount = document.getElementById('cprResultsCount');
        const controlsDiv = document.getElementById('cprResultsControls');
        
        if (resultsList && resultsCount) {
            resultsCount.textContent = detected + ' found';
            
            // Show/hide controls based on data size
            if (controlsDiv) {
                controlsDiv.style.display = detected > 5 ? 'flex' : 'none';
            }
            
            if (detected === 0) {
                resultsList.innerHTML = '<div class="no-results">No narrow CPR stocks detected</div>';
            } else {
                // Store raw data for filtering/sorting
                this.cprData = data;
                this.renderPaginatedResults('cpr', data);
                this.setupDataControls('cpr');
            }
        }
    }
    
    updatePivotResults(data) {
        const detected = data.length || 0;
        this.updateMetric('pivotDetected', detected);
        
        const resultsList = document.getElementById('pivotResultsList');
        const resultsCount = document.getElementById('pivotResultsCount');
        const controlsDiv = document.getElementById('pivotResultsControls');
        
        if (resultsList && resultsCount) {
            resultsCount.textContent = detected + ' found';
            
            // Show/hide controls based on data size
            if (controlsDiv) {
                controlsDiv.style.display = detected > 5 ? 'flex' : 'none';
            }
            
            if (detected === 0) {
                resultsList.innerHTML = '<div class="no-results">No stocks near pivot detected</div>';
            } else {
                // Store raw data for filtering/sorting
                this.pivotData = data;
                this.renderPaginatedResults('pivot', data);
                this.setupDataControls('pivot');
            }
        }
    }
    
    showMultiScanProgress(show) {
        const progressElement = document.getElementById('summaryProgress');
        if (progressElement) {
            progressElement.style.display = show ? 'block' : 'none';
        }
    }
    
    updateMultiScanProgress(percent, message) {
        const progressBar = document.getElementById('multiScanProgressBar');
        const progressText = document.getElementById('multiScanProgressText');
        const progressMessage = document.getElementById('multiScanProgressMessage');
        
        if (progressBar) progressBar.style.width = percent + '%';
        if (progressText) progressText.textContent = percent + '%';
        if (progressMessage) progressMessage.textContent = message;
    }
    
    addActivityLog(message) {
        const activityLog = document.getElementById('activityLog');
        if (activityLog) {
            const time = new Date().toLocaleTimeString();
            const item = document.createElement('div');
            item.className = 'activity-item';
            item.innerHTML = `
                <span class="activity-time">${time}</span>
                <span class="activity-message">${message}</span>
            `;
            
            activityLog.insertBefore(item, activityLog.firstChild);
            
            // Keep only last 10 items
            while (activityLog.children.length > 10) {
                activityLog.removeChild(activityLog.lastChild);
            }
        }
    }
    
    getCurrentMonth() {
        const now = new Date();
        return now.getFullYear() + '-' + String(now.getMonth() + 1).padStart(2, '0');
    }
    
    getCurrentTime() {
        return new Date().toLocaleTimeString();
    }
    
    getCurrentPricesFromTable() {
        const prices = {};
        const rows = document.querySelectorAll('#scannerTable tbody tr');
        
        rows.forEach(row => {
            const cells = row.querySelectorAll('td');
            if (cells.length > 1) {
                const symbol = cells[0].textContent.trim();
                const price = parseFloat(cells[1].textContent.trim());
                if (symbol && !isNaN(price)) {
                    prices[symbol] = price;
                }
            }
        });
        
        return prices;
    }
    
    getBreakoutCountFromTable() {
        let count = 0;
        const rows = document.querySelectorAll('#scannerTable tbody tr');
        
        rows.forEach(row => {
            const cells = row.querySelectorAll('td');
            if (cells.length > 6 && cells[6].textContent.includes('BREAKOUT')) {
                count++;
            }
        });
        
        return count;
    }
    
    updateScanSummary() {
        const totalScans = 3; // CPR + Pivot + Breakout
        const activeAlerts = this.alerts.length;
        const successRate = '100%'; // TODO: Calculate based on actual success/failure
        
        this.updateMetric('totalScans', totalScans);
        this.updateMetric('activeAlerts', activeAlerts);
        this.updateMetric('successRate', successRate);
    }
    
    async pollForLevelsCompletion() {
        const maxPolls = 30; // 5 minutes max
        let polls = 0;
        
        const pollInterval = setInterval(async () => {
            polls++;
            
            try {
                const response = await fetch('/api/levels/premarket-summary');
                const data = await response.json();
                
                if (data.status === 'completed' || polls >= maxPolls) {
                    clearInterval(pollInterval);
                    
                    if (data.status === 'completed') {
                        this.addActivityLog(`Levels updated: ${data.success_count} symbols`);
                    } else {
                        this.addActivityLog('Level calculation timeout');
                    }
                }
                
            } catch (error) {
                console.error('Polling error:', error);
                clearInterval(pollInterval);
            }
        }, 10000); // Poll every 10 seconds
    }
    
    // Professional Data Handling Methods
    renderPaginatedResults(scannerType, data, page = 1, pageSize = 10) {
        const resultsList = document.getElementById(`${scannerType}ResultsList`);
        const paginationDiv = document.getElementById(`${scannerType}Pagination`);
        
        if (!resultsList || !data || data.length === 0) return;
        
        // Calculate pagination
        const totalPages = Math.ceil(data.length / pageSize);
        const startIndex = (page - 1) * pageSize;
        const endIndex = Math.min(startIndex + pageSize, data.length);
        const pageData = data.slice(startIndex, endIndex);
        
        // Show/hide pagination controls
        if (paginationDiv) {
            paginationDiv.style.display = totalPages > 1 ? 'flex' : 'none';
            
            if (totalPages > 1) {
                const prevBtn = document.getElementById(`${scannerType}PrevPage`);
                const nextBtn = document.getElementById(`${scannerType}NextPage`);
                const pageInfo = document.getElementById(`${scannerType}PageInfo`);
                
                if (prevBtn) prevBtn.disabled = page <= 1;
                if (nextBtn) nextBtn.disabled = page >= totalPages;
                if (pageInfo) pageInfo.textContent = `${page} / ${totalPages}`;
            }
        }
        
        // Render results with professional features
        resultsList.innerHTML = pageData.map(item => {
            return this.renderResultItem(scannerType, item);
        }).join('');
        
        // Store pagination state
        this[`${scannerType}CurrentPage`] = page;
        this[`${scannerType}PageSize`] = pageSize;
    }
    
    renderResultItem(scannerType, item) {
        let priorityClass = '';
        let badges = '';
        let details = '';
        
        // Determine priority and features based on scanner type
        if (scannerType === 'cpr') {
            const width = item.cpr_width_percent || 0;
            priorityClass = width < 0.2 ? 'priority-high' : width < 0.3 ? 'priority-medium' : 'priority-low';
            
            badges = `
                <div class="result-badges">
                    ${width < 0.2 ? '<span class="result-badge volume-high">Ultra Narrow</span>' : ''}
                    ${item.volume_above_avg ? '<span class="result-badge volume-high">High Vol</span>' : ''}
                </div>
            `;
            
            details = `
                <div class="result-detail">
                    <span>Pivot: ₹${item.pivot ? item.pivot.toFixed(2) : 'N/A'}</span>
                    <span class="result-change ${item.change >= 0 ? 'positive' : 'negative'}">
                        ${item.change >= 0 ? '+' : ''}${item.change ? item.change.toFixed(2) : '0.00'}%
                    </span>
                </div>
            `;
            
            return `
                <div class="result-item ${priorityClass}">
                    <div>
                        <span class="result-symbol">${item.symbol}</span>
                        <span class="result-value narrow">${width.toFixed(3)}%</span>
                    </div>
                    ${badges}
                    ${details}
                </div>
            `;
            
        } else if (scannerType === 'pivot') {
            const proximity = Math.abs(item.proximity_percent || 0);
            priorityClass = proximity < 0.1 ? 'priority-high' : proximity < 0.5 ? 'priority-medium' : 'priority-low';
            
            badges = `
                <div class="result-badges">
                    ${proximity < 0.1 ? '<span class="result-badge price-action">Very Close</span>' : ''}
                    ${item.trend_strength === 'strong' ? '<span class="result-badge trend-strong">Strong Trend</span>' : ''}
                </div>
            `;
            
            details = `
                <div class="result-detail">
                    <span>Price: ₹${item.current_price ? item.current_price.toFixed(2) : 'N/A'}</span>
                    <span>Pivot: ₹${item.pivot ? item.pivot.toFixed(2) : 'N/A'}</span>
                </div>
            `;
            
            return `
                <div class="result-item ${priorityClass}">
                    <div>
                        <span class="result-symbol">${item.symbol}</span>
                        <span class="result-value near-pivot">${item.proximity_percent.toFixed(3)}%</span>
                    </div>
                    ${badges}
                    ${details}
                </div>
            `;
        }
        
        // Fallback for other scanner types
        return `
            <div class="result-item">
                <span class="result-symbol">${item.symbol}</span>
                <span class="result-value">${item.value || 'N/A'}</span>
            </div>
        `;
    }
    
    setupDataControls(scannerType) {
        const searchInput = document.getElementById(`${scannerType}Search`);
        const sortSelect = document.getElementById(`${scannerType}Sort`);
        const densityBtns = document.querySelectorAll(`#${scannerType}ResultsControls .density-btn`);
        const resultsList = document.getElementById(`${scannerType}ResultsList`);
        
        // Search functionality
        if (searchInput) {
            searchInput.addEventListener('input', (e) => {
                const query = e.target.value.toLowerCase();
                const data = this[`${scannerType}Data`] || [];
                const filtered = data.filter(item => 
                    item.symbol.toLowerCase().includes(query)
                );
                this.renderPaginatedResults(scannerType, filtered, 1);
            });
        }
        
        // Sort functionality
        if (sortSelect) {
            sortSelect.addEventListener('change', (e) => {
                const sortBy = e.target.value;
                const data = [...(this[`${scannerType}Data`] || [])];
                
                data.sort((a, b) => {
                    if (sortBy === 'symbol') {
                        return a.symbol.localeCompare(b.symbol);
                    } else if (sortBy === 'width') {
                        return (a.cpr_width_percent || 0) - (b.cpr_width_percent || 0);
                    } else if (sortBy === 'proximity') {
                        return Math.abs(a.proximity_percent || 0) - Math.abs(b.proximity_percent || 0);
                    } else if (sortBy === 'volume') {
                        return (b.volume || 0) - (a.volume || 0);
                    }
                    return 0;
                });
                
                this.renderPaginatedResults(scannerType, data, 1);
            });
        }
        
        // Density controls
        densityBtns.forEach(btn => {
            btn.addEventListener('click', (e) => {
                densityBtns.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                
                const density = btn.dataset.density;
                if (resultsList) {
                    resultsList.className = resultsList.className.replace(/\b(compact|ultra-compact)\b/g, '');
                    
                    if (density === 'compact') {
                        resultsList.classList.add('compact');
                    } else if (density === 'ultra') {
                        resultsList.classList.add('ultra-compact');
                    }
                }
            });
        });
        
        // Pagination controls
        const prevBtn = document.getElementById(`${scannerType}PrevPage`);
        const nextBtn = document.getElementById(`${scannerType}NextPage`);
        
        if (prevBtn) {
            prevBtn.addEventListener('click', () => {
                const currentPage = this[`${scannerType}CurrentPage`] || 1;
                if (currentPage > 1) {
                    const data = this[`${scannerType}Data`] || [];
                    this.renderPaginatedResults(scannerType, data, currentPage - 1);
                }
            });
        }
        
        if (nextBtn) {
            nextBtn.addEventListener('click', () => {
                const currentPage = this[`${scannerType}CurrentPage`] || 1;
                const data = this[`${scannerType}Data`] || [];
                const pageSize = this[`${scannerType}PageSize`] || 10;
                const totalPages = Math.ceil(data.length / pageSize);
                
                if (currentPage < totalPages) {
                    this.renderPaginatedResults(scannerType, data, currentPage + 1);
                }
            });
        }
    }
}

// Initialize scanner when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.scanner = new DhanScanner();
});