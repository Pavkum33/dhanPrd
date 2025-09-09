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
}

// Initialize scanner when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.scanner = new DhanScanner();
});