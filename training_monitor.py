#!/usr/bin/env python3
"""
Training Monitor Web Dashboard
Provides real-time monitoring of Enhanced SIM-ONE training on localhost:5001
"""

import os
import json
import time
import subprocess
import psutil
import re
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template_string, jsonify
import threading

app = Flask(__name__)

class TrainingMonitor:
    def __init__(self):
        self.repo_root = Path(__file__).parent
        self.log_files = [
            self.repo_root / "logs" / "simone_enhanced_training.log",
            self.repo_root / "logs" / "h200_training.log"
        ]

    def get_gpu_status(self):
        """Get GPU status using nvidia-smi"""
        try:
            result = subprocess.run([
                'nvidia-smi', '--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=10)

            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                gpus = []
                for line in lines:
                    parts = line.split(', ')
                    if len(parts) >= 5:
                        gpus.append({
                            'name': parts[0],
                            'memory_used': int(parts[1]),
                            'memory_total': int(parts[2]),
                            'utilization': int(parts[3]),
                            'temperature': int(parts[4])
                        })
                return gpus
        except Exception as e:
            print(f"GPU status error: {e}")
        return []

    def get_training_status(self):
        """Check if training processes are running"""
        training_pids = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = ' '.join(proc.info['cmdline'] or [])
                if 'enhanced_train.py' in cmdline or 'train_all_models.py' in cmdline:
                    training_pids.append({
                        'pid': proc.info['pid'],
                        'name': proc.info['name'],
                        'cmdline': cmdline
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return training_pids

    def parse_training_logs(self):
        """Parse training logs for progress information"""
        progress = {
            'current_epoch': 0,
            'total_epochs': 7,
            'current_step': 0,
            'total_steps': 0,
            'loss': 0.0,
            'learning_rate': 0.0,
            'last_update': None,
            'recent_logs': []
        }

        for log_file in self.log_files:
            if not log_file.exists():
                continue

            try:
                with open(log_file, 'r') as f:
                    lines = f.readlines()

                # Get last 10 lines for recent logs
                progress['recent_logs'].extend(lines[-10:])

                # Parse for training metrics
                for line in reversed(lines[-50:]):  # Check last 50 lines
                    line = line.strip()

                    # Look for epoch information
                    epoch_match = re.search(r'Epoch (\d+)/(\d+)', line)
                    if epoch_match:
                        progress['current_epoch'] = int(epoch_match.group(1))
                        progress['total_epochs'] = int(epoch_match.group(2))

                    # Look for step information
                    step_match = re.search(r'Step (\d+)/(\d+)', line)
                    if step_match:
                        progress['current_step'] = int(step_match.group(1))
                        progress['total_steps'] = int(step_match.group(2))

                    # Look for loss
                    loss_match = re.search(r'Loss: ([\d.]+)', line)
                    if loss_match:
                        progress['loss'] = float(loss_match.group(1))

                    # Look for learning rate
                    lr_match = re.search(r'LR: ([\de.-]+)', line)
                    if lr_match:
                        progress['learning_rate'] = float(lr_match.group(1))

                    # Update timestamp if we found recent data
                    if any([epoch_match, step_match, loss_match, lr_match]):
                        progress['last_update'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        break

            except Exception as e:
                print(f"Error reading {log_file}: {e}")

        # Keep only last 20 log lines
        progress['recent_logs'] = progress['recent_logs'][-20:]

        return progress

    def get_system_info(self):
        """Get system resource information"""
        try:
            return {
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent,
                'uptime': time.time() - psutil.boot_time()
            }
        except Exception as e:
            print(f"System info error: {e}")
            return {}

monitor = TrainingMonitor()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Enhanced SIM-ONE Training Monitor</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #1a1a1a;
            color: #ffffff;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { text-align: center; margin-bottom: 30px; }
        .header h1 { color: #4CAF50; margin: 0; }
        .header p { color: #cccccc; margin: 5px 0; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .card {
            background: #2d2d2d;
            border-radius: 8px;
            padding: 20px;
            border-left: 4px solid #4CAF50;
        }
        .card h3 { margin-top: 0; color: #4CAF50; }
        .metric { margin: 10px 0; }
        .metric-label { color: #cccccc; font-size: 14px; }
        .metric-value { font-size: 18px; font-weight: bold; color: #ffffff; }
        .progress-bar {
            width: 100%;
            height: 20px;
            background-color: #444;
            border-radius: 10px;
            overflow: hidden;
            margin: 5px 0;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #4CAF50, #45a049);
            transition: width 0.3s ease;
        }
        .status-running { color: #4CAF50; }
        .status-stopped { color: #f44336; }
        .logs {
            background: #1a1a1a;
            padding: 15px;
            border-radius: 5px;
            font-family: 'Courier New', monospace;
            font-size: 12px;
            max-height: 300px;
            overflow-y: auto;
            border: 1px solid #444;
        }
        .gpu-card { border-left-color: #FF9800; }
        .gpu-card h3 { color: #FF9800; }
        .system-card { border-left-color: #2196F3; }
        .system-card h3 { color: #2196F3; }
        .last-update { color: #888; font-size: 12px; margin-top: 10px; }
        .auto-refresh { color: #4CAF50; font-size: 12px; }
    </style>
    <script src="https://unpkg.com/htmx.org@1.9.6"></script>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Enhanced SIM-ONE Training Monitor</h1>
            <p>Truth-Leaning AI Training via Singular Source Consistency</p>
            <p class="auto-refresh">🔄 Auto-refreshing every 30 seconds</p>
        </div>

        <div class="grid" hx-get="/api/status" hx-trigger="every 30s" hx-target="#dashboard">
            <div id="dashboard">
                <!-- Content will be loaded here -->
            </div>
        </div>
    </div>

    <script>
        // Initial load
        htmx.ajax('GET', '/api/status', {target:'#dashboard'});

        // Update timestamp
        setInterval(function() {
            document.querySelector('.last-update').textContent = 'Last updated: ' + new Date().toLocaleString();
        }, 1000);
    </script>
</body>
</html>
"""

STATUS_TEMPLATE = """
<div class="card">
    <h3>📊 Training Progress</h3>
    <div class="metric">
        <div class="metric-label">Current Epoch</div>
        <div class="metric-value">{{ progress.current_epoch }} / {{ progress.total_epochs }}</div>
        <div class="progress-bar">
            <div class="progress-fill" style="width: {{ (progress.current_epoch / progress.total_epochs * 100) if progress.total_epochs > 0 else 0 }}%"></div>
        </div>
    </div>

    {% if progress.total_steps > 0 %}
    <div class="metric">
        <div class="metric-label">Current Step</div>
        <div class="metric-value">{{ progress.current_step }} / {{ progress.total_steps }}</div>
        <div class="progress-bar">
            <div class="progress-fill" style="width: {{ (progress.current_step / progress.total_steps * 100) }}%"></div>
        </div>
    </div>
    {% endif %}

    {% if progress.loss > 0 %}
    <div class="metric">
        <div class="metric-label">Loss</div>
        <div class="metric-value">{{ "%.4f"|format(progress.loss) }}</div>
    </div>
    {% endif %}

    {% if progress.learning_rate > 0 %}
    <div class="metric">
        <div class="metric-label">Learning Rate</div>
        <div class="metric-value">{{ "%.2e"|format(progress.learning_rate) }}</div>
    </div>
    {% endif %}

    <div class="metric">
        <div class="metric-label">Training Status</div>
        <div class="metric-value {{ 'status-running' if training_processes else 'status-stopped' }}">
            {{ 'Running' if training_processes else 'Stopped' }}
        </div>
    </div>

    {% if progress.last_update %}
    <div class="last-update">Last update: {{ progress.last_update }}</div>
    {% endif %}
</div>

<div class="card gpu-card">
    <h3>🎮 GPU Status</h3>
    {% if gpus %}
        {% for gpu in gpus %}
        <div class="metric">
            <div class="metric-label">{{ gpu.name }}</div>
            <div class="metric-value">{{ gpu.memory_used }}MB / {{ gpu.memory_total }}MB ({{ "%.1f"|format(gpu.memory_used / gpu.memory_total * 100) }}%)</div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {{ gpu.memory_used / gpu.memory_total * 100 }}%"></div>
            </div>
        </div>
        <div class="metric">
            <div class="metric-label">GPU Utilization</div>
            <div class="metric-value">{{ gpu.utilization }}%</div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {{ gpu.utilization }}%"></div>
            </div>
        </div>
        <div class="metric">
            <div class="metric-label">Temperature</div>
            <div class="metric-value">{{ gpu.temperature }}°C</div>
        </div>
        {% endfor %}
    {% else %}
        <div class="metric-value status-stopped">No GPU detected</div>
    {% endif %}
</div>

<div class="card system-card">
    <h3>💻 System Resources</h3>
    {% if system_info %}
    <div class="metric">
        <div class="metric-label">CPU Usage</div>
        <div class="metric-value">{{ "%.1f"|format(system_info.cpu_percent) }}%</div>
        <div class="progress-bar">
            <div class="progress-fill" style="width: {{ system_info.cpu_percent }}%"></div>
        </div>
    </div>
    <div class="metric">
        <div class="metric-label">Memory Usage</div>
        <div class="metric-value">{{ "%.1f"|format(system_info.memory_percent) }}%</div>
        <div class="progress-bar">
            <div class="progress-fill" style="width: {{ system_info.memory_percent }}%"></div>
        </div>
    </div>
    <div class="metric">
        <div class="metric-label">Disk Usage</div>
        <div class="metric-value">{{ "%.1f"|format(system_info.disk_usage) }}%</div>
        <div class="progress-bar">
            <div class="progress-fill" style="width: {{ system_info.disk_usage }}%"></div>
        </div>
    </div>
    {% endif %}
</div>

<div class="card" style="grid-column: 1 / -1;">
    <h3>📝 Recent Training Logs</h3>
    <div class="logs">
        {% if progress.recent_logs %}
            {% for log_line in progress.recent_logs %}
                {{ log_line|safe }}<br>
            {% endfor %}
        {% else %}
            No recent log entries found.
        {% endif %}
    </div>
</div>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/status')
def api_status():
    progress = monitor.parse_training_logs()
    gpus = monitor.get_gpu_status()
    training_processes = monitor.get_training_status()
    system_info = monitor.get_system_info()

    return render_template_string(STATUS_TEMPLATE,
                                progress=progress,
                                gpus=gpus,
                                training_processes=training_processes,
                                system_info=system_info)

@app.route('/api/data')
def api_data():
    """JSON API endpoint for raw data"""
    return jsonify({
        'progress': monitor.parse_training_logs(),
        'gpus': monitor.get_gpu_status(),
        'training_processes': monitor.get_training_status(),
        'system_info': monitor.get_system_info(),
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("🚀 Starting Enhanced SIM-ONE Training Monitor...")
    print("📊 Dashboard available at: http://localhost:5001")
    print("🔄 Auto-refreshing every 30 seconds")
    print("📡 API endpoint: http://localhost:5001/api/data")

    app.run(host='0.0.0.0', port=5001, debug=False)