"""
Burpee Tracker with Real-time Video Feed
Robust body marker tracking for accurate burpee counting
"""

import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, Response, render_template_string, request, jsonify
import time

app = Flask(__name__)


class Tracker:
    """Tracks burpee count, time, and workout state with robust detection"""
    
    def __init__(self):
        self.count = 0
        self.target = 0
        self.stage = 'down'
        self.active = False
        self.start_time = None
        self.paused_time = 0
        self.rep_times = []
    
    def start(self, target_reps):
        self.count = 0
        self.target = target_reps
        self.stage = 'down'
        self.start_time = time.time()
        self.paused_time = 0
        self.rep_times = []
        self.active = True
    
    def pause(self):
        if self.active:
            self.paused_time = self.get_elapsed_time()
            self.active = False
    
    def resume(self):
        if not self.active and self.start_time:
            self.start_time = time.time() - self.paused_time
            self.active = True
    
    def reset(self):
        self.__init__()
    
    def update_state(self, body_state):
        """Update state based on body visibility"""
        if not self.active:
            return
        
        if body_state is None:
            if self.stage != 'down':
                self.stage = 'down'
                self.count += 1
                
                if self.rep_times:
                    self.rep_times.append(time.time() - sum(self.rep_times) - self.start_time)
                else:
                    self.rep_times.append(time.time() - self.start_time)
            return
        
        is_visible = body_state['is_visible']
        
        if is_visible:
            if self.stage != 'up':
                self.stage = 'up'
        else:
            if self.stage != 'down':
                self.stage = 'down'
                self.count += 1
                
                if self.rep_times:
                    self.rep_times.append(time.time() - sum(self.rep_times) - self.start_time)
                else:
                    self.rep_times.append(time.time() - self.start_time)
    
    def get_elapsed_time(self):
        if not self.start_time:
            return 0
        if self.active:
            return time.time() - self.start_time
        return self.paused_time
    
    def get_speed(self):
        elapsed = self.get_elapsed_time()
        return (self.count / elapsed) * 60 if elapsed > 0 and self.count > 0 else 0
    
    def get_metrics(self):
        return {
            'count': self.count,
            'target': self.target,
            'time': int(self.get_elapsed_time()),
            'speed': self.get_speed(),
            'stage': self.stage,
            'active': self.active,
            'complete': self.target > 0 and self.count >= self.target
        }


class Camera:
    """Captures and processes camera feed with body marker detection"""
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=1
        )
        self.camera = cv2.VideoCapture(0)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    def get_frame(self):
        success, frame = self.camera.read()
        return frame if success else None
    
    def get_body_state(self, landmarks):
        """Determine if body is visible based on landmark visibility"""
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE.value]
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        
        nose_visible = nose.visibility > 0.5
        shoulders_visible = (left_shoulder.visibility > 0.5 and right_shoulder.visibility > 0.5)
        hips_visible = (left_hip.visibility > 0.5 and right_hip.visibility > 0.5)
        
        is_visible = nose_visible and shoulders_visible and hips_visible
        
        return {
            'is_visible': is_visible,
            'nose_y': nose.y
        }
    
    def get_user_state(self, frame):
        """Process frame and return body state"""
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.pose.process(image)
        
        if not results.pose_landmarks:
            return None
        
        landmarks = results.pose_landmarks.landmark
        return self.get_body_state(landmarks)
    
    def draw_skeleton(self, frame, landmarks):
        """Draw skeleton overlay on frame"""
        h, w, _ = frame.shape
        
        connections = [
            (self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_SHOULDER),
            (self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.LEFT_HIP),
            (self.mp_pose.PoseLandmark.RIGHT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_HIP),
            (self.mp_pose.PoseLandmark.LEFT_HIP, self.mp_pose.PoseLandmark.RIGHT_HIP),
            (self.mp_pose.PoseLandmark.LEFT_HIP, self.mp_pose.PoseLandmark.LEFT_KNEE),
            (self.mp_pose.PoseLandmark.RIGHT_HIP, self.mp_pose.PoseLandmark.RIGHT_KNEE),
            (self.mp_pose.PoseLandmark.LEFT_KNEE, self.mp_pose.PoseLandmark.LEFT_ANKLE),
            (self.mp_pose.PoseLandmark.RIGHT_KNEE, self.mp_pose.PoseLandmark.RIGHT_ANKLE),
            (self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.LEFT_WRIST),
            (self.mp_pose.PoseLandmark.RIGHT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_WRIST),
        ]
        
        for connection in connections:
            start = landmarks[connection[0].value]
            end = landmarks[connection[1].value]
            start_pos = (int(start.x * w), int(start.y * h))
            end_pos = (int(end.x * w), int(end.y * h))
            cv2.line(frame, start_pos, end_pos, (0, 255, 0), 4)
        
        key_points = [
            self.mp_pose.PoseLandmark.NOSE,
            self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            self.mp_pose.PoseLandmark.LEFT_HIP,
            self.mp_pose.PoseLandmark.RIGHT_HIP,
            self.mp_pose.PoseLandmark.LEFT_KNEE,
            self.mp_pose.PoseLandmark.RIGHT_KNEE,
            self.mp_pose.PoseLandmark.LEFT_ANKLE,
            self.mp_pose.PoseLandmark.RIGHT_ANKLE,
            self.mp_pose.PoseLandmark.LEFT_WRIST,
            self.mp_pose.PoseLandmark.RIGHT_WRIST,
        ]
        
        for point in key_points:
            landmark = landmarks[point.value]
            pos = (int(landmark.x * w), int(landmark.y * h))
            cv2.circle(frame, pos, 8, (0, 255, 0), -1)
    
    def process_frame(self, frame):
        """Process frame with pose detection and visualization"""
        frame = cv2.flip(frame, 1)
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.pose.process(image)
        
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if results.pose_landmarks:
            self.draw_skeleton(image, results.pose_landmarks.landmark)
        
        return image
    
    def release(self):
        self.camera.release()


tracker = Tracker()
camera = Camera()


def generate_frames():
    while True:
        frame = camera.get_frame()
        if frame is None:
            break
        
        user_state = camera.get_user_state(frame)
        if user_state:
            tracker.update_state(user_state)
        
        annotated_frame = camera.process_frame(frame)
        
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/metrics')
def get_metrics():
    return jsonify(tracker.get_metrics())


@app.route('/')
def index():
    return render_template_string('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Burpee Tracker</title>
            <link rel="preconnect" href="https://fonts.googleapis.com">
            <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
            <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
            <style>
                * {
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }
                html, body {
                    width: 100%;
                    height: 100%;
                    overflow: hidden;
                }
                body {
                    font-family: 'Inter', sans-serif;
                    background: linear-gradient(135deg, #1a4c75 0%, #03080d 100%);
                    color: #f2f8fc;
                    position: relative;
                }
                body::before {
                    content: '';
                    position: absolute;
                    top: 0;
                    left: 0;
                    right: 0;
                    bottom: 0;
                    background: radial-gradient(circle at 20% 50%, rgba(45, 133, 205, 0.1) 0%, transparent 50%),
                                radial-gradient(circle at 80% 50%, rgba(45, 133, 205, 0.08) 0%, transparent 50%);
                    pointer-events: none;
                }
                .setup-container {
                    background: rgba(30, 41, 59, 0.8);
                    backdrop-filter: blur(20px);
                    padding: 50px;
                    border-radius: 24px;
                    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.6),
                                0 0 0 1px rgba(255, 255, 255, 0.1);
                    position: absolute;
                    top: 50%;
                    left: 50%;
                    transform: translate(-50%, -50%);
                    z-index: 1;
                    width: 500px;
                    text-align: center;
                }
                .setup-title {
                    font-size: 36px;
                    font-weight: 700;
                    color: #f2f8fc;
                    letter-spacing: -0.5px;
                    margin: 0 0 16px 0;
                }
                .setup-subtitle {
                    color: #f2f8fc;
                    font-size: 15px;
                    font-weight: 400;
                    margin: 0 0 40px 0;
                    line-height: 1.5;
                }
                .setup-label {
                    color: #f2f8fc;
                    font-size: 12px;
                    font-weight: 600;
                    letter-spacing: 1.2px;
                    display: block;
                    margin-bottom: 12px;
                }
                .setup-input {
                    width: 100%;
                    padding: 18px 24px;
                    font-size: 20px;
                    font-weight: 600;
                    background: rgba(15, 23, 42, 0.6);
                    border: 2px solid rgba(148, 163, 184, 0.2);
                    border-radius: 12px;
                    color: #f2f8fc;
                    font-family: 'Inter', sans-serif;
                    text-align: center;
                    transition: all 0.3s ease;
                    box-sizing: border-box;
                    margin-bottom: 32px;
                }
                .setup-input:focus {
                    outline: none;
                    border-color: #2d85cd;
                    background: rgba(15, 23, 42, 0.8);
                    box-shadow: 0 0 0 4px rgba(45, 133, 205, 0.2);
                    transform: scale(1.02);
                }
                .setup-button {
                    width: 100%;
                    padding: 18px 32px;
                    font-size: 14px;
                    font-weight: 700;
                    border: none;
                    border-radius: 8px;
                    cursor: pointer;
                    background: #2d85cd;
                    color: #f2f8fc;
                    box-shadow: 0 8px 24px rgba(45, 133, 205, 0.4);
                    position: relative;
                    overflow: hidden;
                    letter-spacing: 0.8px;
                    font-family: 'Inter', sans-serif;
                    transition: all 0.2s;
                }
                .setup-button::before {
                    content: '';
                    position: absolute;
                    top: 0;
                    left: -100%;
                    width: 100%;
                    height: 100%;
                    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
                    transition: left 0.5s;
                }
                .setup-button:hover {
                    background: #3a9be0;
                    transform: translateY(-2px);
                    box-shadow: 0 12px 32px rgba(45, 133, 205, 0.5);
                }
                .setup-button:hover::before {
                    left: 100%;
                }
                .setup-footer {
                    margin-top: 40px;
                    text-align: center;
                    color: #f2f8fc;
                    font-size: 12px;
                }
                .setup-footer a {
                    color: #f2f8fc;
                    text-decoration: none;
                    transition: opacity 0.2s;
                }
                .setup-footer a:hover {
                    opacity: 0.7;
                }
                .setup-footer p {
                    margin-top: 4px;
                }
                .workout-container {
                    width: 100%;
                    height: 100%;
                    display: flex;
                    flex-direction: column;
                    padding: 20px;
                    gap: 16px;
                    max-width: 1200px;
                    margin: 0 auto;
                }
                .stats-bar {
                    display: flex;
                    gap: 12px;
                    justify-content: center;
                    flex-shrink: 0;
                }
                .stat-card {
                    min-width: 120px;
                    background: rgba(30, 41, 59, 0.8);
                    backdrop-filter: blur(10px);
                    padding: 16px 20px;
                    border-radius: 12px;
                    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
                    text-align: center;
                    border: 1px solid rgba(45, 133, 205, 0.2);
                }
                .stat-label {
                    font-size: 10px;
                    font-weight: 500;
                    color: #f2f8fc;
                    text-transform: uppercase;
                    letter-spacing: 1px;
                    margin-bottom: 6px;
                }
                .stat-value {
                    font-size: 24px;
                    font-weight: 600;
                    color: #f2f8fc;
                    line-height: 1;
                    height: 29px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }
                .stat-value.complete {
                    color: #2d85cd;
                }
                .video-section {
                    flex: 1;
                    display: flex;
                    flex-direction: column;
                    gap: 12px;
                    min-height: 0;
                }
                .video-container {
                    flex: 1;
                    background: rgba(30, 41, 59, 0.6);
                    border-radius: 12px;
                    overflow: hidden;
                    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    min-height: 0;
                    border: 1px solid rgba(45, 133, 205, 0.2);
                }
                .video-container img {
                    width: 100%;
                    height: 100%;
                    object-fit: cover;
                    display: block;
                    border-radius: 12px;
                }
                .controls {
                    display: flex;
                    gap: 12px;
                    justify-content: center;
                    flex-shrink: 0;
                }
                button {
                    padding: 12px 28px;
                    font-size: 12px;
                    font-weight: 600;
                    border: none;
                    border-radius: 8px;
                    cursor: pointer;
                    background: rgba(30, 41, 59, 0.8);
                    color: #f2f8fc;
                    transition: all 0.2s;
                    text-transform: uppercase;
                    letter-spacing: 0.8px;
                    font-family: 'Inter', sans-serif;
                    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
                    border: 1px solid rgba(45, 133, 205, 0.3);
                }
                button:hover {
                    background: rgba(45, 133, 205, 0.3);
                    transform: translateY(-1px);
                    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
                }
                button:active {
                    transform: translateY(0);
                    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.3);
                }
                button.primary {
                    background: #2d85cd;
                    color: #f2f8fc;
                    font-weight: 700;
                    box-shadow: 0 8px 24px rgba(45, 133, 205, 0.4);
                    position: relative;
                    overflow: hidden;
                    border: none;
                }
                button.primary::before {
                    content: '';
                    position: absolute;
                    top: 0;
                    left: -100%;
                    width: 100%;
                    height: 100%;
                    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
                    transition: left 0.5s;
                }
                button.primary:hover {
                    background: #3a9be0;
                    transform: translateY(-2px);
                    box-shadow: 0 12px 32px rgba(45, 133, 205, 0.5);
                }
                button.primary:hover::before {
                    left: 100%;
                }
                #setupSection {
                    display: block;
                }
                #workoutSection {
                    display: none;
                }
            </style>
        </head>
        <body>
            <div id="setupSection" class="setup-container">
                <h1 class="setup-title">Burpee Tracker</h1>
                <p class="setup-subtitle">Track your burpee workout with real-time body tracking and speed metrics.</p>
                <input type="number" class="setup-input" id="targetReps" min="1" value="100" />
                <button class="setup-button" onclick="startWorkout()">Start Burpees</button>
                
                <div class="setup-footer">
                    <a href="https://simonwong.site" target="_blank" rel="noopener noreferrer">Visit simonwong.site</a>
                    <p>© 2025 Simon Wong. All rights reserved.</p>
                </div>
            </div>
            
            <div id="workoutSection" class="workout-container">
                <div class="stats-bar">
                    <div class="stat-card">
                        <div class="stat-label">Count</div>
                        <div class="stat-value" id="countValue">0</div>
                    </div>
                    
                    <div class="stat-card">
                        <div class="stat-label">Target</div>
                        <div class="stat-value" id="targetValue">0</div>
                    </div>
                    
                    <div class="stat-card">
                        <div class="stat-label">Time</div>
                        <div class="stat-value" id="timeValue">0s</div>
                    </div>
                    
                    <div class="stat-card">
                        <div class="stat-label">Speed</div>
                        <div class="stat-value" id="speedValue">0</div>
                    </div>
                    
                    <div class="stat-card">
                        <div class="stat-label">Stage</div>
                        <div class="stat-value" id="stageValue">Down</div>
                    </div>
                    
                    <div class="stat-card">
                        <div class="stat-label">Status</div>
                        <div class="stat-value" id="statusValue">Ready</div>
                    </div>
                </div>
                
                <div class="video-section">
                    <div class="video-container">
                        <img src="{{ url_for('video_feed') }}" alt="Video Feed" />
                    </div>
                    
                    <div class="controls">
                        <button id="pauseBtn" onclick="pauseWorkout()">PAUSE</button>
                        <button id="resumeBtn" onclick="resumeWorkout()" style="display:none;">RESUME</button>
                    </div>
                </div>
            </div>
            
            <script>
                let metricsInterval = null;
                
                function updateMetrics() {
                    fetch('/metrics')
                        .then(response => response.json())
                        .then(data => {
                            document.getElementById('countValue').textContent = data.count;
                            document.getElementById('targetValue').textContent = data.target;
                            
                            const minutes = Math.floor(data.time / 60);
                            const seconds = data.time % 60;
                            document.getElementById('timeValue').textContent = 
                                `${minutes}:${seconds.toString().padStart(2, '0')}`;
                            
                            document.getElementById('speedValue').textContent = data.speed.toFixed(1);
                            
                            const stageEl = document.getElementById('stageValue');
                            const statusEl = document.getElementById('statusValue');
                            const countEl = document.getElementById('countValue');
                            
                            if (data.stage === 'down') {
                                stageEl.textContent = 'Down';
                            } else if (data.stage === 'up') {
                                stageEl.textContent = 'Up';
                            } else {
                                stageEl.textContent = 'Down';
                            }
                            
                            if (data.complete) {
                                statusEl.textContent = 'COMPLETE';
                                statusEl.className = 'stat-value complete';
                                countEl.className = 'stat-value complete';
                            } else if (!data.active && data.time > 0) {
                                statusEl.textContent = 'Paused';
                                statusEl.className = 'stat-value';
                                countEl.className = 'stat-value';
                            } else if (data.active) {
                                statusEl.textContent = 'Ready';
                                statusEl.className = 'stat-value';
                                countEl.className = 'stat-value';
                            } else {
                                statusEl.textContent = 'Ready';
                                statusEl.className = 'stat-value';
                                countEl.className = 'stat-value';
                            }
                        });
                }
                
                function startWorkout() {
                    const target = document.getElementById('targetReps').value;
                    if (target < 1) {
                        alert('Please enter a valid number of burpees');
                        return;
                    }
                    
                    fetch('/start', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({target: parseInt(target)})
                    }).then(() => {
                        document.getElementById('setupSection').style.display = 'none';
                        document.getElementById('workoutSection').style.display = 'flex';
                        metricsInterval = setInterval(updateMetrics, 100);
                    });
                }
                
                function pauseWorkout() {
                    fetch('/pause', {method: 'POST'}).then(() => {
                        document.getElementById('pauseBtn').style.display = 'none';
                        document.getElementById('resumeBtn').style.display = 'inline-block';
                    });
                }
                
                function resumeWorkout() {
                    fetch('/resume', {method: 'POST'}).then(() => {
                        document.getElementById('pauseBtn').style.display = 'inline-block';
                        document.getElementById('resumeBtn').style.display = 'none';
                    });
                }
                
                function resetWorkout() {
                    fetch('/reset', {method: 'POST'}).then(() => {
                        if (metricsInterval) {
                            clearInterval(metricsInterval);
                            metricsInterval = null;
                        }
                        document.getElementById('setupSection').style.display = 'flex';
                        document.getElementById('workoutSection').style.display = 'none';
                        document.getElementById('pauseBtn').style.display = 'inline-block';
                        document.getElementById('resumeBtn').style.display = 'none';
                    });
                }
                
                // Keyboard shortcuts
                document.addEventListener('keydown', function(e) {
                    // Enter key - start workout
                    if (e.key === 'Enter' && document.getElementById('setupSection').style.display !== 'none') {
                        e.preventDefault();
                        startWorkout();
                    }
                    
                    // Space key - pause/resume workout
                    if (e.key === ' ' && document.getElementById('workoutSection').style.display !== 'none') {
                        e.preventDefault();
                        const pauseBtn = document.getElementById('pauseBtn');
                        const resumeBtn = document.getElementById('resumeBtn');
                        
                        if (pauseBtn.style.display !== 'none') {
                            pauseWorkout();
                        } else {
                            resumeWorkout();
                        }
                    }
                });
            </script>
        </body>
        </html>
    ''')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start', methods=['POST'])
def start_workout():
    data = request.get_json()
    target = data.get('target', 20)
    tracker.start(target)
    return {'status': 'started'}


@app.route('/pause', methods=['POST'])
def pause_workout():
    tracker.pause()
    return {'status': 'paused'}


@app.route('/resume', methods=['POST'])
def resume_workout():
    tracker.resume()
    return {'status': 'resumed'}


@app.route('/reset', methods=['POST'])
def reset_workout():
    tracker.reset()
    return {'status': 'reset'}


if __name__ == '__main__':
    print("Starting Burpee Tracker...")
    print("Open your browser and navigate to: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
