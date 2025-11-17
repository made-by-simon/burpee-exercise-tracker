"""
Burpee Tracker with Simplified Hand Tracking
Counts reps when hands reach top of frame
"""

import cv2
import mediapipe as mp
import time
from flask import Flask, Response, render_template_string, request, jsonify

app = Flask(__name__)  # WSGI for browser interface. 


class Tracker:
    """
    Class containing core logic for tracking burpees.
    Flow control is handled by "start", "pause", "resume", and "reset" classes. 
    """
    
    def __init__(self):
        """Initialize class variables."""
        self.count = 0
        self.target = 0
        self.active = False  
        self.start_time = None
        self.paused_time = 0
        self.hands_at_top = False
    
    def start(self, target_reps):
        """Set tracker variables for initial start."""
        self.count = 0
        self.target = target_reps
        self.start_time = time.time()
        self.paused_time = 0
        self.active = True
        self.hands_at_top = False
    
    def pause(self):
        """Set tracker variables for pausing."""
        if self.active:
            self.paused_time = self.get_elapsed_time()
            self.active = False
    
    def resume(self):
        """Set tracker variables for resuming."""
        if not self.active and self.start_time:
            self.start_time = time.time() - self.paused_time
            self.active = True
    
    def reset(self):
        """Reset tracker class."""
        self.__init__()
    
    def update(self, hand_y_pos):
        """Keep track when hands reach top of frame."""
        if not self.active or hand_y_pos is None:
            return
        
        # Hands are considered to be "at the top" if they are in the top 10% vertical range (y) of the frame.
        # Vertical range (y) is measured from the top (zero at the top).
        at_top = hand_y_pos < 0.1
        
        # Increment burpee count when hands get to the top. 
        if at_top and not self.hands_at_top:
            self.count += 1
            self.hands_at_top = True
        elif not at_top:
            self.hands_at_top = False
    
    def get_elapsed_time(self):
        """Get relevant time interval (active time if not pasued), (paused time if paused)."""
        if not self.start_time:
            return 0
        # Elapsed time is determined differently depending on active status. 
        return time.time() - self.start_time if self.active else self.paused_time  
    
    def get_metrics(self):
        """Retrieve/calculate metrics then pack them into a dictionary."""
        elapsed = self.get_elapsed_time()
        speed = (self.count / elapsed * 60) if elapsed > 0 else 0
        
        return {
            "count": self.count,
            "target": self.target,
            "time": int(elapsed),
            "speed": speed,
            "stage": "Up" if self.hands_at_top else "Down",
            "active": self.active,
            "complete": self.target > 0 and self.count >= self.target
        }


class Camera:
    """
    Class handling camera and computer vision.
    OpenCV (cv2) is used for video input and computer vision.
    MediaPipe (mp) is used for pose estimation / body marking. 
    """

    def __init__(self):
        """Initialize class variables."""
        self.mp_pose = mp.solutions.pose  # Store reference to MediaPipe Pose module.
        # Create pose estimation object. 
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5 
        )
        # Use webcam with VGA resolution. 
        self.camera = cv2.VideoCapture(0)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    def get_frame(self):
        """Get frame from webcam."""
        success, frame = self.camera.read()
        return cv2.flip(frame, 1) if success else None  # Flip frame horizontally.
    
    def get_hand_position(self, landmarks):
        """Get highest hand/elbow position (lowest y value)"""
        
        # Get specific points from the list of landmarks (there are 33). 
        left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
        right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
        left_elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value]
        right_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        
        # Collect all points that are not obscured or occluded. 
        points = []
        if left_wrist.visibility > 0.5:
            points.append(left_wrist.y)
        if right_wrist.visibility > 0.5:
            points.append(right_wrist.y)
        if left_elbow.visibility > 0.5:
            points.append(left_elbow.y)
        if right_elbow.visibility > 0.5:
            points.append(right_elbow.y)
        
        # Return the highest point (lowest y value). 
        return min(points) if points else None
    
    def draw_skeleton(self, frame, landmarks):
        """Draw simple skeleton overlay"""
        h, w = frame.shape[:2]
        
        # Define connections. 
        connections = [
            """
            Landmark indices from MediaPipe Pose module: 
                11: left shoulder
                12: right shoulder
                13: left elbow
                14: right elbow
                15: left wrist
                16: right wrist
                23: left hip
                24: right hip
                25: left knee
                26: right knee
                27: left ankle
                28: right ankle
            """
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # Arms. 
            (11, 23), (12, 24), (23, 24),  # Torso. 
            (23, 25), (25, 27), (24, 26), (26, 28)  # Legs. 
        ]
        
        # Draw lines. 
        for start_idx, end_idx in connections:
            start = landmarks[start_idx]
            end = landmarks[end_idx]
            if start.visibility > 0.5 and end.visibility > 0.5:
                start_pos = (int(start.x * w), int(start.y * h))
                end_pos = (int(end.x * w), int(end.y * h))
                cv2.line(frame, start_pos, end_pos, (0, 255, 0), 4)
        
        # Draw nodes (line connection points).
        for idx in [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]:
            landmark = landmarks[idx]
            if landmark.visibility > 0.5:
                pos = (int(landmark.x * w), int(landmark.y * h))
                cv2.circle(frame, pos, 8, (0, 255, 0), -1)
    
    def process_frame(self, frame):
        """Process frame and return annotated frame and hand position"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        hand_pos = None
        if results.pose_landmarks:
            self.draw_skeleton(frame, results.pose_landmarks.landmark)
            hand_pos = self.get_hand_position(results.pose_landmarks.landmark)
        
        return frame, hand_pos
    
    def release(self):
        self.camera.release()


tracker = Tracker()
camera = Camera()


def generate_frames():
    """Generator continuously grabs and processes frames then yiels image encoded as JPEG in binary stream."""
    while True:
        frame = camera.get_frame()
        if frame is None:
            break
        
        annotated_frame, hand_pos = camera.process_frame(frame)
        tracker.update(hand_pos)
        
        ret, buffer = cv2.imencode(".jpg", annotated_frame)
        # Yield multi-part stream so that web server can stream live to browser.  
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")


# HTTP endpoint for metrics. 
@app.route("/metrics")
def get_metrics():
    return jsonify(tracker.get_metrics())


# HTTP endpoint that serves the main web UI for the burpee tracker. 
# Render the full HTML/CSS/JavaScript interface (setup screen, live video feed, and real-time metrics).
@app.route("/")
def index():
    return render_template_string("""
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
                    font-family: "Inter", sans-serif;
                    background: linear-gradient(135deg, #1a4c75 0%, #03080d 100%);
                    color: #f2f8fc;
                    position: relative;
                }
                body::before {
                    content: "";
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
                    font-family: "Inter", sans-serif;
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
                    font-family: "Inter", sans-serif;
                    transition: all 0.2s;
                }
                .setup-button::before {
                    content: "";
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
                    font-family: "Inter", sans-serif;
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
                    content: "";
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
                        <img src="{{ url_for("video_feed") }}" alt="Video Feed" />
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
                    fetch("/metrics")
                        .then(response => response.json())
                        .then(data => {
                            document.getElementById("countValue").textContent = data.count;
                            document.getElementById("targetValue").textContent = data.target;
                            
                            const minutes = Math.floor(data.time / 60);
                            const seconds = data.time % 60;
                            document.getElementById("timeValue").textContent = 
                                `${minutes}:${seconds.toString().padStart(2, "0")}`;
                            
                            document.getElementById("speedValue").textContent = data.speed.toFixed(1);
                            
                            const stageEl = document.getElementById("stageValue");
                            const statusEl = document.getElementById("statusValue");
                            const countEl = document.getElementById("countValue");
                            
                            stageEl.textContent = data.stage;
                            
                            if (data.complete) {
                                statusEl.textContent = "COMPLETE";
                                statusEl.className = "stat-value complete";
                                countEl.className = "stat-value complete";
                            } else if (!data.active && data.time > 0) {
                                statusEl.textContent = "Paused";
                                statusEl.className = "stat-value";
                                countEl.className = "stat-value";
                            } else if (data.active) {
                                statusEl.textContent = "Ready";
                                statusEl.className = "stat-value";
                                countEl.className = "stat-value";
                            } else {
                                statusEl.textContent = "Ready";
                                statusEl.className = "stat-value";
                                countEl.className = "stat-value";
                            }
                        });
                }
                
                function startWorkout() {
                    const target = document.getElementById("targetReps").value;
                    if (target < 1) {
                        alert("Please enter a valid number of burpees");
                        return;
                    }
                    
                    fetch("/start", {
                        method: "POST",
                        headers: {"Content-Type": "application/json"},
                        body: JSON.stringify({target: parseInt(target)})
                    }).then(() => {
                        document.getElementById("setupSection").style.display = "none";
                        document.getElementById("workoutSection").style.display = "flex";
                        metricsInterval = setInterval(updateMetrics, 100);
                    });
                }
                
                function pauseWorkout() {
                    fetch("/pause", {method: "POST"}).then(() => {
                        document.getElementById("pauseBtn").style.display = "none";
                        document.getElementById("resumeBtn").style.display = "inline-block";
                    });
                }
                
                function resumeWorkout() {
                    fetch("/resume", {method: "POST"}).then(() => {
                        document.getElementById("pauseBtn").style.display = "inline-block";
                        document.getElementById("resumeBtn").style.display = "none";
                    });
                }
                
                function resetWorkout() {
                    fetch("/reset", {method: "POST"}).then(() => {
                        if (metricsInterval) {
                            clearInterval(metricsInterval);
                            metricsInterval = null;
                        }
                        document.getElementById("setupSection").style.display = "flex";
                        document.getElementById("workoutSection").style.display = "none";
                        document.getElementById("pauseBtn").style.display = "inline-block";
                        document.getElementById("resumeBtn").style.display = "none";
                    });
                }
                
                // Keyboard shortcuts
                document.addEventListener("keydown", function(e) {
                    // Enter key - start workout
                    if (e.key === "Enter" && document.getElementById("setupSection").style.display !== "none") {
                        e.preventDefault();
                        startWorkout();
                    }
                    
                    // Space key - pause/resume workout
                    if (e.key === " " && document.getElementById("workoutSection").style.display !== "none") {
                        e.preventDefault();
                        const pauseBtn = document.getElementById("pauseBtn");
                        const resumeBtn = document.getElementById("resumeBtn");
                        
                        if (pauseBtn.style.display !== "none") {
                            pauseWorkout();
                        } else {
                            resumeWorkout();
                        }
                    }
                });
            </script>
        </body>
        </html>
    """)


# HTTP endpoint for video feed. 
@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# HTTP endpoint for starting workout with given target. 
@app.route("/start", methods=["POST"])
def start_workout():
    data = request.get_json()
    target = data.get("target", 20)
    tracker.start(target)
    return {"status": "started"}


# HTTP endpoint for pausing workout. 
@app.route("/pause", methods=["POST"])
def pause_workout():
    tracker.pause()
    return {"status": "paused"}


# HTTP endpoint for resuming workout. 
@app.route("/resume", methods=["POST"])
def resume_workout():
    tracker.resume()
    return {"status": "resumed"}


# HTTP endpoint for resetting workout (resets entire app/interface). 
@app.route("/reset", methods=["POST"])
def reset_workout():
    tracker.reset()
    return {"status": "reset"}


if __name__ == "__main__":
    print("Starting Burpee Tracker...")
    print("Open your browser and navigate to: http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)