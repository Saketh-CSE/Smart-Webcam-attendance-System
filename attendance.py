import base64
import csv
import os
import re
import shutil
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template_string, request, send_from_directory
from flask_cors import CORS
from pandas.errors import EmptyDataError


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
APP_NAME = "Smart Attendance System – Facial Recognition Attendance"
HAAR_CASCADE_PATH = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")
TRAINER_PATH = os.path.join(BASE_DIR, "TrainingImage", "Trainner.yml")
STUDENT_DETAILS_FILE = os.path.join(BASE_DIR, "StudentDetails", "studentdetails.csv")
TRAINING_IMAGE_PATH = os.path.join(BASE_DIR, "TrainingImage")
ATTENDANCE_PATH = os.path.join(BASE_DIR, "Attendance")

os.makedirs(os.path.dirname(STUDENT_DETAILS_FILE), exist_ok=True)
os.makedirs(TRAINING_IMAGE_PATH, exist_ok=True)
os.makedirs(ATTENDANCE_PATH, exist_ok=True)

if not os.path.exists(STUDENT_DETAILS_FILE):
    pd.DataFrame(columns=["ID", "Enrollment No.", "Name"]).to_csv(STUDENT_DETAILS_FILE, index=False)

app = Flask(__name__)
CORS(app)

face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
if face_cascade.empty():
    raise RuntimeError(f"Could not load Haar cascade file: {HAAR_CASCADE_PATH}")

recognizer = cv2.face.LBPHFaceRecognizer_create()


def load_model():
    global recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    if os.path.exists(TRAINER_PATH) and os.path.getsize(TRAINER_PATH) > 0:
        recognizer.read(TRAINER_PATH)
        return True
    return False


model_loaded = load_model()


def safe_folder_part(value):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_") or "student"


def get_new_id():
    try:
        df = pd.read_csv(STUDENT_DETAILS_FILE)
        if df.empty or "ID" not in df.columns:
            return 1
        return int(pd.to_numeric(df["ID"], errors="coerce").max()) + 1
    except (FileNotFoundError, EmptyDataError, ValueError):
        return 1


def get_id_map():
    try:
        df = pd.read_csv(STUDENT_DETAILS_FILE)
        return {int(row["ID"]): row["Name"] for _, row in df.iterrows()}
    except (FileNotFoundError, EmptyDataError, KeyError, ValueError):
        return {}


def decode_image(data_url, color_mode=cv2.IMREAD_GRAYSCALE):
    encoded_image = data_url.split(",", 1)[1]
    decoded_image = base64.b64decode(encoded_image)
    nparr = np.frombuffer(decoded_image, np.uint8)
    return cv2.imdecode(nparr, color_mode)


def train_model():
    trainer = cv2.face.LBPHFaceRecognizer_create()
    faces = []
    ids = []

    for folder_name in os.listdir(TRAINING_IMAGE_PATH):
        folder_path = os.path.join(TRAINING_IMAGE_PATH, folder_name)
        if not os.path.isdir(folder_path):
            continue

        try:
            student_id = int(folder_name.split("_", 1)[0])
        except ValueError:
            continue

        for image_name in os.listdir(folder_path):
            if not image_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            image_path = os.path.join(folder_path, image_name)
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                faces.append(img)
                ids.append(student_id)

    if not faces:
        return False

    trainer.train(faces, np.array(ids))
    trainer.write(TRAINER_PATH)
    return True


def mark_attendance(student_id, name):
    today = datetime.now().strftime("%Y-%m-%d")
    now = datetime.now().strftime("%H:%M:%S")
    file_path = os.path.join(ATTENDANCE_PATH, f"Attendance_{today}.csv")
    already_marked = False

    if os.path.exists(file_path):
        with open(file_path, newline="", encoding="utf-8") as attendance_file:
            for row in csv.DictReader(attendance_file):
                if row.get("ID") == str(student_id):
                    already_marked = True
                    break

    if not already_marked:
        file_exists = os.path.exists(file_path)
        with open(file_path, "a", newline="", encoding="utf-8") as attendance_file:
            writer = csv.DictWriter(attendance_file, fieldnames=["ID", "Name", "Date", "Time"])
            if not file_exists:
                writer.writeheader()
            writer.writerow({"ID": student_id, "Name": name, "Date": today, "Time": now})

    return {"id": student_id, "name": name, "date": today, "time": now, "already_marked": already_marked}


FRONTEND_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Smart Attendance System – Facial Recognition Attendance</title>
    <link rel="icon" href="/AMS.ico">
    <style>
        :root {
            --bg: #eef3f8;
            --panel: #ffffff;
            --ink: #1d2633;
            --muted: #64748b;
            --line: #d8e1ec;
            --blue: #2563eb;
            --blue-dark: #1e40af;
            --green: #15803d;
            --red: #dc2626;
            --amber: #b45309;
            --glass: rgba(255, 255, 255, 0.76);
        }

        * { box-sizing: border-box; }

        body {
            margin: 0;
            min-height: 100vh;
            font-family: Arial, Helvetica, sans-serif;
            color: var(--ink);
            background: var(--bg);
        }

        .shell {
            display: grid;
            grid-template-columns: 260px 1fr;
            min-height: 100vh;
        }

        .sidebar {
            background: #122033;
            color: #f8fafc;
            padding: 28px 22px;
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 32px;
        }

        .brand-mark {
            display: grid;
            place-items: center;
            width: 44px;
            height: 44px;
            border-radius: 8px;
            background: #f8fafc;
            color: #122033;
            font-weight: 800;
        }

        .brand strong { display: block; font-size: 18px; }
        .brand span { color: #b9c7d8; font-size: 13px; }

        .nav {
            display: grid;
            gap: 10px;
        }

        .nav button {
            display: flex;
            align-items: center;
            gap: 10px;
            width: 100%;
            border: 1px solid transparent;
            border-radius: 8px;
            padding: 12px 14px;
            color: #dbeafe;
            background: transparent;
            cursor: pointer;
            text-align: left;
            font-size: 15px;
        }

        .nav button.active,
        .nav button:hover {
            background: #1d3350;
            border-color: #315177;
        }

        .main {
            padding: 28px;
        }

        .topbar {
            display: flex;
            justify-content: space-between;
            gap: 20px;
            align-items: center;
            margin-bottom: 24px;
        }

        h1, h2, p { margin-top: 0; }
        h1 { font-size: 30px; margin-bottom: 6px; }
        h2 { font-size: 22px; margin-bottom: 16px; }
        .muted { color: var(--muted); }

        .status-pill {
            border: 1px solid var(--line);
            border-radius: 999px;
            padding: 8px 12px;
            background: var(--panel);
            color: var(--muted);
            white-space: nowrap;
        }

        .glass-tab {
            position: fixed;
            top: 18px;
            right: 18px;
            z-index: 50;
            width: min(420px, calc(100vw - 36px));
            border: 1px solid rgba(255, 255, 255, 0.58);
            border-left: 5px solid var(--blue);
            border-radius: 8px;
            padding: 14px 16px;
            color: #172033;
            background: var(--glass);
            box-shadow: 0 18px 40px rgba(15, 23, 42, 0.18);
            backdrop-filter: blur(16px);
            transform: translateY(-130%);
            opacity: 0;
            pointer-events: none;
            transition: transform 180ms ease, opacity 180ms ease;
        }

        .glass-tab.show {
            transform: translateY(0);
            opacity: 1;
            pointer-events: auto;
        }

        .glass-tab.error { border-left-color: var(--red); }
        .glass-tab.success { border-left-color: var(--green); }

        .glass-tab strong {
            display: block;
            margin-bottom: 4px;
            font-size: 14px;
        }

        .glass-tab span {
            color: #334155;
            line-height: 1.4;
        }

        .view { display: none; }
        .view.active { display: block; }

        .grid {
            display: grid;
            grid-template-columns: minmax(320px, 420px) minmax(360px, 1fr);
            gap: 20px;
            align-items: start;
        }

        .panel {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 10px 28px rgba(15, 23, 42, 0.06);
        }

        .form-grid {
            display: grid;
            gap: 14px;
        }

        label {
            display: grid;
            gap: 7px;
            color: #334155;
            font-weight: 700;
            font-size: 14px;
        }

        input {
            width: 100%;
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 12px 13px;
            font: inherit;
            color: var(--ink);
            background: #f8fafc;
        }

        input:focus {
            outline: 2px solid #bfdbfe;
            border-color: var(--blue);
        }

        .actions {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 16px;
        }

        button.primary,
        button.secondary,
        button.danger {
            border: 0;
            border-radius: 8px;
            padding: 12px 16px;
            font: inherit;
            font-weight: 700;
            cursor: pointer;
        }

        button.primary { background: var(--blue); color: white; }
        button.primary:hover { background: var(--blue-dark); }
        button.secondary { background: #e2e8f0; color: #1e293b; }
        button.secondary:hover { background: #cbd5e1; }
        button.danger { background: #fee2e2; color: var(--red); }

        button:disabled {
            cursor: not-allowed;
            opacity: 0.65;
        }

        video {
            width: 100%;
            aspect-ratio: 4 / 3;
            object-fit: cover;
            border-radius: 8px;
            background: #020617;
            border: 1px solid #111827;
        }

        .message {
            min-height: 42px;
            margin-top: 14px;
            padding: 12px;
            border-radius: 8px;
            background: #eff6ff;
            color: #1e3a8a;
            border: 1px solid #bfdbfe;
        }

        .message.error {
            color: #991b1b;
            background: #fef2f2;
            border-color: #fecaca;
        }

        .message.success {
            color: #14532d;
            background: #f0fdf4;
            border-color: #bbf7d0;
        }

        .recognition-result {
            display: grid;
            gap: 10px;
            place-items: center;
            min-height: 190px;
            text-align: center;
            border: 1px dashed var(--line);
            border-radius: 8px;
            background: #f8fafc;
        }

        .recognition-result strong {
            font-size: 34px;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            overflow: hidden;
        }

        th, td {
            padding: 12px;
            border-bottom: 1px solid var(--line);
            text-align: left;
        }

        th {
            background: #f8fafc;
            color: #475569;
            font-size: 13px;
            text-transform: uppercase;
        }

        @media (max-width: 900px) {
            .shell { grid-template-columns: 1fr; }
            .sidebar { position: static; }
            .grid { grid-template-columns: 1fr; }
            .topbar { align-items: flex-start; flex-direction: column; }
        }
    </style>
</head>
<body>
    <div class="shell">
        <aside class="sidebar">
            <div class="brand">
                <div class="brand-mark">AMS</div>
                <div>
                    <strong>Smart Attendance</strong>
                    <span>Facial Recognition</span>
                </div>
            </div>
            <nav class="nav">
                <button class="active" data-view="register-view">Register Student</button>
                <button data-view="attendance-view">Take Attendance</button>
                <button data-view="records-view">View Records</button>
            </nav>
        </aside>

        <main class="main">
            <div class="topbar">
                <div>
                    <h1>Smart Attendance System – Facial Recognition Attendance</h1>
                    <p class="muted">Register faces, take attendance, and review records from one clean screen.</p>
                </div>
                <div id="server-status" class="status-pill">Server ready</div>
            </div>

            <section id="register-view" class="view active">
                <div class="grid">
                    <div class="panel">
                        <h2>Register Student</h2>
                        <div class="form-grid">
                            <label>Enrollment No.
                                <input id="reg-enrollment" type="text" autocomplete="off">
                            </label>
                            <label>Student Name
                                <input id="reg-name" type="text" autocomplete="off">
                            </label>
                        </div>
                        <div class="actions">
                            <button id="register-button" class="primary" onclick="startRegistration()">Capture & Register</button>
                            <button class="secondary" onclick="stopCamera()">Stop Camera</button>
                        </div>
                        <div id="reg-status" class="message">Enter student details and keep the face centered in the preview.</div>
                    </div>
                    <div class="panel">
                        <h2>Camera Preview</h2>
                        <video id="reg-webcam" autoplay playsinline></video>
                    </div>
                </div>
            </section>

            <section id="attendance-view" class="view">
                <div class="grid">
                    <div class="panel">
                        <h2>Live Recognition</h2>
                        <div class="recognition-result">
                            <span class="muted">Current result</span>
                            <strong id="att-result">Waiting</strong>
                            <span id="att-subtitle" class="muted">Start attendance to begin scanning.</span>
                        </div>
                        <div class="actions">
                            <button class="primary" onclick="startAttendance()">Start Attendance</button>
                            <button class="secondary" onclick="stopAttendance()">Stop</button>
                        </div>
                        <div id="att-status" class="message">Model status: {{ model_status }}</div>
                    </div>
                    <div class="panel">
                        <h2>Camera Preview</h2>
                        <video id="att-webcam" autoplay playsinline></video>
                    </div>
                </div>
            </section>

            <section id="records-view" class="view">
                <div class="panel">
                    <div class="topbar">
                        <div>
                            <h2>Attendance Records</h2>
                            <p class="muted">Latest marked attendance entries.</p>
                        </div>
                        <button class="primary" onclick="fetchAttendance()">Refresh</button>
                    </div>
                    <table id="attendance-table">
                        <thead>
                            <tr><th>ID</th><th>Name</th><th>Date</th><th>Time</th></tr>
                        </thead>
                        <tbody></tbody>
                    </table>
                </div>
            </section>
        </main>
    </div>

    <canvas id="canvas" style="display:none;"></canvas>
    <div id="glass-tab" class="glass-tab" role="status" aria-live="polite">
        <strong id="glass-title">Status</strong>
        <span id="glass-message">Ready</span>
    </div>

    <script>
        const SERVER_URL = window.location.origin;
        const CAPTURE_COUNT = 60;
        const CAPTURE_DELAY_MS = 180;
        const RECOGNITION_INTERVAL_MS = 1800;

        let webcamStream = null;
        let recognitionTimer = null;
        let glassTimer = null;

        document.querySelectorAll(".nav button").forEach(button => {
            button.addEventListener("click", () => showView(button.dataset.view));
        });

        function showView(viewId) {
            document.querySelectorAll(".view").forEach(view => view.classList.remove("active"));
            document.getElementById(viewId).classList.add("active");
            document.querySelectorAll(".nav button").forEach(button => {
                button.classList.toggle("active", button.dataset.view === viewId);
            });

            stopAttendance();
            if (viewId === "records-view") {
                fetchAttendance();
            }
        }

        function setMessage(id, text, type = "") {
            const element = document.getElementById(id);
            element.textContent = text;
            element.className = `message ${type}`;
            showGlass(type === "error" ? "Error" : type === "success" ? "Success" : "Status", text, type);
        }

        function showGlass(title, message, type = "") {
            const tab = document.getElementById("glass-tab");
            document.getElementById("glass-title").textContent = title;
            document.getElementById("glass-message").textContent = message;
            tab.className = `glass-tab show ${type}`;

            clearTimeout(glassTimer);
            glassTimer = setTimeout(() => {
                tab.className = `glass-tab ${type}`;
            }, type === "error" ? 8000 : 4200);
        }

        async function startCamera(videoId) {
            stopCamera();
            const video = document.getElementById(videoId);
            webcamStream = await navigator.mediaDevices.getUserMedia({
                video: { width: { ideal: 800 }, height: { ideal: 600 } },
                audio: false
            });
            video.srcObject = webcamStream;

            if (video.readyState < 2) {
                await new Promise(resolve => {
                    video.onloadedmetadata = resolve;
                });
            }
            return video;
        }

        function stopCamera() {
            if (webcamStream) {
                webcamStream.getTracks().forEach(track => track.stop());
                webcamStream = null;
            }
        }

        function captureFrame(video) {
            const canvas = document.getElementById("canvas");
            const context = canvas.getContext("2d");
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            context.drawImage(video, 0, 0, canvas.width, canvas.height);
            return canvas.toDataURL("image/jpeg", 0.9);
        }

        async function parseResponse(response) {
            const text = await response.text();
            try {
                return text ? JSON.parse(text) : {};
            } catch (error) {
                throw new Error(text || `Server returned ${response.status}`);
            }
        }

        async function startRegistration() {
            const button = document.getElementById("register-button");
            const enrollment_no = document.getElementById("reg-enrollment").value.trim();
            const name = document.getElementById("reg-name").value.trim();

            if (!enrollment_no || !name) {
                setMessage("reg-status", "Enrollment number and student name are required.", "error");
                return;
            }

            button.disabled = true;
            try {
                setMessage("reg-status", "Starting camera...");
                const video = await startCamera("reg-webcam");
                const images = [];

                for (let count = 0; count < CAPTURE_COUNT; count++) {
                    images.push(captureFrame(video));
                    setMessage("reg-status", `Captured ${count + 1} of ${CAPTURE_COUNT} images.`);
                    await new Promise(resolve => setTimeout(resolve, CAPTURE_DELAY_MS));
                }

                stopCamera();
                setMessage("reg-status", "Registering student and training model...");
                const response = await fetch(`${SERVER_URL}/register`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ enrollment_no, name, images })
                });
                const result = await parseResponse(response);

                if (!response.ok) {
                    setMessage("reg-status", result.error || "Registration failed.", "error");
                    return;
                }

                setMessage("reg-status", result.message, "success");
                document.getElementById("reg-enrollment").value = "";
                document.getElementById("reg-name").value = "";
            } catch (error) {
                console.error(error);
                setMessage("reg-status", error.message || "Could not register student.", "error");
                stopCamera();
            } finally {
                button.disabled = false;
            }
        }

        async function startAttendance() {
            try {
                const video = await startCamera("att-webcam");
                setMessage("att-status", "Scanning for registered faces...");
                document.getElementById("att-result").textContent = "Scanning";
                document.getElementById("att-subtitle").textContent = "Keep the face visible in the frame.";

                recognitionTimer = setInterval(async () => {
                    try {
                        const image = captureFrame(video);
                        const response = await fetch(`${SERVER_URL}/recognize`, {
                            method: "POST",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify({ image })
                        });
                        const result = await parseResponse(response);

                        if (!response.ok) {
                            setMessage("att-status", result.error || "Recognition failed.", "error");
                            return;
                        }

                        if (result.name && result.name !== "Unknown") {
                            document.getElementById("att-result").textContent = result.name;
                            document.getElementById("att-subtitle").textContent = result.already_marked ? "Attendance was already marked today." : "Attendance marked successfully.";
                            setMessage("att-status", result.message, "success");
                        } else {
                            document.getElementById("att-result").textContent = "Scanning";
                            document.getElementById("att-subtitle").textContent = "No registered face matched yet.";
                        }
                    } catch (error) {
                        setMessage("att-status", error.message || "Recognition error.", "error");
                    }
                }, RECOGNITION_INTERVAL_MS);
            } catch (error) {
                setMessage("att-status", error.message || "Could not start camera.", "error");
            }
        }

        function stopAttendance() {
            if (recognitionTimer) {
                clearInterval(recognitionTimer);
                recognitionTimer = null;
            }
            stopCamera();
        }

        async function fetchAttendance() {
            const tbody = document.querySelector("#attendance-table tbody");
            tbody.innerHTML = "<tr><td colspan='4'>Loading...</td></tr>";
            try {
                const response = await fetch(`${SERVER_URL}/attendance_records`);
                const records = await parseResponse(response);
                tbody.innerHTML = "";

                if (!records.length) {
                    tbody.innerHTML = "<tr><td colspan='4'>No attendance records found.</td></tr>";
                    return;
                }

                records.forEach(record => {
                    const row = document.createElement("tr");
                    row.innerHTML = `<td>${record.id}</td><td>${record.name}</td><td>${record.date}</td><td>${record.time}</td>`;
                    tbody.appendChild(row);
                });
            } catch (error) {
                tbody.innerHTML = `<tr><td colspan='4'>${error.message || "Failed to load records."}</td></tr>`;
            }
        }
    </script>
</body>
</html>
"""


@app.errorhandler(Exception)
def handle_unexpected_error(error):
    print(f"Unexpected server error: {error}")
    return jsonify({"error": str(error)}), 500


@app.route("/")
def index():
    status = "trained" if model_loaded else "not trained yet"
    return render_template_string(FRONTEND_HTML, model_status=status)


@app.route("/AMS.ico")
@app.route("/favicon.ico")
def favicon():
    return send_from_directory(BASE_DIR, "AMS.ico")


@app.route("/register", methods=["POST"])
def register_student():
    folder_path = None
    try:
        data = request.get_json(silent=True) or {}
        enrollment_no = str(data.get("enrollment_no", "")).strip()
        name = str(data.get("name", "")).strip()
        images = data.get("images", [])

        if not enrollment_no or not name or not images:
            return jsonify({"error": "Enrollment number, name, and images are required."}), 400

        new_id = int(get_new_id())
        folder_path = os.path.join(TRAINING_IMAGE_PATH, f"{new_id}_{safe_folder_part(name)}")
        if os.path.exists(folder_path):
            return jsonify({"error": "Student image folder already exists. Try a different name or clean old data."}), 409

        os.makedirs(folder_path, exist_ok=False)
        saved_count = 0

        for image_data in images:
            try:
                img = decode_image(image_data)
            except (IndexError, TypeError, ValueError):
                continue

            if img is None:
                continue

            faces = face_cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=5)
            if len(faces) == 0:
                continue

            x, y, w, h = faces[0]
            face = cv2.resize(img[y:y + h, x:x + w], (100, 100))
            cv2.imwrite(os.path.join(folder_path, f"{saved_count}.jpg"), face)
            saved_count += 1

        if saved_count < 10:
            shutil.rmtree(folder_path, ignore_errors=True)
            return jsonify({"error": f"Only {saved_count} usable face images were captured. Face the camera clearly and try again."}), 400

        df_new = pd.DataFrame([{"ID": new_id, "Enrollment No.": enrollment_no, "Name": name}])
        if os.path.exists(STUDENT_DETAILS_FILE):
            try:
                df_existing = pd.read_csv(STUDENT_DETAILS_FILE)
                df_updated = pd.concat([df_existing, df_new], ignore_index=True)
            except EmptyDataError:
                df_updated = df_new
        else:
            df_updated = df_new

        df_updated.to_csv(STUDENT_DETAILS_FILE, index=False)

        if train_model():
            load_model()
            return jsonify({"message": f"{name} registered with {saved_count} face images. Model trained successfully.", "new_id": new_id})

        return jsonify({"message": f"{name} registered with {saved_count} face images, but model training failed.", "new_id": new_id}), 200
    except Exception as error:
        if folder_path and os.path.isdir(folder_path):
            shutil.rmtree(folder_path, ignore_errors=True)
        print(f"Registration error: {error}")
        return jsonify({"error": f"Registration failed: {error}"}), 500


@app.route("/recognize", methods=["POST"])
def recognize_face():
    global model_loaded
    if not model_loaded:
        model_loaded = load_model()
    if not model_loaded:
        return jsonify({"error": "Model is not trained yet. Register a student first."}), 400

    data = request.get_json(silent=True) or {}
    if "image" not in data:
        return jsonify({"error": "No image data provided."}), 400

    try:
        img = decode_image(data["image"], cv2.IMREAD_COLOR)
    except (IndexError, TypeError, ValueError):
        return jsonify({"error": "Invalid image data."}), 400

    if img is None:
        return jsonify({"error": "Could not decode image."}), 400

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    id_map = get_id_map()

    for x, y, w, h in faces:
        face = cv2.resize(gray[y:y + h, x:x + w], (100, 100))
        student_id, confidence = recognizer.predict(face)
        if confidence < 65 and int(student_id) in id_map:
            name = id_map[int(student_id)]
            attendance = mark_attendance(int(student_id), name)
            message = f"Welcome, {name}. Attendance already marked today." if attendance["already_marked"] else f"Welcome, {name}. Attendance marked."
            return jsonify({"name": name, "message": message, **attendance})

    return jsonify({"name": "Unknown", "message": "No matching registered face found."})


@app.route("/attendance_records")
def attendance_records():
    records = []
    for file_name in sorted(os.listdir(ATTENDANCE_PATH), reverse=True):
        if not file_name.lower().endswith(".csv"):
            continue
        file_path = os.path.join(ATTENDANCE_PATH, file_name)
        with open(file_path, newline="", encoding="utf-8") as attendance_file:
            for row in csv.DictReader(attendance_file):
                records.append({
                    "id": row.get("ID", ""),
                    "name": row.get("Name", ""),
                    "date": row.get("Date", ""),
                    "time": row.get("Time", ""),
                })
    return jsonify(records[:100])


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
