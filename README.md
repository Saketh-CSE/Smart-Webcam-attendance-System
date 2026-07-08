# Smart Attendance System – Facial Recognition Attendance

A Flask and OpenCV face-recognition attendance app with a built-in web frontend.

## Features

- Register students with webcam face capture.
- Train an LBPH face-recognition model automatically after registration.
- Mark attendance from live camera recognition.
- View recent attendance records in the browser.
- Single main entry point: `attendance.py`.

## Setup

Install Python dependencies:

```powershell
pip install -r requirements.txt
```

Run the app:

```powershell
python attendance.py
```

Open the browser:

```text
http://127.0.0.1:5000/
```

## Project Files

- `attendance.py` - main Flask app, backend routes, and embedded frontend.
- `requirements.txt` - Python dependencies.
- `haarcascade_frontalface_default.xml` - OpenCV face detector.
- `AMS.ico` - favicon.
- `TrainingImage/`, `StudentDetails/`, and `Attendance/` are runtime data folders and are ignored by Git.

## Notes

- Use Chrome or Edge and allow camera permission.
- Keep the student's face centered during registration.
- Restart the Flask server after code changes.
