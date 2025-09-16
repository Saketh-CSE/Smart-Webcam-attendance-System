Face Recognition Attendance System
This is a web-based attendance management system that uses facial recognition technology to securely and efficiently mark student attendance. Built with a Python backend using Flask, the system provides a robust solution for educational institutions to automate the attendance process.

Key Features
Secure Student Registration: A web-based interface allows for the registration of new students by capturing their facial data via a webcam. Facial encodings are securely stored for future recognition.

Automated Attendance Marking: Students can mark their attendance for a specific subject simply by appearing in front of a webcam. The system uses a live video stream to detect and identify faces in real-time.

Liveness Detection: To prevent spoofing, the system employs blink detection using facial landmarks. This ensures that a live person is present and not just a photograph or a video.

Geolocation Verification: Attendance can only be marked if the user is within a predefined geographical radius of the campus, adding an extra layer of security and accuracy.

Real-time Visual Feedback: The application provides live visual cues on the video feed. A green box and facial landmarks appear on a recognized face, while a red box indicates an unknown individual.

Data Management: Attendance records are logged in a structured CSV file, and facial encodings are stored in a dedicated pickle file, ensuring data integrity and easy access.

Intuitive Web Interface: A clean and modern web interface allows users to easily navigate between registering students, marking attendance, and viewing attendance records.

Technology Stack
Backend: Flask (Python)

Frontend: HTML, CSS, JavaScript

Computer Vision: OpenCV, face_recognition, dlib

Data Handling: numpy, pickle, csv, os

Other Libraries: requests, math

Setup and Installation
Clone the repository:

Bash

git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
Install Python dependencies:

Bash

pip install Flask opencv-python dlib face_recognition numpy
(Note: dlib can be tricky to install. You may need to install CMake and Visual C++ Build Tools first.)

Download the dlib shape predictor file:

Download shape_predictor_68_face_landmarks.dat.bz2 from dlib.net.

Extract the .dat file and place it in the project's root directory.

Run the application:

Bash

python app.py
Usage
Register a Student:

Navigate to the /register page.

Enter the student's name and registration number.

The webcam will activate to capture facial data. The system will take multiple images and save the facial encoding.

Mark Attendance:

Navigate to the /attendance page.

Select a subject and click "Start Attendance".

The webcam will begin streaming. Look at the camera and blink to confirm your attendance.

View Records:

Navigate to the /view_attendance page to see all the attendance records stored in the system.
