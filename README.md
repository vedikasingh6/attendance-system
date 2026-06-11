# 🎓 AttendAI – Facial Recognition Attendance System

AttendAI is a web-based attendance management system that automates student attendance using facial recognition. The application enables students to register face samples, trains a recognition model using OpenCV's Local Binary Pattern Histogram (LBPH) algorithm, and allows teachers to mark attendance automatically from classroom photographs.

Built using Flask, OpenCV, SQLite, HTML, CSS, and JavaScript, the system provides role-based authentication, attendance tracking, attendance analytics, CSV export functionality, and persistent local data storage.

## Key Features

### Student Portal

* Secure student registration and login
* Face image registration for model training
* Attendance percentage tracking and analytics
* Attendance history dashboard

### Teacher Portal

* Secure teacher authentication
* Automatic attendance marking from classroom images
* Student management and face-sample monitoring
* Attendance filtering and record management
* CSV export of attendance records

### Facial Recognition Pipeline

* Face detection using Haar Cascade Classifiers
* Face recognition using OpenCV LBPH Face Recognizer
* Multi-face detection from classroom photographs
* Confidence-based identity matching
* Duplicate attendance prevention
* Automatic model retraining after new face registrations

## Technology Stack

| Layer           | Technology              |
| --------------- | ----------------------- |
| Backend         | Python, Flask           |
| Database        | SQLite                  |
| Computer Vision | OpenCV, NumPy           |
| Frontend        | HTML5, CSS3, JavaScript |
| Authentication  | Flask Sessions          |

## Project Highlights

* Designed a role-based attendance management platform with separate student and teacher workflows.
* Implemented a complete facial recognition pipeline using OpenCV for automated attendance marking.
* Built a persistent SQLite database for managing students, attendance records, and face samples.
* Developed attendance analytics, search, filtering, and CSV export functionality.
* Created a responsive dashboard interface for attendance monitoring and management.
