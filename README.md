# 🎓 AttendAI – Facial Recognition Attendance System

AttendAI is an AI-powered attendance management system that automates classroom attendance using facial recognition technology. Instead of manually marking attendance, teachers can upload a classroom image and the system automatically identifies registered students and records their attendance.

The project combines Computer Vision, Machine Learning, Web Development, and Database Management to provide an efficient, secure, and scalable attendance solution.

---

## 🚀 Features

### 👨‍🎓 Student Module

* Student registration and authentication
* Face sample collection and registration
* Personal attendance dashboard
* Attendance percentage tracking
* Attendance history and records

### 👨‍🏫 Teacher Module

* Teacher login and authentication
* Upload classroom photographs
* Automatic attendance marking
* View attendance records
* Filter attendance by student/date
* Export attendance data to CSV

### 🤖 Facial Recognition System

* Face detection using Haar Cascade Classifiers
* Face recognition using OpenCV LBPH Recognizer
* Multi-face recognition in group photographs
* Confidence-based matching
* Automatic attendance recording
* Duplicate attendance prevention
* Automatic model retraining when new students are registered

---

## 🛠️ Tech Stack

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
