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

| Category         | Technologies          |
| ---------------- | --------------------- |
| Backend          | Python, Flask         |
| Database         | SQLite                |
| Computer Vision  | OpenCV, NumPy         |
| Frontend         | HTML, CSS, JavaScript |
| Authentication   | Flask Sessions        |
| Machine Learning | LBPH Face Recognition |

---

## 📂 Project Structure

```text
AttendAI/
│
├── app.py
├── database.db
├── requirements.txt
│
├── static/
│   ├── css/
│   ├── js/
│   └── uploads/
│
├── templates/
│   ├── login.html
│   ├── register.html
│   ├── teacher_dashboard.html
│   └── student_dashboard.html
│
├── dataset/
│   ├── student_1/
│   ├── student_2/
│   └── ...
│
├── trainer/
│   └── trainer.yml
│
└── README.md
```

---

## ⚙️ Installation

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/AttendAI.git
cd AttendAI
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

Activate Environment:

**Windows**

```bash
venv\Scripts\activate
```

**Linux / Mac**

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run Application

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

---


## 🔄 Workflow

1. Student registers an account.
2. Student uploads face samples.
3. System trains the LBPH recognition model.
4. Teacher uploads a classroom image.
5. Faces are detected and recognized.
6. Attendance records are automatically generated.
7. Students can monitor attendance through their dashboard.

---

## 📊 Database Tables

### Students

* Student ID
* Name
* Email
* Password

### Attendance

* Attendance ID
* Student ID
* Date
* Status

### Teachers

* Teacher ID
* Name
* Email
* Password

---

## 🎯 Project Highlights

* Built a complete end-to-end attendance automation system.
* Implemented facial recognition using OpenCV and LBPH.
* Designed role-based access control for teachers and students.
* Integrated attendance analytics and reporting.
* Developed CSV export functionality for attendance records.
* Created a responsive web interface using Flask and JavaScript.

---



## 👩‍💻 Author

**Vedika Singh**

Computer and Communication Engineering Student



```
```
