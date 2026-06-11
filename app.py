"""
AttendAI – Facial Recognition Attendance System
================================================
Stack : Flask · SQLite · OpenCV LBPH · NumPy
No dlib · No TensorFlow · No DeepFace · No CMake
"""

import os
import io
import csv
import sqlite3
import numpy as np
from datetime import date, datetime
from functools import wraps

import cv2
from flask import (Flask, render_template, request, redirect, url_for,
                   session, flash, send_file, g)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

# ── Config ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = "attendai_secret_2024"

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATABASE      = os.path.join(BASE_DIR, "database.db")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static",   "uploads")
FACE_DATA_DIR = os.path.join(BASE_DIR, "face_data")
MODEL_PATH    = os.path.join(BASE_DIR, "face_data", "lbph_model.yml")
EXPORTS_DIR   = os.path.join(BASE_DIR, "exports")

for _d in [UPLOAD_FOLDER, FACE_DATA_DIR, EXPORTS_DIR]:
    os.makedirs(_d, exist_ok=True)

ALLOWED_EXT   = {"png", "jpg", "jpeg"}
LBPH_CONF_MAX = 85        # lower confidence = better match in LBPH

# Haar cascade (ships with opencv-python, no extra download)
CASCADE_PATH  = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
FACE_CASCADE  = cv2.CascadeClassifier(CASCADE_PATH)

# ── Database ────────────────────────────────────────────────────────────────
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DATABASE)
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA journal_mode=WAL")
    return g.db

@app.teardown_appcontext
def close_db(exc):
    db = g.pop("db", None)
    if db:
        db.close()

def init_db():
    db = sqlite3.connect(DATABASE)
    db.row_factory = sqlite3.Row
    db.executescript("""
        CREATE TABLE IF NOT EXISTS students (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            name       TEXT    NOT NULL,
            roll_no    TEXT    NOT NULL UNIQUE,
            email      TEXT    NOT NULL UNIQUE,
            password   TEXT    NOT NULL,
            created_at TEXT    DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS teachers (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            name       TEXT    NOT NULL,
            email      TEXT    NOT NULL UNIQUE,
            password   TEXT    NOT NULL,
            created_at TEXT    DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS face_samples (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            image_path TEXT    NOT NULL,
            created_at TEXT    DEFAULT (datetime('now')),
            FOREIGN KEY (student_id) REFERENCES students(id)
        );
        CREATE TABLE IF NOT EXISTS attendance (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id   INTEGER NOT NULL,
            student_name TEXT    NOT NULL,
            roll_no      TEXT    NOT NULL,
            date         TEXT    NOT NULL,
            status       TEXT    NOT NULL DEFAULT 'Present',
            marked_by    TEXT,
            created_at   TEXT    DEFAULT (datetime('now')),
            FOREIGN KEY (student_id) REFERENCES students(id),
            UNIQUE (student_id, date)
        );
    """)
    db.commit()
    if not db.execute("SELECT id FROM teachers LIMIT 1").fetchone():
        db.execute(
            "INSERT INTO teachers (name, email, password) VALUES (?,?,?)",
            ("Admin Teacher", "teacher@college.edu",
             generate_password_hash("teacher123"))
        )
        db.commit()
    db.close()

# ── Helpers ─────────────────────────────────────────────────────────────────
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

def login_required(role):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            if role == "student" and "student_id" not in session:
                flash("Please log in as a student.", "warning")
                return redirect(url_for("student_login"))
            if role == "teacher" and "teacher_id" not in session:
                flash("Please log in as a teacher.", "warning")
                return redirect(url_for("teacher_login"))
            return f(*args, **kwargs)
        return wrapper
    return decorator

# ── LBPH helpers ─────────────────────────────────────────────────────────────

def extract_face_gray(image_path: str):
    """
    Read an image, detect the largest face, return a resized 200x200
    grayscale crop.  Returns None if no face found.
    """
    img  = cv2.imread(image_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Equalise histogram for lighting robustness
    gray = cv2.equalizeHist(gray)
    faces = FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60)
    )
    if len(faces) == 0:
        return None
    # Pick the largest face
    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
    face_crop = gray[y:y+h, x:x+w]
    return cv2.resize(face_crop, (200, 200))


def train_model():
    """
    Train LBPH on all stored face samples and save to MODEL_PATH.
    Returns (True, n_students) or (False, error_message).
    """
    db = sqlite3.connect(DATABASE)
    db.row_factory = sqlite3.Row
    rows = db.execute(
        "SELECT fs.student_id, fs.image_path "
        "FROM face_samples fs"
    ).fetchall()
    db.close()

    faces, labels = [], []
    for row in rows:
        face = extract_face_gray(row["image_path"])
        if face is not None:
            faces.append(face)
            labels.append(int(row["student_id"]))

    if len(faces) < 1:
        return False, "No valid face samples found to train on."

    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=1, neighbors=8, grid_x=8, grid_y=8
    )
    recognizer.train(faces, np.array(labels))
    recognizer.save(MODEL_PATH)
    n_students = len(set(labels))
    return True, n_students


def load_recognizer():
    """Load saved LBPH model. Returns None if no model exists yet."""
    if not os.path.exists(MODEL_PATH):
        return None
    rec = cv2.face.LBPHFaceRecognizer_create()
    rec.read(MODEL_PATH)
    return rec


def process_classroom_image(image_path: str):
    """
    Detect all faces in a classroom image, run LBPH on each,
    return list of matched student dicts.
    """
    recognizer = load_recognizer()
    if recognizer is None:
        return [], "Model not trained yet. Ask students to upload face images first."

    img  = cv2.imread(image_path)
    if img is None:
        return [], "Could not read uploaded image."

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.15, minNeighbors=4, minSize=(50, 50)
    )

    if len(faces) == 0:
        return [], "No faces detected in the image."

    db = sqlite3.connect(DATABASE)
    db.row_factory = sqlite3.Row

    matched  = []
    seen_ids = set()

    for (x, y, w, h) in faces:
        face_crop = gray[y:y+h, x:x+w]
        face_crop = cv2.resize(face_crop, (200, 200))

        student_id, confidence = recognizer.predict(face_crop)

        if confidence <= LBPH_CONF_MAX and student_id not in seen_ids:
            student = db.execute(
                "SELECT id, name, roll_no FROM students WHERE id=?",
                (student_id,)
            ).fetchone()
            if student:
                seen_ids.add(student_id)
                matched.append({
                    "student_id": student["id"],
                    "name":       student["name"],
                    "roll_no":    student["roll_no"],
                    "confidence": round(confidence, 1),
                })

    db.close()
    return matched, None


# ── Routes: general ──────────────────────────────────────────────────────────
@app.route("/")
def index():
    return redirect(url_for("student_login"))


# ── Routes: student auth ─────────────────────────────────────────────────────
@app.route("/student/register", methods=["GET", "POST"])
def student_register():
    if request.method == "POST":
        name    = request.form["name"].strip()
        roll_no = request.form["roll_no"].strip()
        email   = request.form["email"].strip().lower()
        pw      = request.form["password"]
        db = get_db()
        try:
            db.execute(
                "INSERT INTO students (name, roll_no, email, password) VALUES (?,?,?,?)",
                (name, roll_no, email, generate_password_hash(pw))
            )
            db.commit()
            flash("Account created! Please log in.", "success")
            return redirect(url_for("student_login"))
        except sqlite3.IntegrityError:
            flash("Roll number or email already registered.", "danger")
    return render_template("register.html")


@app.route("/student/login", methods=["GET", "POST"])
def student_login():
    if request.method == "POST":
        roll_no = request.form["roll_no"].strip()
        pw      = request.form["password"]
        db = get_db()
        s = db.execute(
            "SELECT * FROM students WHERE roll_no=?", (roll_no,)
        ).fetchone()
        if s and check_password_hash(s["password"], pw):
            session.update({
                "student_id":   s["id"],
                "student_name": s["name"],
                "student_roll": s["roll_no"],
            })
            return redirect(url_for("student_dashboard"))
        flash("Invalid roll number or password.", "danger")
    return render_template("login.html", role="student")


@app.route("/student/logout")
def student_logout():
    session.clear()
    return redirect(url_for("student_login"))


# ── Routes: teacher auth ─────────────────────────────────────────────────────
@app.route("/teacher/login", methods=["GET", "POST"])
def teacher_login():
    if request.method == "POST":
        email = request.form["email"].strip().lower()
        pw    = request.form["password"]
        db = get_db()
        t = db.execute(
            "SELECT * FROM teachers WHERE email=?", (email,)
        ).fetchone()
        if t and check_password_hash(t["password"], pw):
            session.update({
                "teacher_id":   t["id"],
                "teacher_name": t["name"],
            })
            return redirect(url_for("teacher_dashboard"))
        flash("Invalid email or password.", "danger")
    return render_template("login.html", role="teacher")


@app.route("/teacher/logout")
def teacher_logout():
    session.clear()
    return redirect(url_for("teacher_login"))


# ── Routes: student dashboard ────────────────────────────────────────────────
@app.route("/student/dashboard")
@login_required("student")
def student_dashboard():
    db  = get_db()
    sid = session["student_id"]

    records = db.execute(
        "SELECT date, status FROM attendance WHERE student_id=? ORDER BY date DESC",
        (sid,)
    ).fetchall()

    total   = len(records)
    present = sum(1 for r in records if r["status"] == "Present")
    pct     = round(present / total * 100, 1) if total else 0

    sample_count = db.execute(
        "SELECT COUNT(*) as c FROM face_samples WHERE student_id=?", (sid,)
    ).fetchone()["c"]

    face_images = db.execute(
        "SELECT id, image_path FROM face_samples WHERE student_id=?",
        (sid,)
    ).fetchall()

    return render_template(
        "student_dashboard.html",
        records=records,
        total=total,
        present=present,
        pct=pct,
        sample_count=sample_count,
        face_images=face_images
    )


@app.route("/student/upload_faces", methods=["POST"])
@login_required("student")
def upload_faces():
    db  = get_db()
    sid = session["student_id"]

    existing = db.execute(
        "SELECT COUNT(*) as c FROM face_samples WHERE student_id=?", (sid,)
    ).fetchone()["c"]

    files = request.files.getlist("face_images")
    if not files or all(f.filename == "" for f in files):
        flash("No files selected.", "warning")
        return redirect(url_for("student_dashboard"))

    saved = 0
    for f in files:
        if existing + saved >= 5:
            flash("Maximum 5 face images allowed.", "warning")
            break
        if not (f and allowed_file(f.filename)):
            continue

        fname = secure_filename(
            f"{sid}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}_{f.filename}"
        )
        path = os.path.join(FACE_DATA_DIR, fname)
        f.save(path)

        # Validate a face is actually present
        face = extract_face_gray(path)
        if face is None:
            os.remove(path)
            flash(f"No face detected in '{f.filename}'. Skipped.", "warning")
            continue

        db.execute(
            "INSERT INTO face_samples (student_id, image_path) VALUES (?,?)",
            (sid, path)
        )
        db.commit()
        saved += 1

    if saved:
        # Re-train model with new samples
        ok, info = train_model()
        if ok:
            flash(f"{saved} image(s) saved. Model retrained on {info} student(s). ✅", "success")
        else:
            flash(f"{saved} image(s) saved but model training failed: {info}", "warning")
    return redirect(url_for("student_dashboard"))


# ── Routes: teacher dashboard ────────────────────────────────────────────────
@app.route("/teacher/dashboard")
@login_required("teacher")
def teacher_dashboard():
    db = get_db()
    students = db.execute(
        "SELECT s.*, COUNT(fs.id) as sample_count "
        "FROM students s "
        "LEFT JOIN face_samples fs ON s.id = fs.student_id "
        "GROUP BY s.id ORDER BY s.name"
    ).fetchall()

    today_att = db.execute(
        "SELECT COUNT(*) as c FROM attendance WHERE date=?",
        (str(date.today()),)
    ).fetchone()["c"]

    model_exists = os.path.exists(MODEL_PATH)

    return render_template("teacher_dashboard.html",
                           students=students,
                           today_att=today_att,
                           total_students=len(students),
                           today=str(date.today()),
                           model_exists=model_exists)


@app.route("/teacher/mark_attendance", methods=["POST"])
@login_required("teacher")
def mark_attendance():
    if "classroom_image" not in request.files:
        flash("No image uploaded.", "warning")
        return redirect(url_for("teacher_dashboard"))

    f = request.files["classroom_image"]
    if not f.filename or not allowed_file(f.filename):
        flash("Invalid file type. Use JPG or PNG.", "danger")
        return redirect(url_for("teacher_dashboard"))

    att_date = request.form.get("att_date", str(date.today()))

    fname = secure_filename(
        f"classroom_{datetime.now().strftime('%Y%m%d%H%M%S')}_{f.filename}"
    )
    path = os.path.join(UPLOAD_FOLDER, fname)
    f.save(path)

    matched, err = process_classroom_image(path)

    if err and not matched:
        flash(err, "warning")
        return redirect(url_for("teacher_dashboard"))

    db = get_db()
    new_marked = 0
    already    = 0
    for m in matched:
        try:
            db.execute(
                "INSERT INTO attendance "
                "(student_id, student_name, roll_no, date, status, marked_by) "
                "VALUES (?,?,?,?,?,?)",
                (m["student_id"], m["name"], m["roll_no"],
                 att_date, "Present", session["teacher_name"])
            )
            new_marked += 1
        except sqlite3.IntegrityError:
            already += 1
    db.commit()

    if new_marked:
        msg = f"✅ {new_marked} student(s) marked Present for {att_date}."
        if already:
            msg += f" ({already} already recorded.)"
        flash(msg, "success")
    elif already:
        flash(f"All detected students were already marked for {att_date}.", "info")
    else:
        flash("No registered faces detected. Check lighting or retrain.", "warning")

    return redirect(url_for("view_attendance"))


# ── Routes: teacher – retrain manually ───────────────────────────────────────
@app.route("/teacher/retrain", methods=["POST"])
@login_required("teacher")
def retrain():
    ok, info = train_model()
    if ok:
        flash(f"Model retrained successfully on {info} student(s). ✅", "success")
    else:
        flash(f"Training failed: {info}", "danger")
    return redirect(url_for("teacher_dashboard"))


@app.route("/teacher/attendance")
@login_required("teacher")
def view_attendance():

    db = get_db()

    filter_date = request.args.get("date", "").strip()
    filter_roll = request.args.get("roll_no", "").strip()

    query = "SELECT * FROM attendance WHERE 1=1"
    params = []

    if filter_date:
        query += " AND date=?"
        params.append(filter_date)

    if filter_roll:
        query += """
            AND (
                roll_no LIKE ?
                OR student_name LIKE ?
            )
        """
        params.append(f"%{filter_roll}%")
        params.append(f"%{filter_roll}%")

    query += " ORDER BY date DESC, student_name"

    records = db.execute(query, params).fetchall()

    all_dates = db.execute(
        "SELECT DISTINCT date FROM attendance ORDER BY date DESC"
    ).fetchall()

    return render_template(
        "attendance.html",
        records=records,
        all_dates=all_dates,
        filter_date=filter_date,
        filter_roll=filter_roll
    )

# ── Routes: export CSV ────────────────────────────────────────────────────────
@app.route("/teacher/export_csv")
@login_required("teacher")
def export_csv():
    db = get_db()
    filter_date = request.args.get("date", "")

    query  = ("SELECT student_name, roll_no, date, status, marked_by "
              "FROM attendance WHERE 1=1")
    params = []
    if filter_date:
        query  += " AND date=?"
        params.append(filter_date)
    query += " ORDER BY date DESC, student_name"

    rows = db.execute(query, params).fetchall()

    si = io.StringIO()
    w  = csv.writer(si)
    w.writerow(["Student Name", "Roll No", "Date", "Status", "Marked By"])
    for r in rows:
        w.writerow([r["student_name"], r["roll_no"], r["date"],
                    r["status"], r["marked_by"] or ""])

    output = io.BytesIO()
    output.write(si.getvalue().encode("utf-8-sig"))
    output.seek(0)

    fname = (f"attendance_{filter_date or 'all'}_"
             f"{datetime.now().strftime('%Y%m%d%H%M%S')}.csv")
    return send_file(output, mimetype="text/csv",
                     as_attachment=True, download_name=fname)


# ── Routes: delete student face data ─────────────────────────────────────────
@app.route("/teacher/delete_faces/<int:student_id>", methods=["POST"])
@login_required("teacher")
def delete_faces(student_id):
    db = get_db()
    rows = db.execute(
        "SELECT image_path FROM face_samples WHERE student_id=?", (student_id,)
    ).fetchall()
    for r in rows:
        try:
            if r["image_path"] and os.path.exists(r["image_path"]):
                os.remove(r["image_path"])
        except Exception:
            pass
    db.execute("DELETE FROM face_samples WHERE student_id=?", (student_id,))
    db.commit()

    # Retrain without this student
    remaining = db.execute("SELECT COUNT(*) as c FROM face_samples").fetchone()["c"]
    if remaining > 0:
        train_model()
    elif os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)

    flash("Face data cleared and model retrained.", "success")
    return redirect(url_for("teacher_dashboard"))


# ── Routes: student – delete individual face image ────────────────────────────
@app.route("/student/delete_face/<int:face_id>", methods=["POST"])
@login_required("student")
def delete_face(face_id):
    db = get_db()
    face = db.execute(
        "SELECT * FROM face_samples WHERE id=? AND student_id=?",
        (face_id, session["student_id"])
    ).fetchone()
    if not face:
        flash("Face image not found.", "danger")
        return redirect(url_for("student_dashboard"))
    try:
        if os.path.exists(face["image_path"]):
            os.remove(face["image_path"])
    except Exception:
        pass
    db.execute(
        "DELETE FROM face_samples WHERE id=?",
        (face_id,)
    )
    db.commit()
    remaining = db.execute(
        "SELECT COUNT(*) as c FROM face_samples"
    ).fetchone()["c"]
    if remaining > 0:
        train_model()
    elif os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
    flash("Face image deleted successfully.", "success")
    return redirect(url_for("student_dashboard"))


@app.route("/face_image/<int:face_id>")
@login_required("student")
def face_image(face_id):

    db = get_db()

    face = db.execute(
        """
        SELECT *
        FROM face_samples
        WHERE id=? AND student_id=?
        """,
        (face_id, session["student_id"])
    ).fetchone()

    if not face:
        return "", 404

    return send_file(face["image_path"])

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    init_db()
    print("\n" + "=" * 54)
    print("  AttendAI — Facial Recognition Attendance System")
    print("  Running at  →  http://127.0.0.1:5000")
    print("  Teacher login →  teacher@college.edu / teacher123")
    print("=" * 54 + "\n")
    app.run(debug=True, host="127.0.0.1", port=5000)