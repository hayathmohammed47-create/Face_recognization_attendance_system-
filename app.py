from flask import Flask, render_template, request, redirect, url_for, session, Response, flash
from pathlib import Path
from datetime import datetime
import csv, threading
import cv2
try:
    import face_recognition
    FACE_OK = True
except Exception:
    face_recognition = None
    FACE_OK = False

BASE = Path(__file__).resolve().parent
FACES = BASE / 'known_faces'
DATA = BASE / 'data'
CSV_FILE = DATA / 'attendance.csv'
app = Flask(__name__)
app.secret_key = 'change-this-demo-secret-key'
USERNAME, PASSWORD = 'admin', 'admin123'
TOLERANCE = 0.50
camera = None
camera_lock = threading.Lock()
known_encodings, known_names = [], []

def ensure_csv():
    DATA.mkdir(exist_ok=True); FACES.mkdir(exist_ok=True)
    if not CSV_FILE.exists():
        with CSV_FILE.open('w', newline='', encoding='utf-8') as f: csv.writer(f).writerow(['Name','Date','Time'])

def load_faces():
    global known_encodings, known_names
    known_encodings, known_names = [], []
    if not FACE_OK: return
    for p in sorted(FACES.iterdir()):
        if p.suffix.lower() not in ('.jpg','.jpeg','.png'): continue
        try:
            img = face_recognition.load_image_file(str(p)); enc = face_recognition.face_encodings(img)
            if enc:
                known_encodings.append(enc[0]); known_names.append(p.stem.replace('_',' ').replace('-',' ').title())
        except Exception as e: print('Could not load', p.name, e)

def mark_attendance(name):
    ensure_csv(); today = datetime.now().strftime('%Y-%m-%d')
    with CSV_FILE.open('r', newline='', encoding='utf-8') as f: rows = list(csv.DictReader(f))
    if any(r.get('Name') == name and r.get('Date') == today for r in rows): return
    with CSV_FILE.open('a', newline='', encoding='utf-8') as f: csv.writer(f).writerow([name, today, datetime.now().strftime('%H:%M:%S')])

def records():
    ensure_csv()
    with CSV_FILE.open('r', newline='', encoding='utf-8') as f: return list(csv.DictReader(f))

@app.route('/')
def home(): return redirect(url_for('dashboard')) if session.get('logged_in') else render_template('login.html')
@app.route('/login', methods=['POST'])
def login():
    if request.form.get('username','').strip() == USERNAME and request.form.get('password','') == PASSWORD:
        session['logged_in'] = True; return redirect(url_for('dashboard'))
    flash('Incorrect username or password.', 'error'); return redirect(url_for('home'))
@app.route('/dashboard')
def dashboard():
    if not session.get('logged_in'): return redirect(url_for('home'))
    return render_template('dashboard.html', total=len(records()), face_ok=FACE_OK)
@app.route('/attendance')
def attendance():
    if not session.get('logged_in'): return redirect(url_for('home'))
    return render_template('attendance.html', records=list(reversed(records())))
@app.route('/camera')
def camera_page():
    if not session.get('logged_in'): return redirect(url_for('home'))
    if not FACE_OK: flash('Face recognition package is not installed. See README.md.', 'error')
    return render_template('camera.html', face_ok=FACE_OK)
@app.route('/reload_faces', methods=['POST'])
def reload_faces():
    if not session.get('logged_in'): return redirect(url_for('home'))
    load_faces(); flash(f'Loaded {len(known_names)} registered face(s).', 'success'); return redirect(url_for('camera_page'))
def generate_frames():
    global camera
    with camera_lock:
        if camera is None or not camera.isOpened(): camera = cv2.VideoCapture(0)
    if camera is None or not camera.isOpened(): return
    while True:
        with camera_lock:
            if camera is None or not camera.isOpened(): break
            ok, frame = camera.read()
        if not ok: break
        label = 'Face recognition unavailable' if not FACE_OK else ('Add photos to known_faces/' if not known_encodings else 'Looking for registered face...')
        if FACE_OK and known_encodings:
            try:
                small = cv2.resize(frame, (0,0), fx=.25, fy=.25); rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                locs = face_recognition.face_locations(rgb, model='hog'); encs = face_recognition.face_encodings(rgb, locs)
                for enc, loc in zip(encs, locs):
                    matches = face_recognition.compare_faces(known_encodings, enc, tolerance=TOLERANCE)
                    distances = face_recognition.face_distance(known_encodings, enc); name = 'Unknown'
                    if len(distances):
                        best = distances.argmin()
                        if matches[best]: name = known_names[best]; mark_attendance(name)
                    top,right,bottom,left = [v*4 for v in loc]; color = (40,190,90) if name != 'Unknown' else (30,70,220)
                    cv2.rectangle(frame,(left,top),(right,bottom),color,2); cv2.rectangle(frame,(left,bottom-30),(right,bottom),color,cv2.FILLED)
                    cv2.putText(frame,name,(left+6,bottom-8),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2); label = 'Recognized: '+name
            except Exception as e: print('Recognition error:', e); label = 'Recognition error'
        cv2.putText(frame,label,(12,28),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2)
        ok, buf = cv2.imencode('.jpg',frame)
        if ok: yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n'
@app.route('/video_feed')
def video_feed():
    if not session.get('logged_in'): return redirect(url_for('home'))
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global camera
    with camera_lock:
        if camera is not None: camera.release(); camera = None
    return redirect(url_for('dashboard'))
@app.route('/logout')
def logout(): session.clear(); return redirect(url_for('home'))
if __name__ == '__main__':
    ensure_csv(); load_faces(); print('Face recognition available:', FACE_OK, '| Registered faces:', known_names)
    app.run(host='127.0.0.1', port=5000, debug=True, threaded=True)
