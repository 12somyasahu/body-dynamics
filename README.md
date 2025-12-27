

---

# 🧍 Body Dynamics

**Live human pose tracking and body movement analysis in the browser.**

Body Dynamics lets you use your webcam to see real-time body pose landmarks and movement statistics directly on screen. It runs locally on your machine and does not upload video anywhere.

---

## ✨ What it does

* Uses your webcam to detect human body pose
* Shows live pose landmarks (red dots) on video
* Displays real-time performance stats (FPS)
* Calculates body movement metrics (e.g. elbow angle)
* Runs fully on your computer (no cloud, no uploads)

---

## 🖥️ How to use

### 1️⃣ Start the backend

```bash
cd backend
venv\Scripts\activate   # Windows
python -m uvicorn app.main:app
```

You should see the server running at:

```
http://127.0.0.1:8000
```

---

### 2️⃣ Open the frontend

* Open `frontend/cam_test.html` in your browser
* Allow camera access when prompted

---

### 3️⃣ What you’ll see

* Live camera feed
* Red dots showing detected body joints
* FPS counter
* Live movement stats (e.g. elbow angle)

Try bending and straightening your arm — the angle value will change in real time.

---

## 📐 Current stats shown

* **FPS** – how fast frames are processed
* **Left elbow angle** – measured in degrees

More stats will be added over time.

---

## ⚠️ Notes for best results

* Make sure your full upper body is visible
* Use good lighting
* Stand 1–2 meters from the camera
* Only one person should be in frame for best accuracy

---

## 🔒 Privacy

* Video stays on your device
* No data is sent to any server other than your local machine
* No recordings are saved

---

## 🚧 Work in progress

This project is actively being developed. Upcoming features include:

* More joint angles
* Skeleton lines
* Balance and movement analysis
* Sports-specific metrics




