from flask import Flask, render_template, Response, request, redirect
import main
import pandas as pd
import os
import datetime
import cv2
from PIL import Image
import numpy as np

app = Flask(__name__)
camera = cv2.VideoCapture(0)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("TrainingImageLabel/Trainner.yml")

face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)
marked_ids = set()

def mark_attendance_web(emp_id, name):

    today = datetime.datetime.now().strftime("%d-%m-%Y")
    now_time = datetime.datetime.now().strftime("%H:%M:%S")

    file_path = "Attendance/Attendance_" + today + ".csv"

    columns = ["Id", "Name", "Date", "In Time", "Out Time", "Status"]

    if os.path.isfile(file_path):
        df_att = pd.read_csv(file_path)
    else:
        df_att = pd.DataFrame(columns=columns)

    existing = df_att[df_att["Id"].astype(str) == str(emp_id)]

    if existing.empty:
        new_row = {
            "Id": emp_id,
            "Name": name,
            "Date": today,
            "In Time": now_time,
            "Out Time": "",
            "Status": "Present"
        }

        df_att = pd.concat([df_att, pd.DataFrame([new_row])], ignore_index=True)

    else:
        index = existing.index[0]

        if pd.isna(df_att.loc[index, "Out Time"]) or df_att.loc[index, "Out Time"] == "":
            df_att.loc[index, "Out Time"] = now_time

    df_att.to_csv(file_path, index=False)


def generate_frames():

    while True:

        success, frame = camera.read()

        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.2, 5)

        try:
            df = pd.read_csv("StudentDetails/StudentDetails.csv")
        except:
            df = pd.DataFrame()

        for (x, y, w, h) in faces:

            serial, conf = recognizer.predict(
                gray[y:y+h, x:x+w]
            )

            name = "Unknown"

            if conf < 70:

                try:
                    aa = df.loc[df['SERIAL NO.'] == serial]['NAME'].values
                    if len(aa) > 0:
                        name = str(aa[0])

                    emp_id_values = df.loc[df['SERIAL NO.'] == serial]['ID'].values
                    if len(emp_id_values) > 0:
                        emp_id = str(emp_id_values[0])
                        if emp_id not in marked_ids:
                            mark_attendance_web(emp_id, name)
                            marked_ids.add(emp_id)
                except Exception:
                    pass

                # Face box
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                # Top name background
                cv2.rectangle(frame, (x, y-35), (x+w, y), (0, 255, 0), -1)

                # Name text
                cv2.putText(
                    frame,
                    name,
                    (x + 8, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 0),
                    2
                )

                # Corner lines
                corner_length = 25
                corner_color = (255, 255, 255)
                corner_thickness = 3

                cv2.line(frame, (x, y), (x + corner_length, y), corner_color, corner_thickness)
                cv2.line(frame, (x, y), (x, y + corner_length), corner_color, corner_thickness)

                cv2.line(frame, (x+w, y), (x+w - corner_length, y), corner_color, corner_thickness)
                cv2.line(frame, (x+w, y), (x+w, y + corner_length), corner_color, corner_thickness)

                cv2.line(frame, (x, y+h), (x + corner_length, y+h), corner_color, corner_thickness)
                cv2.line(frame, (x, y+h), (x, y+h - corner_length), corner_color, corner_thickness)

                cv2.line(frame, (x+w, y+h), (x+w - corner_length, y+h), corner_color, corner_thickness)
                cv2.line(frame, (x+w, y+h), (x+w, y+h - corner_length), corner_color, corner_thickness)
            else:
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x+w, y+h),
                    (0, 0, 255),
                    2
                )

                cv2.putText(
                    frame,
                    "Unknown",
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2
                )

        ret, buffer = cv2.imencode('.jpg', frame)

        frame = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
        )


def get_dashboard_data():
    today = datetime.datetime.now().strftime("%d-%m-%Y")

    student_file = "StudentDetails/StudentDetails.csv"
    attendance_file = f"Attendance/Attendance_{today}.csv"

    total_employees = 0
    present_today = 0
    late_entries = 0
    absent = 0
    attendance_rows = []

    if os.path.isfile(student_file):
        df_students = pd.read_csv(student_file)
        df_students = df_students.drop_duplicates(subset=["ID"], keep="last")
        total_employees = len(df_students)

    if os.path.isfile(attendance_file):
        df_att = pd.read_csv(attendance_file)
        df_att = df_att.drop_duplicates(subset=["Id"], keep="last")
        present_today = len(df_att)

        if "In Time" in df_att.columns:
            late_entries = len(df_att[df_att["In Time"].astype(str) > "09:30:00"])

        attendance_rows = df_att.to_dict(orient="records")

    absent = max(total_employees - present_today, 0)

    return {
        "total": total_employees,
        "present": present_today,
        "late": late_entries,
        "absent": absent,
        "rows": attendance_rows,
        "today": today
    }


@app.route("/")
def dashboard():
    data = get_dashboard_data()
    return render_template("dashboard.html", data=data)

@app.route("/attendance")
def attendance():
    return render_template("attendance.html")

@app.route("/register")
def register():
    return render_template("register.html")

@app.route("/admin")
def admin():
    employees = []

    student_file = "StudentDetails/StudentDetails.csv"

    if os.path.isfile(student_file):
        df = pd.read_csv(student_file)
        df = df.drop_duplicates(subset=["ID"], keep="last")
        employees = df.to_dict(orient="records")

    return render_template(
        "admin.html",
        employees=employees
    )


@app.route("/logout")
def logout():
    return render_template("logout.html")


@app.route("/reports")
def reports():

    today = datetime.datetime.now().strftime("%d-%m-%Y")

    attendance_file = f"Attendance/Attendance_{today}.csv"

    attendance = []

    if os.path.isfile(attendance_file):
        df = pd.read_csv(attendance_file)
        attendance = df.to_dict(orient="records")

    return render_template(
        "reports.html",
        attendance=attendance
    )


@app.route("/settings")
def settings():
    return render_template("settings.html")    

    emp_id = request.form["emp_id"]
    emp_name = request.form["emp_name"]

    class FakeEntry:
        def __init__(self, value):
            self.value = value

        def get(self):
            return self.value
        
    main.txt = FakeEntry(emp_id)
    main.txt2 = FakeEntry(emp_name)    

    main.TakeImages()
    main.TrainImages()

    return redirect("/")

@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)