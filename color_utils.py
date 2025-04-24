import numpy as np
import json
import cv2
from ultralytics import YOLO

# تحميل الألوان من ملف JSON
with open("colors.json", "r") as f:
    colors = json.load(f)

team1_colors = colors["Team1"]
team2_colors = colors["Team2"]
referee_colors = colors["Referee"]  # الألوان التي تم تحديدها للحكم

def color_distance(c1, c2):
    return np.linalg.norm(np.array(c1) - np.array(c2))

def classify_team(color):
    dist1 = min([color_distance(color, ref) for ref in team1_colors])
    dist2 = min([color_distance(color, ref) for ref in team2_colors])
    dist_ref = min([color_distance(color, ref) for ref in referee_colors])  # مقارنة الألوان مع ألوان الحكم

    # إذا كانت المسافة أقرب إلى ألوان الحكم، نقوم بتصنيفه كحكم
    if dist_ref < dist1 and dist_ref < dist2:
        return "Referee"
    elif dist1 < dist2:
        return "Team1"
    else:
        return "Team2"

# تحميل نموذج YOLO
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("video.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            class_name = result.names[cls_id]

            # التعامل مع اللاعبين
            if class_name == "person":
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = frame[y1:y2, x1:x2]
                shirt_crop = crop[int((y2-y1)*0.2):int((y2-y1)*0.6), int((x2-x1)*0.25):int((x2-x1)*0.75)]

                try:
                    dominant_color = cv2.mean(shirt_crop)[:3]
                    color = tuple(map(int, dominant_color))
                    role = classify_team(color)

                    # تحديد اللون المميز لكل دور (فريق 1 أو فريق 2 أو حكم)
                    if role == "Team1":
                        color_box = (0, 255, 0)  # لون الفريق الأول (أخضر)
                    elif role == "Team2":
                        color_box = (255, 0, 0)  # لون الفريق الثاني (أزرق)
                    elif role == "Referee":
                        color_box = (0, 255, 255)  # لون الحكم (أصفر)
                    else:
                        color_box = (255, 255, 255)  # اللون الافتراضي (أبيض)

                    # رسم مستطيل حول الشخص مع تحديد النص
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color_box, 2)
                    cv2.putText(frame, role, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_box, 2)
                except:
                    continue

            # التعامل مع الكرة
            elif class_name == "sports ball":
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # رسم دائرة حول الكرة
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                radius = min(x2 - x1, y2 - y1) // 2
                cv2.circle(frame, center, radius, (0, 255, 255), 2)  # لون أصفر للكرة
                cv2.putText(frame, "Ball", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # عرض الإطار المعدل
    cv2.imshow("Match Analysis", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 