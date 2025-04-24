import cv2
import numpy as np
import json

selected_colors = {"Team1": [], "Team2": [], "Referee": []}
current_team = "Team1"

def get_dominant_color(crop, k=2):
    crop = cv2.resize(crop, (50, 50))
    data = crop.reshape((-1, 3)).astype("float32")
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    dominant = centers[np.argmax(np.bincount(labels.flatten()))]
    return tuple(map(int, dominant))

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        crop = frame[y-30:y+30, x-20:x+20]
        color = get_dominant_color(crop)
        selected_colors[current_team].append(color)
        print(f"{current_team} ->", color)

cap = cv2.VideoCapture("video.mp4")
ret, frame = cap.read()

cv2.namedWindow("Pick Players")
cv2.setMouseCallback("Pick Players", mouse_callback)

# اختيار لاعبي Team1
print("اختر 2 لاعبين من Team1 (اضغط عليهم)، ثم اضغط Enter")
cv2.imshow("Pick Players", frame)
cv2.waitKey(0)

# تغيير الفريق إلى Team2
current_team = "Team2"
print("اختر 2 لاعبين من Team2، ثم اضغط Enter")
cv2.imshow("Pick Players", frame)
cv2.waitKey(0)

# تغيير الفريق إلى Referee
current_team = "Referee"
print("اختر لون الحكم (اضغط عليه)، ثم اضغط Enter")
cv2.imshow("Pick Players", frame)
cv2.waitKey(0)

# حفظ الألوان المحددة في ملف JSON
with open("colors.json", "w") as f:
    json.dump(selected_colors, f)

cv2.destroyAllWindows()
cap.release() 