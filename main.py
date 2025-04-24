import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import cv2
import json
import numpy as np
from PIL import Image, ImageTk  # Import the Pillow library
from ultralytics import YOLO

# Initialize the main window
window = tk.Tk()
window.title("فيصل Ai")
window.geometry("600x400")  # Size of the window
window.config(bg="#2C3E50")  # Background color (fallback in case image loading fails)

# Load background image using PIL
background_image = Image.open("background.jpg")  # Replace with your image path
background_image = background_image.resize((600, 400), Image.Resampling.LANCZOS)  # Resize image to fit the window

# Convert the image for Tkinter
bg_photo = ImageTk.PhotoImage(background_image)

# Create a label for the background and place it at the bottom of the window
background_label = tk.Label(window, image=bg_photo)
background_label.place(relwidth=1, relheight=1)  # This makes it fill the entire window


# Header Label (Centered)
header_label = tk.Label(window, text="Faisal AI", font=("Arial", 24, "bold"), fg="#3498DB", bg="#2C3E50")
header_label.pack(pady=20)

# Instructions Label (Centered)
instructions_label = tk.Label(window, text="Select a video and pick the team colors for detection", font=("Arial", 14), fg="white", bg="#2C3E50")
instructions_label.pack(pady=10)

# Frame for buttons
button_frame = tk.Frame(window, bg="#2C3E50")
button_frame.pack(pady=20)

# Function to calculate dominant color
def get_dominant_color(crop, k=2):
    """Get the dominant color in the selected region of the video."""
    crop = cv2.resize(crop, (50, 50))  # Resize the crop for better processing
    data = crop.reshape((-1, 3)).astype("float32")
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    dominant = centers[np.argmax(np.bincount(labels.flatten()))]
    return tuple(map(int, dominant))

# Color selection variables
selected_colors = {"Team1": [], "Team2": [], "Referee": []}
current_team = "Team1"
frame = None  # Global frame variable to be used in callback

# Function to handle mouse clicks and pick color
def mouse_callback(event, x, y, flags, param):
    global current_team
    if event == cv2.EVENT_LBUTTONDOWN:
        crop = frame[y-30:y+30, x-20:x+20]  # Select the region around the clicked point
        dominant_color = get_dominant_color(crop)
        selected_colors[current_team].append(dominant_color)
        print(f"{current_team} color selected: {dominant_color}")

def start_color_picker(video_path):
    """Start the color picker for selecting team and referee colors."""
    global frame
    cap = cv2.VideoCapture(video_path)  # Open the video file
    ret, frame = cap.read()

    if not ret:
        print("Failed to read the video frame.")
        return

    # Define the mouse callback function
    cv2.namedWindow("Pick Players")
    cv2.setMouseCallback("Pick Players", mouse_callback)

    # Pick colors for Team1
    print("Pick 2 players from Team1 (click on them, then press Enter)")
    cv2.imshow("Pick Players", frame)
    cv2.waitKey(0)

    # Change team to Team2
    global current_team
    current_team = "Team2"
    print("Pick 2 players from Team2, then press Enter")
    cv2.imshow("Pick Players", frame)
    cv2.waitKey(0)

    # Change team to Referee
    current_team = "Referee"
    print("Pick the Referee color (click on it, then press Enter)")
    cv2.imshow("Pick Players", frame)
    cv2.waitKey(0)

    # Save the selected colors
    cv2.destroyAllWindows()
    cap.release()

    # Save colors to JSON file
    with open("colors.json", "w") as json_file:
        json.dump(selected_colors, json_file)
    print("Colors saved to colors.json")

    messagebox.showinfo("Success", "Colors saved successfully!")

def trigger_color_picker():
    video_path = filedialog.askopenfilename(title="Select Video", filetypes=(("MP4 Files", ".mp4"), ("All Files", ".*")))
    if not video_path:
        return
    start_color_picker(video_path)

def load_and_process_video():
    if not selected_colors["Team1"] or not selected_colors["Team2"] or not selected_colors["Referee"]:
        messagebox.showwarning("Warning", "Please select colors for all teams before processing the video.")
        return

    video_path = filedialog.askopenfilename(title="Select Video", filetypes=(("MP4 Files", ".mp4"), ("All Files", ".*")))
    if not video_path:
        return

    print(f"Video selected: {video_path}")
    model = YOLO("yolov8n.pt")  # Load YOLO model
    cap = cv2.VideoCapture(video_path)  # Open the video file

    # Get video dimensions for output
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)  # Get the frame rate (FPS) of the video

    # Create VideoWriter to save the output video
    output_path = 'output_video.mp4'  # Path to save the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4 video
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))  # Initialize VideoWriter

    while cap.isOpened():
        ret, frame = cap.read()  # Read each frame from the video
        if not ret:
            break

        results = model(frame)  # Perform object detection on the frame

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                label = result.names[cls_id]
                conf = float(box.conf[0])  # Confidence score of detection

                # Skip detections below the threshold
                if conf < 0.4:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                # Classify persons based on color
                if label == "person":
                    crop = frame[y1:y2, x1:x2]
                    shirt_crop = crop[int((y2 - y1) * 0.2):int((y2 - y1) * 0.6), int((x2 - x1) * 0.25):int((x2 - x1) * 0.75)]
                    dominant_color = cv2.mean(shirt_crop)[:3]
                    color = tuple(map(int, dominant_color))
                    team = classify_team(color)

                    color_map = {"Team1": (0, 255, 0), "Team2": (255, 0, 0), "Referee": (0, 255, 255)}
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color_map[team], 2)
                    cv2.putText(frame, team, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_map[team], 2)

                elif label == "sports ball":
                    cv2.circle(frame, (cx, cy), 10, (0, 0, 255), -1)  # Red for ball
                    cv2.putText(frame, "Ball", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Write the processed frame to the output video
        out.write(frame)

        # Display the processed frame (comment this if running in environments without GUI support)
        cv2.imshow("Match Analysis", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()  # Release the video writer object
    cv2.destroyAllWindows()

    print(f"Video saved successfully to: {output_path}")
    messagebox.showinfo("Success", f"Video saved successfully to: {output_path}")

# Function to classify team based on the color
def classify_team(color):
    dist1 = min([color_distance(color, ref) for ref in selected_colors["Team1"]])
    dist2 = min([color_distance(color, ref) for ref in selected_colors["Team2"]])
    dist_ref = min([color_distance(color, ref) for ref in selected_colors["Referee"]])

    if dist_ref < dist1 and dist_ref < dist2:
        return "Referee"
    elif dist1 < dist2:
        return "Team1"
    else:
        return "Team2"

# Function to calculate the color distance
def color_distance(c1, c2):
    return np.linalg.norm(np.array(c1) - np.array(c2))

# Add buttons to trigger actions (color picker and video processing)
button_color_picker = tk.Button(button_frame, text="Select and Save Colors", font=("Arial", 14), bg="#2ECC71", fg="white", command=trigger_color_picker)
button_color_picker.grid(row=0, column=0, padx=10, pady=10)

button_process_video = tk.Button(button_frame, text="Start Video Processing", font=("Arial", 14), bg="#3498DB", fg="white", command=load_and_process_video)
button_process_video.grid(row=1, column=0, padx=10, pady=10)

# Run the interface
window.mainloop()