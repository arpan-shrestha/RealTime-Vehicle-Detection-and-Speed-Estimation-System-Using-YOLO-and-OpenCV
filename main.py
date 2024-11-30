import math
import cv2
from ultralytics import YOLO
import cvzone

class Tracker:
    def __init__(self):
        self.center_points = {}  
        self.id_count = 0  
        self.last_positions = {}  

    def update(self, objects_rect):
        objects_bbs_ids = []

        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2 
            cy = (y + y + h) // 2  

            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 35:
                    self.center_points[id] = (cx, cy)
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True
                    break

            if not same_object_detected:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        self.center_points = new_center_points.copy()
        return objects_bbs_ids

def calculate_speed(last_position, current_position, fps, pixel_to_meter_ratio):
    distance_pixels = math.hypot(current_position[0] - last_position[0], current_position[1] - last_position[1])

    distance_meters = distance_pixels * pixel_to_meter_ratio

    speed_mps = distance_meters * fps  

    speed_kmph = speed_mps * 3.6
    return speed_kmph

model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture('qw.mp4')
# cap = cv2.VideoCapture(0)

tracker = Tracker()
pixel_to_meter_ratio = 0.05  
fps = int(cap.get(cv2.CAP_PROP_FPS))  

car_count = 0
cy1 = 180  # Crossing line for vehicle detection (adjust for your video)
offset = 8  # Margin for detecting vehicles crossing the line

while True:
    ret, frame = cap.read()  
    if not ret:
        break  

    frame = cv2.resize(frame, (1020, 500))  

    results = model.predict(frame)
    detections = results[0].boxes.data

    cars = []

    for row in detections:
        x1, y1, x2, y2 = map(int, row[:4])
        c = row[5]
        if c == 2:  
            cars.append([x1, y1, x2, y2])

    car_boxes = tracker.update(cars)

    cv2.line(frame, (1, cy1), (1018, cy1), (0, 255, 0), 2)

    for bbox in car_boxes:
        x1, y1, x2, y2, vehicle_id = bbox
        cx = (x1 + x2) // 2  
        cy = (y1 + y2) // 2  

        if (cy > cy1 - offset) and (cy < cy1 + offset):
            car_count += 1

        if vehicle_id in tracker.last_positions:
            last_position = tracker.last_positions[vehicle_id]
            current_position = (cx, cy)
            speed = calculate_speed(last_position, current_position, fps, pixel_to_meter_ratio)
            cv2.putText(frame, f"Speed: {speed:.2f} km/h", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        tracker.last_positions[vehicle_id] = (cx, cy)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cvzone.putTextRect(frame, f"ID: {vehicle_id}", (x1, y1), 1, 1)

    cv2.imshow("Vehicle Detection", frame)

    # Exit if 'Esc' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(f'Total car count: {car_count}')

cap.release()
cv2.destroyAllWindows()
