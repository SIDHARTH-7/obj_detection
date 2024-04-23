import cv2
import numpy as np
import tensorflow as tf
import time
import threading

class ObjectDetection:
    def __init__(self):
        self.person_detected = False
        self.stopped = False

    def detect_person(self):
        # Load label dictionary
        label_dict = self.load_label_dict('models/ssd_mobilenet/labels.txt')

        # Load TensorFlow model
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile('models/ssd_mobilenet/frozen_inference_graph.pb', 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            sess = tf.compat.v1.Session(graph=detection_graph)

        # Open PC's camera (webcam)
        camera = cv2.VideoCapture(0)

        while not self.stopped:
            # Capture frame-by-frame
            ret, frame = camera.read()
            if not ret:
                print("Error: failed to capture frame")
                break

            # Preprocess frame
            image_np_expanded = np.expand_dims(frame, axis=0)
            frame = cv2.resize(frame, (640, 480))
            # Perform inference
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})

            # Check for person detection
            for i in range(int(num_detections[0])):
                score = scores[0][i]
                if score > 0.5:
                    label_id = int(classes[0][i])
                    label_str = label_dict.get(label_id, 'unknown')
                    if label_str == 'person':
                        self.person_detected = True
                        break
            else:
                self.person_detected = False

        # Release the camera
        camera.release()
        cv2.destroyAllWindows()

    def stop_detection(self):
        self.stopped = True

    def load_label_dict(self, label_file):
        label_dict = {}
        with open(label_file, 'r') as f:
            for line in f:
                label_id, label_name = line.strip().split(':')
                label_dict[int(label_id)] = label_name.strip()
        return label_dict

class Lift:
    def __init__(self, num_floors, object_detection):
        self.num_floors = num_floors
        self.current_floor = 1
        self.direction = 0  # 1 for up, -1 for down, 0 for stop
        self.requests = []
        self.object_detection = object_detection
        self.stopped_due_to_no_person = False

    def move(self, floor):
        while floor != self.current_floor:
            if self.direction == 1:
                self.current_floor += 1
            elif self.direction == -1:
                self.current_floor -= 1
            print(f"Moving to floor {self.current_floor}")
            time.sleep(3)  # 3 seconds delay for moving between floors

            # Check for person detection
            if not self.object_detection.person_detected:
                print("No person detected. Stopping the lift.")
                self.stop()
                self.stopped_due_to_no_person = True
                break

    def stop(self):
        print(f"Stopping at floor {self.current_floor}")
        self.direction = 0

    def process_requests(self):
        while self.requests:
            request = self.requests.pop(0)
            floor, direction = request
            if floor == self.current_floor:
                self.stop()
            elif floor > self.current_floor:
                self.direction = 1
                self.move(floor)
                self.stop()
            else:
                self.direction = -1
                self.move(floor)
                self.stop()

    def add_request(self, floor, direction):
        self.requests.append((floor, direction))
        if direction == 1:
            print(f"Request to go UP from floor {floor}")
        elif direction == -1:
            print(f"Request to go DOWN from floor {floor}")

    def reset_lift(self):
        self.requests = []
        self.direction = 0
        self.stopped_due_to_no_person = False

def main():
    num_floors = 10
    object_detection = ObjectDetection()

    # Start person detection thread
    detection_thread = threading.Thread(target=object_detection.detect_person)
    detection_thread.start()

    print("started detection thread")
    lift = Lift(num_floors, object_detection)

    lift_thread = threading.Thread(target=lift.process_requests)
    lift_thread.start()

    while True:
        user_input = input("Enter floor number to go (1-10) or 'q' to quit: ")
        if user_input.lower() == 'q':
            print("Exiting...")
            object_detection.stop_detection()
            break
        try:
            floor = int(user_input)
            if floor < 1 or floor > num_floors:
                print("Invalid floor number. Please enter a number between 1 and 10.")
                continue
            direction = 1 if floor > lift.current_floor else -1
            lift.add_request(floor, direction)
            lift.process_requests()
        except ValueError:
            print("Invalid input. Please enter a valid floor number or 'q' to quit.")

        if lift.stopped_due_to_no_person:
            lift.reset_lift()

if __name__ == "__main__":
    main()
