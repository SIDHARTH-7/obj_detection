import cv2
import numpy as np
import tensorflow as tf

def load_label_dict(label_file):
    label_dict = {}
    with open(label_file, 'r') as f:
        for line in f:
            label_id, label_name = line.strip().split(':')
            label_dict[int(label_id)] = label_name.strip()
    return label_dict

def main():
    label_dict = load_label_dict('models/ssd_mobilenet/labels.txt')

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

    color = (23, 230, 210)

    while True:
        # Capture frame-by-frame
        ret, frame = camera.read()
        if not ret:
            print("Error: failed to capture frame")
            break

        # Preprocess frame
        image_np_expanded = np.expand_dims(frame, axis=0)

        # Perform inference
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        (boxes, scores, classes, num_detections) = sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})

        # Visualize detections
        for i in range(int(num_detections[0])):
            score = scores[0][i]
            if score > 0.5:
                label_id = int(classes[0][i])
                label_str = label_dict.get(label_id, 'unknown')

                h, w, _ = frame.shape
                box = boxes[0][i] * np.array([h, w, h, w])
                (top, left, bottom, right) = box.astype(int)

                cv2.rectangle(frame, (left, top), (right, bottom), color, thickness=2)
                cv2.putText(frame, f'{label_str}: {score:.2f}', (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow('press Q to exit', frame)

        key = cv2.waitKey(delay=1) & 0xff

        # if 'q' was pressed exit
        if key == ord('q'):
            break

    # Release the camera
    camera.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
