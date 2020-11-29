import os
import cv2
import numpy as np

base_dir = os.path.dirname(__file__)
prototxt_path = os.path.join(base_dir + '/model_data/deploy.prototxt')
caffemodel_path = os.path.join(base_dir + '/model_data/weights.caffemodel')

model = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)


def body_coordinates(img, start_point, end_point):
        (height, width) = img.shape[:2]
        under_head_size = height - end_point[1]
        
        head_width = end_point[0] - start_point[0]
        middle_point = ((start_point[0] + end_point[0]) // 2, end_point[1])

        start_body_point = [int(middle_point[0] - head_width * 2), middle_point[1]]
        end_body_point = [int(middle_point[0] + head_width * 2), height]

        start_body_point[0] = max(0, start_body_point[0])
        end_body_point[0] = min(width, end_body_point[0])
        end_body_point[1] = min(height, end_body_point[1])

        return start_body_point, end_body_point


def humen_boxes(img, face_boxes):
        boxes = []
        for face in face_boxes:
                body_box = body_coordinates(img, face[0], face[1])
                
                box_start = (body_box[0][0], face[0][1])
                box_end = body_box[1]
                boxes.append((box_start, box_end))

        return boxes


def find_people_coords(image):
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

        model.setInput(blob)
        detections = model.forward()

        # Create frame around face
        face_boxes = []
        for i in range(0, detections.shape[2]):
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                confidence = detections[0, 0, i, 2]

                if (confidence > 0.5):
                        face_boxes.append(((startX, startY), (endX, endY)))

        boxes = humen_boxes(image, face_boxes)
        
        return boxes
