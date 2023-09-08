# -*- coding: utf-8 -*-

"""
    YOLOv5-v5.0 TensorRT Detector
"""

import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer, as_array
import cv2

from labels import label_dict


class YoloDetector:

    OUTPUT_SIZE = 6001  # we assume the yololayer outputs no more than MAX_OUTPUT_BBOX_COUNT boxes that conf >= 0.1;
    INPUT_HEIGHT = 640
    INPUT_WIDTH = 640
    NMS_THRESH = 0.5
    CONF_THRESH = 0.4

    def __init__(self, trt_file="./resources/model.plan", gpu_id=0):
        self.yolo_infer_lib = ctypes.cdll.LoadLibrary("./lib/libyolo_infer.so")
        # self.cpp_yolo_detector = self.yolo_infer_lib.YoloDetecter_new()
        self.yolo_infer_lib.YoloDetecter_new.argtypes = [ctypes.c_char_p, ctypes.c_int]
        self.cpp_yolo_detector = self.yolo_infer_lib.YoloDetecter_new(trt_file.encode('utf-8'), gpu_id)


    def release(self):
        self.yolo_infer_lib.destroy(self.cpp_yolo_detector)


    def infer(self, image):
        # build input data
        height, width = image.shape[:2]
        in_data = np.ascontiguousarray(image.copy().reshape(-1))

        # inference
        self.yolo_infer_lib.inference_one.argtypes = [ctypes.c_int, ndpointer(ctypes.c_ubyte), ctypes.c_int, ctypes.c_int]
        self.yolo_infer_lib.inference_one.restype = ndpointer(ctypes.c_float, shape=(self.OUTPUT_SIZE,))
        out_data = self.yolo_infer_lib.inference_one(self.cpp_yolo_detector, in_data, height, width)
        out_data = as_array(out_data).copy().reshape(-1)

        # postprocess
        return self._postprocess(image, out_data)

    def _postprocess(self, image, model_output):
        detected_objects = []  # [[center_x, center_y, w, h, conf, class_id], [...], [...]]
        num_bboxes = int(model_output[0])
        # NMS
        if num_bboxes:
            bboxes = model_output[1: (num_bboxes * 6 + 1)].reshape(-1, 6)
            # bboxes shape : (num_bboxes, 6)
            # 6 dims are : center_x, center_y, w, h, conf, class_id, see in yololayer.h struct Detection
            labels = set(bboxes[:, 5].astype(int))

            for label in labels:
                selected_bboxes = bboxes[np.where((bboxes[:, 5] == label) & (bboxes[:, 4] >= self.CONF_THRESH))]
                selected_bboxes_keep = selected_bboxes[self._nms(selected_bboxes[:, :4], selected_bboxes[:, 4], self.NMS_THRESH)]
                detected_objects += selected_bboxes_keep.tolist()

        # yolo input image bbox to original image bbox
        final_res = []  # [[left, top, right, bottom, conf, class_id], [...], [...]]
        for obj in detected_objects:
            left, top, right, bottom = self._get_rect(image, obj[:4])
            conf = obj[4]
            class_id = int(obj[5])
            final_res.append([left, top, right, bottom, conf, class_id])

        return final_res


    @staticmethod
    def _nms(boxes, box_confidences, nms_threshold=0.5):
        x_coord = boxes[:, 0]
        y_coord = boxes[:, 1]
        width = boxes[:, 2]
        height = boxes[:, 3]

        areas = width * height
        ordered = box_confidences.argsort()[::-1]

        keep = list()
        while ordered.size > 0:
            i = ordered[0]
            keep.append(i)
            xx1 = np.maximum(x_coord[i], x_coord[ordered[1:]])
            yy1 = np.maximum(y_coord[i], y_coord[ordered[1:]])
            xx2 = np.minimum(x_coord[i] + width[i], x_coord[ordered[1:]] + width[ordered[1:]])
            yy2 = np.minimum(y_coord[i] + height[i], y_coord[ordered[1:]] + height[ordered[1:]])

            width1 = np.maximum(0.0, xx2 - xx1 + 1)
            height1 = np.maximum(0.0, yy2 - yy1 + 1)
            intersection = width1 * height1
            union = (areas[i] + areas[ordered[1:]] - intersection)

            iou = intersection / union

            indexes = np.where(iou <= nms_threshold)[0]
            ordered = ordered[indexes + 1]
        keep = np.array(keep).astype(int)
        return keep

    def _get_rect(self, image, bbox):
        img_height, img_width = image.shape[:2]
        r_w = self.INPUT_WIDTH / img_width
        r_h = self.INPUT_HEIGHT / img_height
        if r_h > r_w:
            l = bbox[0] - bbox[2] / 2
            r = bbox[0] + bbox[2] / 2
            t = bbox[1] - bbox[3] / 2 - (self.INPUT_HEIGHT - r_w * img_height) / 2
            b = bbox[1] + bbox[3] / 2 - (self.INPUT_HEIGHT - r_w * img_height) / 2
            l /= r_w
            r /= r_w
            t /= r_w
            b /= r_w
        else:
            l = bbox[0] - bbox[2] / 2 - (self.INPUT_WIDTH - r_h * img_width) / 2
            r = bbox[0] + bbox[2] / 2 - (self.INPUT_WIDTH - r_h * img_width) / 2
            t = bbox[1] - bbox[3] / 2
            b = bbox[1] + bbox[3] / 2
            l /= r_h
            r /= r_h
            t /= r_h
            b /= r_h
        return int(l), int(t), int(r), int(b)

    @staticmethod
    def draw_image(detected_list, image, line_color=(255, 0, 255), label_color=(255, 255, 255), line_thickness=2):
        for x1, y1, x2, y2, conf, class_id in detected_list:
            c1, c2 = (x1, y1), (x2, y2)
            cv2.rectangle(image, c1, c2, line_color, thickness=line_thickness, lineType=cv2.LINE_AA)

            label = f"{label_dict[class_id]}-{conf:.2f}"
            t_size = cv2.getTextSize(label, 0, fontScale=line_thickness / 3, thickness=line_thickness)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(image, c1, c2, line_color, -1, cv2.LINE_AA)  # filled
            cv2.putText(image, label, (c1[0], c1[1] - 2), 0, line_thickness / 3, label_color, thickness=line_thickness, lineType=cv2.LINE_AA)
