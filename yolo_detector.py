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
        detected_objects = []  # [[x1, y1, x2, y2, conf, class_id], [...], [...]]
        num_bboxes = int(model_output[0])
        # NMS
        if num_bboxes:
            bboxes = model_output[1: (num_bboxes * 6 + 1)].reshape(-1, 6)
            # bboxes shape : (num_bboxes, 6)
            # 6 dims are : center_x, center_y, w, h, conf, class_id, see in yololayer.h struct Detection
            bboxes = self._xywh2xyxy(bboxes)
            labels = set(bboxes[:, 5].astype(int))

            for label in labels:
                selected_bboxes = bboxes[np.where((bboxes[:, 5] == label) & (bboxes[:, 4] >= self.CONF_THRESH))]
                selected_bboxes_keep = selected_bboxes[self._nms(selected_bboxes[:, :4], selected_bboxes[:, 4], self.NMS_THRESH)]
                detected_objects += selected_bboxes_keep.tolist()

        if detected_objects:
            detected_objects = np.array(detected_objects)
        else:
            return []

        detected_objects[:, :4] = self._scale_coords((self.INPUT_HEIGHT, self.INPUT_WIDTH), detected_objects[:, :4], image.shape[:2])

        final_res = []
        for x1, y1, x2, y2, conf, class_id in detected_objects.tolist():
            final_res.append([int(x1), int(y1), int(x2), int(y2), float(conf), int(class_id)])

        return final_res

    
    @staticmethod
    def _xywh2xyxy(bboxes):
        """
            Convert nx4 boxes from [center x, center y, w, h, conf, class_id] to [x1, y1, x2, y2, conf, class_id] 
        where xy1=top-left, xy2=bottom-right
        """
        out = bboxes.copy()
        out[:, 0] = bboxes[:, 0] - bboxes[:, 2] / 2  # top left x
        out[:, 1] = bboxes[:, 1] - bboxes[:, 3] / 2  # top left y
        out[:, 2] = bboxes[:, 0] + bboxes[:, 2] / 2  # bottom right x
        out[:, 3] = bboxes[:, 1] + bboxes[:, 3] / 2  # bottom right y
        return out


    def _scale_coords(self, img1_shape, coords, img0_shape):
        # Rescale coords (xyxy) from img1_shape to img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding

        coords[:, [0, 2]] -= pad[0]  # x padding
        coords[:, [1, 3]] -= pad[1]  # y padding
        coords[:, :4] /= gain
        self._clip_coords(coords, img0_shape)
        return coords

    @staticmethod
    def _clip_coords(boxes, img_shape):
        # Clip bounding xyxy bounding boxes to image shape (height, width)
        boxes[:, 0] = boxes[:, 0].clip(0, img_shape[1])  # x1
        boxes[:, 1] = boxes[:, 1].clip(0, img_shape[0])  # y1
        boxes[:, 2] = boxes[:, 2].clip(0, img_shape[1])  # x2
        boxes[:, 3] = boxes[:, 3].clip(0, img_shape[0])  # y2


    @staticmethod
    def _nms(bboxes, scores, threshold=0.5):
        x1 = bboxes[:, 0]
        y1 = bboxes[:, 1]
        x2 = bboxes[:, 2]
        y2 = bboxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, (xx2 - xx1))
            h = np.maximum(0.0, (yy2 - yy1))
            inter = w * h

            iou = inter / (areas[i] + areas[order[1:]] - inter)

            indexes = np.where(iou <= threshold)[0]
            order = order[indexes + 1]
        keep = np.array(keep).astype(int)
        return keep

    @staticmethod
    def draw_image(detected_list, image, line_color=(255, 0, 255), label_color=(255, 255, 255), line_thickness=2):
        for x1, y1, x2, y2, conf, class_id in detected_list:
            c1, c2 = (x1, y1), (x2, y2)
            cv2.rectangle(image, c1, c2, line_color, thickness=line_thickness, lineType=cv2.LINE_AA)

            # label = f"{label_dict[class_id]}-{conf:.2f}"
            label = label_dict[class_id]
            t_size = cv2.getTextSize(label, 0, fontScale=line_thickness / 3, thickness=line_thickness)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(image, c1, c2, line_color, -1, cv2.LINE_AA)  # filled
            cv2.putText(image, label, (c1[0], c1[1] - 2), 0, line_thickness / 3, label_color, thickness=line_thickness, lineType=cv2.LINE_AA)
