from openvino.inference_engine import IENetwork, IECore
import numpy as np
import cv2
import time
import os
import requests
import json
import base64
from utils import *

MODEL_XML = './model/FP16/pedestrian-detection-adas-0002.xml'
MODEL_BIN = './model/FP16/pedestrian-detection-adas-0002.bin'
SAVE_PATH = "./temp/"

THRESHOLD = 0.6

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)


class Detector():
    def __init__(self):
        print("正在载入检测器,请稍后......")
        start_time = time.time()
        self.N, self.C, self.H, self.W = 0, 0, 0, 0
        self.exec_net, self.input_blob, self.out_blob = self.load_model(MODEL_XML, MODEL_BIN)
        print("检测器载入成功，耗时{}秒".format(time.time() - start_time))

    def load_model(self, model_xml, model_bin):
        ie = IECore()
        net = ie.read_network(model=model_xml, weights=model_bin)
        exec_net = ie.load_network(network=net, num_requests=1, device_name="MYRIAD")
        input_blob = next(iter(net.input_info))
        self.N, self.C, self.H, self.W = net.input_info[input_blob].input_data.shape
        out_blob = next(iter(net.outputs))
        return exec_net, input_blob, out_blob

    def read_image(self, image):
        if isinstance(image, str):
            image = cv2.imread(image)
        elif not isinstance(image, np.ndarray):
            raise Exception("read_image function only support str and cv2 image input")
        in_frame = cv2.resize(image, (self.W, self.H))
        in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        in_frame = in_frame.reshape((self.N, self.C, self.H, self.W))
        initial_w = image.shape[1]
        initial_h = image.shape[0]
        return initial_w, initial_h, frame, in_frame

    def process_out(self, res, image, initial_w, initial_h):
        # [1,1,200,7]=>image_id, label, conf, x_min, y_min, x_max, y_max]
        result = []
        result_names = []
        person_id = 0
        for obj in res[0][0]:
            if obj[2] > THRESHOLD:
                xmin = int(obj[3] * initial_w)
                ymin = int(obj[4] * initial_h)
                xmax = int(obj[5] * initial_w)
                ymax = int(obj[6] * initial_h)
                h, w = ymax - ymin, xmax - xmin
                if h > 0 and w > 0 and 0.5 < h / w < 5 and xmin > 0 and ymin > 0 and xmax < initial_w and ymax < initial_h:
                    crop_img = image[ymin:ymax, xmin:xmax]
                    result.append(crop_img)
                    name = SAVE_PATH + self.get_time_stamp() + "-p{}.jpg".format(person_id)
                    result_names.append(name)
                    person_id += 1
        return result, result_names

    def detect(self, img):
        start_time = time.time()
        initial_w, initial_h, orig_frame, in_frame = self.read_image(img)
        res = self.exec_net.infer(inputs={self.input_blob: in_frame})[self.out_blob]
        result, result_names = self.process_out(res, orig_frame, initial_w, initial_h)
        print("检测耗时:{}秒".format(time.time() - start_time))
        return result, result_names

    def get_time_stamp(self):
        ct = time.time()
        local_time = time.localtime(ct)
        data_head = time.strftime("%Y-%m-%d--%H-%M-%S", local_time)
        data_secs = (ct - int(ct)) * 1000
        time_stamp = "%s.%03d" % (data_head, data_secs)
        return time_stamp


if __name__ == '__main__':
    detector = Detector()
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            detected_imgs, names = detector.detect(frame)
            for i in range(len(names)):
                cv2.imwrite(names[i], detected_imgs[i])
                send(names[i])
