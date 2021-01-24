from openvino.inference_engine import IENetwork, IECore
import cv2
import time
import os

MODEL_XML = './model/FP16/pedestrian-detection-adas-0002.xml'
MODEL_BIN = './model/FP16/pedestrian-detection-adas-0002.bin'
N, C, H, W = 1, 3, 384, 672
THRESHOLD = 0.6
Detected_PATH = "./temp"

if not os.path.exists(Detected_PATH):
    os.makedirs(Detected_PATH)


class Detector():
    def __init__(self):
        super(Detector, self).__init__()
        print("正在载入检测器,请稍后......")
        start_time = time.time()
        self.exec_net, self.input_blob, self.out_blob = self.load_model(MODEL_XML, MODEL_BIN)
        print("检测器载入成功，耗时{}秒".format(time.time() - start_time))

    def load_model(self, model_xml, model_bin):
        ie = IECore()
        net = IENetwork(model=model_xml, weights=model_bin)
        exec_net = ie.load_network(network=net, num_requests=1, device_name="MYRIAD")
        input_blob = next(iter(net.inputs))
        out_blob = next(iter(net.outputs))
        return exec_net, input_blob, out_blob

    def read_image(self, image_path):
        frame = cv2.imread(image_path)
        in_frame = cv2.resize(frame, (W, H))
        in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        in_frame = in_frame.reshape((N, C, H, W))
        initial_w = frame.shape[1]
        initial_h = frame.shape[0]
        return initial_w, initial_h, frame, in_frame

    def read_frame(self, frame):
        in_frame = cv2.resize(frame, (W, H))
        in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        in_frame = in_frame.reshape((N, C, H, W))
        initial_w = frame.shape[1]
        initial_h = frame.shape[0]
        return initial_w, initial_h, frame, in_frame

    def process_out_and_save(self, res, image, initial_w, initial_h, camera_id, frame_id):
        # [1,1,200,7]
        # image_id, label, conf, x_min, y_min, x_max, y_max]
        result = []
        person_id = 0
        for obj in res[0][0]:
            if obj[2] > THRESHOLD:
                xmin = int(obj[3] * initial_w)
                ymin = int(obj[4] * initial_h)
                xmax = int(obj[5] * initial_w)
                ymax = int(obj[6] * initial_h)
                h, w = ymax - ymin, xmax - xmin
                if h > 0 and w > 0 and h / w > 0.5 and h / w < 5 and xmin > 0 and ymin > 0 and xmax < initial_w and ymax < initial_h:
                    crop_img = image[ymin:ymax, xmin:xmax]
                    result.append(crop_img)
                    self.save_image(crop_img, camera_id, frame_id, person_id)
                    person_id += 1
        return result

    def process_out(self, res, image, initial_w, initial_h, camera_id, frame_id):
        # [1,1,200,7]
        # image_id, label, conf, x_min, y_min, x_max, y_max]
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
                if h > 0 and w > 0 and h / w > 0.5 and h / w < 5 and xmin > 0 and ymin > 0 and xmax < initial_w and ymax < initial_h:
                    crop_img = image[ymin:ymax, xmin:xmax]
                    result.append(crop_img)
                    name = Detected_PATH + "c{}_f{}_p{}.jpg".format(camera_id, frame_id, person_id)
                    result_names.append(name)
                    person_id += 1
        return result, result_names

    def save_image(self, crop_img, camera_id, frame_id, person_id):
        save_name = "c{}_f{}_p{}.jpg".format(camera_id, frame_id, person_id)
        save_path = Detected_PATH + save_name
        cv2.imwrite(save_path, crop_img)

    def detect_and_save(self, img, camera_id, frame_id):
        initial_w, initial_h, orig_frame, in_frame = self.read_frame(img)
        res = self.exec_net.infer(inputs={self.input_blob: in_frame})[self.out_blob]
        self.process_out_and_save(res, orig_frame, initial_w, initial_h, camera_id, frame_id)

    def detect(self, img, camera_id, frame_id):
        initial_w, initial_h, orig_frame, in_frame = self.read_frame(img)
        res = self.exec_net.infer(inputs={self.input_blob: in_frame})[self.out_blob]
        result, result_names = self.process_out(res, orig_frame, initial_w, initial_h, camera_id, frame_id)
        return result, result_names


if __name__ == '__main__':
    print("加载模型")
    start_time = time.time()
    detector = Detector()
    print("加载成功,耗时：{}秒".format(time.time() - start_time))
    cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture("/home/pi/projects/openvino_test/video.avi")
    ret, frame = cap.read()
    f_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        cv2.imshow("Detection Results", frame)
        if f_id % 25 == 0:
            detector.detect_and_save(frame, 0, f_id)
        f_id += 1
        cv2.waitKey(1)
