from darkflow.net.build import TFNet
import cv2


def pb_yolo():
    options = {"pbLoad": "./model/yolo.pb",
               "metaLoad": "./model/yolo.meta",
               "threshold": 0.5}
    return TFNet(options)
