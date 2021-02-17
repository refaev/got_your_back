"""
├── yolo
│   ├── labels.txt
│   ├── yolov4-tiny.cfg
│   ├── yolov4-tiny.weights
├── people.jpg
├── people_out.jpg
├── street.jpg
├── street_out.jpg
├── video.mp4
├── video_out.avi
├── yolo_image.py
└── yolo_video.py
 if program cant find yolo folder in main folder it will crash."""
 # example usage: python yolo_image.py -i street.jpg -o output.jpg
import argparse
import time
import glob

import cv2
import numpy as np

#import datetime
#from time import gmtime, strftime


def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=60,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )






parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str, default="",
	help="path to (optional) input image file")
parser.add_argument("-o", "--output", type=str, default="",
	help="path to (optional) output image file. Write only the name, without extension.")
parser.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
parser.add_argument("-t", "--threshold", type=float, default=0.4,
	help="threshold for non maxima supression")

args = vars(parser.parse_args())

CONFIDENCE_THRESHOLD = args["confidence"]
NMS_THRESHOLD = args["threshold"]
impath = args["input"]

weights = glob.glob("yolo/*.weights")[0]
labels = glob.glob("yolo/*.txt")[0]
cfg = glob.glob("yolo/*.cfg")[0]

print("You are now using {} weights ,{} configs and {} labels.".format(weights, cfg, labels))

lbls = list()
with open(labels, "r") as f:
	lbls = [c.strip() for c in f.readlines()]

COLORS = np.random.randint(0, 255, size=(len(lbls), 3), dtype="uint8")

net = cv2.dnn.readNetFromDarknet(cfg, weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

layer = net.getLayerNames()
layer = [layer[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def detect(imgpath, nn):
#def detect(image, nn):
    if 1:
        image = cv2.imread(imgpath)
        (H, W) = image.shape[:2]

        blob = cv2.dnn.blobFromImage(image, 1/255, (416, 416), swapRB=True, crop=False)
        nn.setInput(blob)

        start_time = time.time()
        layer_outs = nn.forward(layer)
        end_time = time.time()

        boxes = list()
        confidences = list()
        class_ids = list()

        for output in layer_outs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > CONFIDENCE_THRESHOLD:
                    if class_id==1 orclass_id==2 or class_id==3 or class_id==5:		# Shlomo filtered to track just these : 1-bicycle 2-car, 3-motorbike 5-bus
											# need to check if "coco names" starts from 0 or from 1.

											# https://github.com/pjreddie/darknet/blob/master/data/coco.names
                        box = detection[0:4] * np.array([W, H, W, H])
                        (center_x, center_y, width, height) = box.astype("int")

                        x = int(center_x - (width / 2))
                        y = int(center_y - (height / 2))

                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

        

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
        if len(idxs) > 0:
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                color = [int(c) for c in COLORS[class_ids[i]]]
                cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
                text = "{}: {:.4f}".format(lbls[class_ids[i]], confidences[i])
                cv2.putText(image, text, (x, y -5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                label = "Inference Time: {:.2f} ms".format(end_time - start_time)
                cv2.putText(image, label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)



        path_str = "./images_results"+imgpath.split('question')[-1]
        print ("path_str =", path_str)
        cv2.imwrite(path_str,image)

        ''' 
	cv2.imshow("image", image)
	if args["output"] != "":
		cv2.imwrite(args["output"], image)
	cv2.waitKey(0)
        '''
    return(boxes,confidences,class_ids)    		# shlolmo Added


def show_camera():

    img_counter = 0

    # To flip the image, modify the flip_method parameter (0 and 2 are the most common)
    print(gstreamer_pipeline(flip_method=0))
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    if cap.isOpened():
        window_handle = cv2.namedWindow("CSI Camera", cv2.WINDOW_AUTOSIZE)
        # Window
        while cv2.getWindowProperty("CSI Camera", 0) >= 0:
            ret_val, img = cap.read()
            path_str = "./images_in_question/"+str(img_counter)+".jpg" 
            print ("path_str =", path_str)
            cv2.imwrite(path_str,img)
            boxes,confidences,class_ids = detect(path_str,net)   # shlomo changed   - Here detect single image, and get relevant results


            # Here add tracking ?				 # shlomo

 
            img_counter += 1

        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Unable to open camera")


if __name__ == "__main__":
    show_camera()








