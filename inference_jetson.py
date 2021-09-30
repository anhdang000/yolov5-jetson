import torch
import cv2
import PIL
import numpy as np
import os
import os.path as osp
import json
import argparse

def parser():
    parser = argparse.ArgumentParser(description="YOLOv5 Object Detection")
    parser.add_argument("--rtsp", type=str, default=0,
                        help="rtsp source. If empty, throw error")
    parser.add_argument("--input", type=str, default=0,
                        help="video source. If empty, uses webcam 0 stream")
    parser.add_argument("--output", type=str, default=None,
                        help="inference video name. Not saved if empty")
    parser.add_argument("--weights", default="phone.pt",
                        help="YOLOv5 weights path")
    parser.add_argument("--show", action='store_true',
                        help="Windown inference display")
    return parser.parse_args()

def check_args_errors(args):
    if not os.path.exists(args.weights) or not args.weights.endswith('.pt'):
        raise(ValueError("Invalid weight path {}".format(os.path.abspath(args.weights))))
    if not (args.rtsp or args.input):
        raise(ValueError("No source found"))

def str2int(video_path):
    """
    argparse returns and string althout webcam uses int (0, 1 ...)
    Cast to int if needed
    """
    try:
        return int(video_path)
    except ValueError:
        return video_path
        
def open_cam_rtsp(uri, latency):
	gst_str = ("rtspsrc location={} latency={}  ! queue ! rtph264depay ! h264parse ! nvv4l2decoder enable-max-performance=1 ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR  ! appsink max-buffers=1 drop=True").format(uri, latency)
	print(gst_str)
	return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

def open_cam_file(uri):
	gst_str = ("filesrc location={} ! qtdemux ! queue ! h264parse ! omxh264dec ! nvvidconv ! video/x-raw,format=BGRx ! queue ! videoconvert ! queue ! video/x-raw, format=BGR ! appsink").format(uri)
	print(gst_str)
	return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

def init_models(args):
    vehicles_model = torch.hub.load('.', 'custom', path='yolov5s.pt', source='local')  # or yolov5m, yolov5l, yolov5x, custom
    main_model = torch.hub.load('.', 'custom', path=args.weights, source='local')
    return vehicles_model, main_model

def draw_boxes(results, frame):
    color = (80, 127, 255)
    text_color = (255, 255, 255)

    for i in range(len(results)):
        xmin = int(results.loc[i]['xmin'])
        ymin = int(results.loc[i]['ymin'])
        xmax = int(results.loc[i]['xmax'])
        ymax = int(results.loc[i]['ymax'])

        # For bounding box
        frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
        # For the text background
        # Finds space required by the text so that we can put a background with that amount of width.
        (w, h), _ = cv2.getTextSize(
                'phone', cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        # Prints the text.    
        frame = cv2.rectangle(frame, (xmin, ymin - 20), (xmin + w, ymin), color, -1)
        frame = cv2.putText(frame, 'phone', (xmin, ymin - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)

    return frame


def inference(vehicles_model, main_model, cap, args):
    vehicles = ['car', 'bus', 'train', 'truck']
    if args.output:
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
    while True:
        ret, frame = cap.read()
        if ret:
            # Make predictions
            frame = frame[:, :, ::-1]
            results = vehicles_model(frame).pandas().xyxy[0]
            candidates = []
            if len(results) == 0:
                continue
            for _, result in results.iterrows():
                if result['name'] in vehicles:
                    result['xmin'] = int(result['xmin'])
                    result['ymin'] = int(result['ymin'])
                    result['xmax'] = int(result['xmax'])
                    result['ymax'] = int(result['ymax'])

                    crop_image = frame[result['ymin']:result['ymax'], result['xmin']:result['xmax'], :]
                    candidates.append(crop_image)

            phone_results = main_model(candidates, size=640).pandas().xyxy
            frame = draw_boxes(phone_results, frame)
            if args.show:
                cv2.imshow('Frame',frame)
            if args.output:
                out.write(frame)
        else:
            break


if __name__ == "__main__":
    args = parser()
    check_args_errors(args)
    if args.rtsp != 0:
        input_path = str2int(args.rtsp)
        cap = open_cam_rtsp(input_path)
    else:
        input_path = str2int(args.input)
        cap = open_cam_file(input_path)

    vehicles_model, main_model = init_models(args)
    inference(vehicles_model, main_model, cap, args)