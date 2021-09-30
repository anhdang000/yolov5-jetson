import torch
import cv2
import PIL
import numpy as np
import os
import os.path as osp
import json
import argparse
from datetime import datetime
import logging
import warnings

# Warning 
warnings.filterwarnings("ignore")

def parser():
    parser = argparse.ArgumentParser(description="YOLOv5 Object Detection")
    parser.add_argument("--rtsp", type=str, default=0,
                        help="rtsp source. If empty, throw error")
    parser.add_argument("--input", type=str, default=0,
                        help="video source. If empty, uses webcam 0 stream")
    parser.add_argument("--weights", default="phone.pt",
                        help="YOLOv5 weights path")
    parser.add_argument("--crop-regions", action='store_true',
                        help="YOLOv5 weights path")
    parser.add_argument("--store-preds", type=str, 
                        help="Folder to save phone usage frame")
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
        
def open_cam_rtsp(uri, latency=200):
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

def draw_boxes_from_crops(phone_results, candidate_results, frame):
    color = (80, 127, 255)
    text_color = (255, 255, 255)

    for result in range(len(phone_results)):
        for i in range(len(result)):
            xmin = int(result.loc[i]['xmin'])
            ymin = int(result.loc[i]['ymin'])
            xmax = int(result.loc[i]['xmax'])
            ymax = int(result.loc[i]['ymax'])

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

def draw_boxes(results, frame):
    color = (80, 127, 255)
    text_color = (255, 255, 255)

    if len(results) == 0:
        return frame

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
    vehicles = ['person', 'car', 'bus', 'train', 'truck']
    count = 0
    while True:
        ret, frame = cap.read()
        if ret:
            count += 1
            logging.info(f'Processing frame {count}')
            if args.crop_regions:
                candidate_results = vehicles_model(frame[:, :, ::-1]).pandas().xyxy[0]
                candidates = []
                if len(candidate_results) == 0:
                    continue
                for _, result in candidate_results.iterrows():
                    if result['name'] in vehicles:
                        result['xmin'] = int(result['xmin'])
                        result['ymin'] = int(result['ymin'])
                        result['xmax'] = int(result['xmax'])
                        result['ymax'] = int(result['ymax'])

                        crop_image = frame[result['ymin']:result['ymax'], result['xmin']:result['xmax'], ::-1]
                        candidates.append(crop_image)

                phone_results = main_model(candidates, size=640).pandas().xyxy
                frame = draw_boxes_from_crops(phone_results, candidate_results, frame)
                if args.show:
                    cv2.imshow('Frame', frame[:, :, ::-1])
                if args.store_preds and len(results) > 0:
                    file_id = '_'.join(str(datetime.now()).split())
                    cv2.imwrite(osp.join(args.store_preds, file_id + '.jpg'), frame[:, :, ::-1])
            else:
                results = main_model(frame[:, :, ::-1], size=640).pandas().xyxy[0]
                logging.info(f'Number of detected boxes: {len(results)}')
                frame = draw_boxes(results, frame)
                if args.store_preds and len(results) > 0:
                    file_id = '_'.join(str(datetime.now()).split())
                    file_path = osp.join(args.store_preds, file_id + '.jpg')
                    cv2.imwrite(file_path, frame)
                    logging.info(f'Saved prediction at: {file_path}')
        else:
            break


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format = (
            '%(levelname)s:\t'
            '%(filename)s:'
            '%(funcName)s():'
            '%(lineno)d\t'
            '%(message)s'
            )
        )

    args = parser()
    check_args_errors(args)
    if args.rtsp != 0:
        input_path = str2int(args.rtsp)
        cap = open_cam_rtsp(input_path)
    else:
        input_path = str2int(args.input)
        cap = open_cam_file(input_path)
    if args.store_preds:
        if not osp.isdir(args.store_preds):
            os.mkdir(args.store_preds)
    logging.info('Initializing models ...')
    vehicles_model, main_model = init_models(args)
    logging.info('Start inference')
    inference(vehicles_model, main_model, cap, args)