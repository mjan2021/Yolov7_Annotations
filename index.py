import asone
from asone import utils
from asone import ASOne
import cv2

video_path = 'data/sample_videos/test.mp4'
detector = ASOne(detector=asone.YOLOV7_PYTORCH, use_cuda=True) # Set use_cuda to False for cpu

filter_classes = ['car'] # Set to None to detect all classes

cap = cv2.VideoCapture(video_path)

file = open("./yolov8_annotations", "w+")
	
while True:
	_, frame = cap.read()
    	if not _:
        	break

    	dets, img_info = detector.detect(frame, filter_classes=filter_classes)

    	bbox_xyxy = dets[:, :4]
    	scores = dets[:, 4]
    	class_ids = dets[:, 5]

	for idx in range(0, len(bbox_xyxy)):
		objects = f"{class_ids[idx]} {bbox_xyxy[0]} {bbox_xyxy[1]} {bbox_xyxy[2]} {bbox_xyxy[3]}"
		file.write(str(objects))

    	#frame = utils.draw_boxes(frame, bbox_xyxy, class_ids=class_ids)

    	#cv2.imshow('result', frame)

    	if cv2.waitKey(25) & 0xFF == ord('q'):
        	break
