import asone
from asone import utils
from asone import ASOne
import cv2

video_path = 'data/sample_videos/test.mp4'
detector = ASOne(detector=asone.YOLOV7_PYTORCH, use_cuda=True) # Set use_cuda to False for cpu

filter_classes = None # Set to None to detect all classes

cap = cv2.VideoCapture(video_path)

file = open("./yolov8_annotations", "a+")
	
total_frames =  int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
counter += 0
os.mkdir('./images')
os.mkdir('./labels')
while True:
	_, frame = cap.read()
    	if not _:
		file.close()
        	break
	counter += 1
    	dets, img_info = detector.detect(frame, filter_classes=filter_classes)

	print(f"Frame {counter} / {total_frames}, total objects: {len(class_ids)}")
    	
	bbox_xyxy = dets[:, :4]
    	scores = dets[:, 4]
    	class_ids = dets[:, 5]
	
	# saving images
	cv.imwrite(f'./image/{counter}.jpg')
	
	# saving labels file
	with open(f'./labels/{counter}.txt', 'w+') as file:
		for idx in range(0, len(bbox_xyxy)):
			objects = f"{class_ids[idx]} {bbox_xyxy[0]} {bbox_xyxy[1]} {bbox_xyxy[2]} {bbox_xyxy[3]}"
			file.write(str(objects))

    	#frame = utils.draw_boxes(frame, bbox_xyxy, class_ids=class_ids)

    	#cv2.imshow('result', frame)
	
    	if cv2.waitKey(25) & 0xFF == ord('q'):
		file.close()
        	break
	file.close()
