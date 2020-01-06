from imutils.video import VideoStream
import argparse
import datetime
import imutils
import time
import cv2
 
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--stream", help="empty or rtsp://stream-address")
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
ap.add_argument("-z", "--zone", type=float, default=0.5, help="percentage of red zone area")
args = vars(ap.parse_args())
 
if args.get("stream", None) is None:
	vs = VideoStream(src=0).start()
	time.sleep(1.0)

else:
	vs = cv2.VideoCapture(args["stream"])
 
firstFrame = None

while True:
	frame = vs.read()

	frame = frame if args.get("stream", None) is None else frame[1]
	text = "Safe"
 
	if frame is None:
		break
 
	frame = imutils.resize(frame, width=500)
	width = frame.shape[0]
	height = frame.shape[1]
	redzone_width = int(width * args.get("zone"))
	redzone = frame[:, :redzone_width]

	gray = cv2.cvtColor(redzone, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (21, 21), 0)
 
	if firstFrame is None:
		firstFrame = gray
		continue
	frameDelta = cv2.absdiff(firstFrame, gray)
	thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
	# dilate the thresholded image to fill in holes, then find contours
	# on thresholded image
	thresh = cv2.dilate(thresh, None, iterations=2)
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)

	for c in cnts:
		(x, y, w, h) = cv2.boundingRect(c)
		if w * h < args["min_area"]:
			continue
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
		text = "Danger"
	cv2.putText(frame, f"Status: {text}", (10, 20),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

	cv2.line(frame, (redzone_width, 0), (redzone_width, height), (0,255,0), 2)
 
	cv2.imshow("Motion detector", frame)
	key = cv2.waitKey(1) & 0xFF
 
	if key == ord("q"):
		break
 
vs.stop() if args.get("stream", None) is None else vs.release()
cv2.destroyAllWindows()