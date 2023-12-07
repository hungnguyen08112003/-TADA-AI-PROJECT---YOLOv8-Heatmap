import cv2

#Parameters
video_file="video.mp4"

#Some functions helper
video=cv2.VideoCapture(video_file)

while True:
    ret, frame = video.read()
    if ret:
        cv2.inshow("Video Heatmap", frame)
        if cv2.waitKey(1)==ord('q'):
            break

video.release()
cv2.DestroyAllWindows()

