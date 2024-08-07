import cv2 as cv
# from tqdm.notebook import tqdm # For Colab Users

from age_recognition import age_gender_detector, getFaceBox

faceProto = "Model Weights/opencv_face_detector.pbtxt"
faceModel = "Model Weights/opencv_face_detector_uint8.pb"

ageProto = "Model Weights/age_deploy.prototxt"
ageModel = "Model Weights/age_net.caffemodel"

genderProto = "Model Weights/gender_deploy.prototxt"
genderModel = "Model Weights/gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# Load network
ageNet = cv.dnn.readNet(ageModel, ageProto)
genderNet = cv.dnn.readNet(genderModel, genderProto)
faceNet = cv.dnn.readNet(faceModel, faceProto)


VIDEO_PATH = 'clip.mp4'
vid = cv.VideoCapture(VIDEO_PATH)

frame_width = int(vid.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(vid.get(cv.CAP_PROP_FRAME_HEIGHT))
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('processed_video.avi', fourcc, 20.0, (frame_width, frame_height))
total_frames = int(vid.get(cv.CAP_PROP_FRAME_COUNT))  # Total number of frames
# pbar = tqdm(total=total_frames) # colab users


while vid.isOpened():
    ret, frame = vid.read()
      
    if not ret:
      break
    detection_frame = age_gender_detector(frame)
    out.write(detection_frame)
    # pbar.update(1) # colab users

vid.release()
out.release()
cv.destroyAllWindows()
