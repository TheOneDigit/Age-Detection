import cv2 as cv
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

image_path = ''
image = cv.imread(image_path)

# cv.imshow(image) # non-colab user 

# colab user
# from google.colab.patches import cv2_imshow 
# cv2_imshow(output) 
