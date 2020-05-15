import os
import sys
import time
import uuid
import glob
import cv2
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.face.models import TrainingStatusType

# Set the FACE_SUBSCRIPTION_KEY environment variable with your key as the value.
# This key will serve all examples in this document.
KEY = os.environ['FACE_SUBSCRIPTION_KEY']

# Set the FACE_ENDPOINT environment variable with the endpoint from your Face service in Azure.
# This endpoint will be used in all examples in this quickstart.
ENDPOINT = os.environ['FACE_ENDPOINT']

# Create an authenticated FaceClient.
face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))

# Used in the Person Group Operations,  Snapshot Operations, and Delete Person Group examples.
# You can call list_person_groups to print a list of preexisting PersonGroups.
# SOURCE_PERSON_GROUP_ID should be all lowercase and alphanumeric. For example, 'mygroupname' (dashes are OK).
PERSON_GROUP_ID = 'group58'

# Used for the Snapshot and Delete Person Group examples.
TARGET_PERSON_GROUP_ID = str(uuid.uuid4())  # assign a random ID (or name it anything)

'''
Create the PersonGroup
'''
# Create empty Person Group. Person Group ID must be lower case, alphanumeric, and/or with '-', '_'.
print('Person group:', PERSON_GROUP_ID)
face_client.person_group.create(person_group_id=PERSON_GROUP_ID, name=PERSON_GROUP_ID)

Rawan = face_client.person_group_person.create(PERSON_GROUP_ID, "Rawan")
Motasem = face_client.person_group_person.create(PERSON_GROUP_ID, "Motasem")
Raneem = face_client.person_group_person.create(PERSON_GROUP_ID, "Raneem")
Mayar = face_client.person_group_person.create(PERSON_GROUP_ID, "Mayar")
Omar = face_client.person_group_person.create(PERSON_GROUP_ID, "Omar")
Salman = face_client.person_group_person.create(PERSON_GROUP_ID, "Salman")

'''
Detect faces and register to correct person
'''
# Find all jpeg images of friends in working directory
Rawan_images = [file for file in glob.glob('*.jpeg') if file.startswith("Rawan")]
Motasem_images = [file for file in glob.glob('*.jpeg') if file.startswith("Motasem")]
Raneem_images = [file for file in glob.glob('*.jpeg') if file.startswith("Raneem")]
Mayar_images = [file for file in glob.glob('*.jpeg') if file.startswith("Mayar")]
Omar_images = [file for file in glob.glob('*.jpeg') if file.startswith("Omar")]
Salman_images = [file for file in glob.glob('*.jpeg') if file.startswith("Salman")]

for image in Rawan_images:
    Rn = open(image, 'r+b')
    face_client.person_group_person.add_face_from_stream(PERSON_GROUP_ID, Rawan.person_id, Rn)

for image in Motasem_images:
    Mm = open(image, 'r+b')
    face_client.person_group_person.add_face_from_stream(PERSON_GROUP_ID, Motasem.person_id, Mm)

for image in Raneem_images:
    Rm = open(image, 'r+b')
    face_client.person_group_person.add_face_from_stream(PERSON_GROUP_ID, Raneem.person_id, Rm)

for image in Mayar_images:
    Mr = open(image, 'r+b')
    face_client.person_group_person.add_face_from_stream(PERSON_GROUP_ID, Mayar.person_id, Mr)

for image in Omar_images:
    Or = open(image, 'r+b')
    face_client.person_group_person.add_face_from_stream(PERSON_GROUP_ID, Omar.person_id, Or)

for image in Salman_images:
    Sn = open(image, 'r+b')
    face_client.person_group_person.add_face_from_stream(PERSON_GROUP_ID, Salman.person_id, Sn)

'''
Train PersonGroup
'''
print()
print('Training the person group...')
# Train the person group
face_client.person_group.train(PERSON_GROUP_ID)

while (True):
    training_status = face_client.person_group.get_training_status(PERSON_GROUP_ID)
    print("Training status: {}.".format(training_status.status))
    print()
    if (training_status.status is TrainingStatusType.succeeded):
        break
    elif (training_status.status is TrainingStatusType.failed):
        sys.exit('Training the person group has failed.')
    time.sleep(5)

'''
Identify a face against a defined PersonGroup
'''
# Group image for testing against
#group_photo = 'people1.jpeg'
#IMAGES_FOLDER = os.path.join(os.path.dirname(os.path.realpath(__file__)))
# Get test image
#test_image_array = glob.glob(os.path.join(IMAGES_FOLDER, group_photo))
#image = open(test_image_array[0], 'r+b')

# Detect faces
#face_ids = []
#faces = face_client.face.detect_with_stream(image)
#for face in faces:
#    face_ids.append(face.face_id)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
cap = cv2.VideoCapture(0)
while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    my_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    # Draw a rectangle around the faces
    for (x, y, w, h) in my_faces:
        print(x, y, w, h)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        image_item = 'face_test.jpeg'
        color = (255, 0, 0)
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)

    save_dist = "C:/Users/Dell XPS OAD/PycharmProjects/Rawan_Abuzaid"
    cv2.imwrite(os.path.join(save_dist, "test.jpeg"), frame)
    # Display the resulting frame
    cv2.imshow('frame', frame)
    break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()

# Group image for testing against
group_photo = 'test.jpeg'
IMAGES_FOLDER = os.path.join(os.path.dirname(os.path.realpath(__file__)))
# Get test image
test_image_array = glob.glob(os.path.join(IMAGES_FOLDER, group_photo))
image = open(test_image_array[0], 'r+b')

# Detect faces
face_ids = []
faces = face_client.face.detect_with_stream(image)
for face in faces:
    face_ids.append(face.face_id)

# Identify faces
results = face_client.face.identify(face_ids, PERSON_GROUP_ID)
print('Identifying faces in {}'.format(os.path.basename(image.name)))
if not results:
    print('No person identified in the person group for faces from {}.'.format(os.path.basename(image.name)))
for person in results:
    print('Person for face ID {} is identified in {} with a confidence of {}.'.format(person.face_id,
                                                                                      os.path.basename(image.name),
                                                                                      person.candidates[0].confidence))
# Delete the main person group.
face_client.person_group.delete(person_group_id=PERSON_GROUP_ID)
print("Deleted the person group {} from the source location.".format(PERSON_GROUP_ID))
print()