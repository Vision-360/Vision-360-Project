from flask import Flask,render_template, Response, redirect, request, session, abort, url_for
from flask import Flask, render_template, request, redirect, url_for, session
import cv2
import face_recognition
import os
from deepface import DeepFace
import numpy as np
import imutils
import sys
from PIL import Image
import tensorflow as tf
from keras.preprocessing import image
import pafy
import youtube_dl
import pickle
import pyrebase
from datetime import datetime
import pandas as pd
from imutils import paths
import sendemail

app=Flask(__name__)
# cap=cv2.VideoCapture(0)

app.secret_key = 'verysecret'

@app.after_request
def add_header(response):
    response.headers['Pragma'] = 'no-cache'
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Expires'] = '0'
    response.cache_control.max_age = 0
    return response

#Add your own details
config = {
  "databaseURL": "https://vision360-minor-default-rtdb.firebaseio.com/",
  'apiKey': "AIzaSyCVqx66-uX7ObK1BDhXIxa34AF5tPaHKT0",
  'authDomain': "vision360-minor.firebaseapp.com",
  'projectId': "vision360-minor",
  'storageBucket': "vision360-minor.appspot.com",
  'messagingSenderId': "278180751682",
  'appId': "1:278180751682:web:c6a0401bd27cd934df80db",
  'measurementId': "G-05NX5W0DGF"
}

#initialize firebase
firebase = pyrebase.initialize_app(config)
auth = firebase.auth()
db = firebase.database()

#Initialze person as dictionary
person = {"is_logged_in": False, "name": "", "email": "", "uid": ""}

def encode():
    #get paths of each file in folder named Images
    #Images here contains my data(folders of various persons)
    imagePaths = list(paths.list_images('known_faces'))
    knownEncodings = []
    knownNames = []
    # loop over the image paths
    for (i, imagePath) in enumerate(imagePaths):
        # extract the person name from the image path
        name = imagePath.split(os.path.sep)[-2]
        # load the input image and convert it from BGR (OpenCV ordering)
        # to dlib ordering (RGB)
        image = cv2.imread(imagePath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #Use Face_recognition to locate faces
        boxes = face_recognition.face_locations(rgb,model='hog')
        # compute the facial embedding for the face
        encodings = face_recognition.face_encodings(rgb, boxes)
        # loop over the encodings
        for encoding in encodings:
            knownEncodings.append(encoding)
            knownNames.append(name)
    #save emcodings along with their names in dictionary data
    data = {"encodings": knownEncodings, "names": knownNames}
    #use pickle to save data into a file for later use
    f = open("face_enc", "wb")
    f.write(pickle.dumps(data))
    f.close()

# def generate_frames():
#     while True:
            
#         ## read the camera frame
#         success,frame=cap.read()
#         if not success:
#             break
#         else:
#             ret,buffer=cv2.imencode('.jpg',frame)
#             frame=buffer.tobytes()

#         yield(b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def log_motion():
    i=0
    with open('log/motion.csv','r+') as f:
        myDataList = f.readlines()
        # print(myDataList[-1][16:])
        now = datetime.now()
        dtString = now.strftime('%H:%M:%S, %d/%m/%Y')
        if myDataList[-1][16:]==dtString:
            return
        f.writelines(f'\n{"Motion Detected"},{dtString}')

def gen_motion():
    # frame_width = int( cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # frame_height =int( cap.get( cv2.CAP_PROP_FRAME_HEIGHT))
    # fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
    # out = cv2.VideoWriter("output.avi", fourcc, 5.0, (1280,720))
    flag=0
    # cap=cv2.VideoCapture('videos/Ground_Floor.mp4')
    cap=cv2.VideoCapture('rtsp://admin:om11072004@192.168.29.220:554/cam/realmonitor?channel=3&subtype=0')
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()
    while cap.isOpened():
        diff = cv2.absdiff(frame1, frame2)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)

            if cv2.contourArea(contour) < 3200:
                # flag=0
                continue
            flag+=1
            log_motion()
            cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame1, "Status: {}".format('Movement'), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (0, 0, 255), 2)
            if flag==10:
                print("why")
                # cv2.imwrite('motion_alert.jpg',frame1)
                # sendemail.SendMail('motion_alert.jpg')
                flag=13
        #cv2.drawContours(frame1, contours, -1, (0, 255, 0), 2)

        image = cv2.resize(frame1, (1280,720))
        # out.write(image)
        # cv2.imshow("feed", frame1)
        frame1 = frame2
        ret, frame2 = cap.read()
        ret, jpeg = cv2.imencode('.jpg', image)

        frame = jpeg.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# def gen_face():
#     KNOWN_FACES_DIR = 'known_faces'
#     TOLERANCE = 0.6
#     FRAME_THICKNESS = 3
#     FONT_THICKNESS = 2
#     MODEL = 'hog'  # default: 'hog', other one can be 'cnn' - CUDA accelerated (if available) deep-learning pretrained model

#     # video=cv2.VideoCapture('Cam4.mp4')
#     video = cv2.VideoCapture(0)

#     # Returns (R, G, B) from name
#     def name_to_color(name):
#         # Take 3 first letters, tolower()
#         # lowercased character ord() value rage is 97 to 122, substract 97, multiply by 8
#         color = [(ord(c.lower())-97)*8 for c in name[:3]]
#         return color


#     print('Loading known faces...')
#     known_faces = []
#     known_names = []

#     # We oranize known faces as subfolders of KNOWN_FACES_DIR
#     # Each subfolder's name becomes our label (name)
#     for name in os.listdir(KNOWN_FACES_DIR):

#         # Next we load every file of faces of known person
#         for filename in os.listdir(f'{KNOWN_FACES_DIR}/{name}'):

#             # Load an image
#             image = face_recognition.load_image_file(f'{KNOWN_FACES_DIR}/{name}/{filename}')

#             # Get 128-dimension face encoding
#             # Always returns a list of found faces, for this purpose we take first face only (assuming one face per image as you can't be twice on one image)
#             encoding = face_recognition.face_encodings(image)[0]

#             # Append encodings and name
#             known_faces.append(encoding)
#             known_names.append(name)

#     while True:
#         ret, image= video.read()

#         # This time we first grab face locations - we'll need them to draw boxes
#         locations = face_recognition.face_locations(image, model=MODEL)

#         encodings = face_recognition.face_encodings(image, locations)

#         for face_encoding, face_location in zip(encodings, locations):

#             # We use compare_faces (but might use face_distance as well)
#             # Returns array of True/False values in order of passed known_faces
#             results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)

#             # Since order is being preserved, we check if any face was found then grab index
#             # then label (name) of first matching known face withing a tolerance
#             match = None
#             if True in results:  # If at least one is true, get a name of first of found labels
#                 match = known_names[results.index(True)]
#                 print(f' - {match} from {results}')

#                 # Each location contains positions in order: top, right, bottom, left
#                 top_left = (face_location[3], face_location[0])
#                 bottom_right = (face_location[1], face_location[2])

#                 # Get color by name using our fancy function
#                 color = name_to_color(match)

#                 # Paint frame
#                 cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)

#                 # Now we need smaller, filled grame below for a name
#                 # This time we use bottom in both corners - to start from bottom and move 50 pixels down
#                 top_left = (face_location[3], face_location[2])
#                 bottom_right = (face_location[1], face_location[2] + 22)

#                 # Paint frame
#                 cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)

#                 # Wite a name
#                 cv2.putText(image, match, (face_location[3] + 10, face_location[2] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), FONT_THICKNESS)

#         # Show image
#         # cv2.imshow(filename, image)
#         ret, jpeg = cv2.imencode('.jpg', image)

#         frame = jpeg.tobytes()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
#         # if cv2.waitKey(1) & 0xFF == ord("q"):
#         #     break
#         # cv2.waitKey(0)
#         # cv2.destroyWindow(filename)

def log_face(name):
    with open('log/face.csv','r+') as f:
        myDataList = f.readlines()
        # print(myDataList[-1][16:])
        now = datetime.now()
        dtString = now.strftime('%H:%M:%S, %d/%m/%Y')
        # if myDataList[-1][16:]==dtString:
        #     return
        f.writelines(f'\n{name},{dtString}')

def gen_face():
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
    # load the known faces and embeddings saved in last file
    data = pickle.loads(open('face_enc', "rb").read())
    
    print("Streaming started")
    video_capture = cv2.VideoCapture(0)
    # video_capture = cv2.VideoCapture('http://192.168.29.248:4747/mjpegfeed')
    # loop over frames from the video file stream
    while True:
        # grab the frame from the threaded video stream
        ret, frame = video_capture.read()

        if not ret:
            print("[INFO] no frame read from stream - exiting")
            sys.exit(0)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray,
                                            scaleFactor=1.1,
                                            minNeighbors=5,
                                            minSize=(60, 60),
                                            flags=cv2.CASCADE_SCALE_IMAGE)
    
        # convert the input frame from BGR to RGB 
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # the facial embeddings for face in input
        encodings = face_recognition.face_encodings(rgb)
        names = []
        # loop over the facial embeddings incase
        # we have multiple embeddings for multiple fcaes
        for encoding in encodings:
        #Compare encodings with encodings in data["encodings"]
        #Matches contain array with boolean values and True for the embeddings it matches closely
        #and False for rest
            matches = face_recognition.compare_faces(data["encodings"],
            encoding)
            #set name =inknown if no encoding matches
            name = "Unknown"
            # check to see if we have found a match
            if True in matches:
                #Find positions at which we get True and store them
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}
                # loop over the matched indexes and maintain a count for
                # each recognized face face
                for i in matchedIdxs:
                    #Check the names at respective indexes we stored in matchedIdxs
                    name = data["names"][i]
                    #increase count for the name we got
                    counts[name] = counts.get(name, 0) + 1
                #set name which has highest count
                name = max(counts, key=counts.get)
    
    
            # update the list of names
            names.append(name)
            log_face(name)
            # loop over the recognized faces
            for ((x, y, w, h), name) in zip(faces, names):
                # rescale the face coordinates
                # draw the predicted face name on the image
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (0, 255, 0), 2)
        image = cv2.resize(frame, (1280,720))
        ret, jpeg = cv2.imencode('.jpg', image)

        frame = jpeg.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def gen_emotion():
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture('http://192.168.29.248:4747/mjpegfeed')
    while True:
        ret, frame = cap.read() ## read one image from a video

        if not ret:
            print("[INFO] no frame read from stream - exiting")
            sys.exit(0)
        result =DeepFace.analyze (frame, actions = ['emotion'])

        gray = cv2.cvtColor (frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(gray,1.1,4)

        # Draw a rectangle around the faces 
        for(x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX

        # Use putText() method for inserting text in video

        cv2.putText(frame, result['dominant_emotion'],(50,50),font,3,(0,0,255),2,cv2.LINE_4)
        # cv2.imshow('Original Video',frame)
        image = cv2.resize(frame, (1280,720))
        ret, jpeg = cv2.imencode('.jpg', image)

        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def log_activity(label):
    with open('log/activity.csv','r+') as f:
        myDataList = f.readlines()
        # print(myDataList[-1][16:])
        now = datetime.now()
        dtString = now.strftime('%H:%M:%S, %d/%m/%Y')
        # if myDataList[-1][16:]==dtString:
        #     return
        f.writelines(f'\n{label},{dtString}')

def gen_activity():
    CLASSES = open("action_recognition_kinetics.txt").read().strip().split("\n")
    SAMPLE_DURATION = 16
    SAMPLE_SIZE = 112

    print("[INFO] loading human activity recognition model...")
    net = cv2.dnn.readNet("resnet-34_kinetics.onnx")

    print("[INFO] accessing video stream...")
    # vs = cv2.VideoCapture('20211017181851.mp4')
    # vs = cv2.VideoCapture('Cam4.mp4')
    vs = cv2.VideoCapture(0)
    # vs = cv2.VideoCapture('http://192.168.29.248:4747/mjpegfeed')

    while True:
        
        frames = []

        
        for i in range(0, SAMPLE_DURATION):
            (grabbed, frame) = vs.read()

            
            if not grabbed:
                print("[INFO] no frame read from stream - exiting")
                sys.exit(0)

            
            frame = imutils.resize(frame, width=400)
            frames.append(frame)

        blob = cv2.dnn.blobFromImages(frames, 1.0,
            (SAMPLE_SIZE, SAMPLE_SIZE), (114.7748, 107.7354, 99.4750),
            swapRB=True, crop=True)
        blob = np.transpose(blob, (1, 0, 2, 3))
        blob = np.expand_dims(blob, axis=0)

        
        net.setInput(blob)
        outputs = net.forward()
        # label = CLASSES[np.argmax(outputs)]
        print(outputs[0][np.argmax(outputs)],CLASSES[np.argmax(outputs)])
        if outputs[0][np.argmax(outputs)]>6.85:
            label = CLASSES[np.argmax(outputs)]
        else:
            label='unknown'
        
        log_activity(label)

        for frame in frames:
            cv2.rectangle(frame, (0, 0), (300, 40), (0, 0, 0), -1)
            cv2.putText(frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2)
            image = cv2.resize(frame, (1280,720))
            # out.write(image)
            # cv2.imshow("Activity Recognition", frame)
            ret, jpeg = cv2.imencode('.jpg', image)

            frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            # key = cv2.waitKey(1) & 0xFF

            # if key == ord("q"):
            #     break

def log_fire():
    with open('log/fire.csv','r+') as f:
        myDataList = f.readlines()
        # print(myDataList[-1][16:])
        now = datetime.now()
        dtString = now.strftime('%H:%M:%S, %d/%m/%Y')
        # if myDataList[-1][16:]==dtString:
        #     return
        f.writelines(f'\n{"Fire Detected"},{dtString}')

def gen_fire():
    model = tf.keras.models.load_model('InceptionV3.h5')
    f=0
    # cap=cv2.VideoCapture('20211017181851.mp4')
    # cap=cv2.VideoCapture('Cam4.mp4')
    cap=cv2.VideoCapture('videos/fire_cut.mp4')
    # url= "https://www.youtube.com/watch?v=whlymAuRtzU"
    # video = pafy.new(url)
    # best = video.getbest(preftype="mp4")
    # cap = cv2.VideoCapture()
    # cap.open(best.url)  
    while True:
        _, frame = cap.read()
        if not _:
            print("[INFO] no frame read from stream - exiting")
            sys.exit(0)
        #Convert the captured frame into RGB
        im = Image.fromarray(frame, 'RGB')
        #Resizing into 224x224 because we trained the model with this image size.
        im = im.resize((224,224))
        img_array = image.img_to_array(im)
        img_array = np.expand_dims(img_array, axis=0) / 255
        probabilities = model.predict(img_array)[0]
        #Calling the predict method on model to predict 'fire' on the image
        prediction = np.argmax(probabilities)
        #if prediction is 0, which means there is fire in the frame.
        if prediction == 0:
                log_fire()
                cv2.rectangle(frame, (0, 0), (300, 40), (0, 0, 0), -1)
                cv2.putText(frame, 'FIRE DETECTED', (10, 25), cv2.FONT_HERSHEY_SIMPLEX,0.8, (255, 255, 255), 3)
                print(probabilities[prediction])
                f=f+1
                if f==12:
                    cv2.imwrite('fire_alert.jpg',frame)
                    sendemail.SendMail('fire_alert.jpg')
                    flag=13
        ima = cv2.resize(frame, (1280,720))

        ret, jpeg = cv2.imencode('.jpg', ima)

        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/video')
def video():
    return render_template('video.html')

@app.route('/video2')
def video2():
    return render_template('video2.html')

@app.route('/video3')
def video3():
    return render_template('video3.html')

@app.route('/video4')
def video4():
    return render_template('video4.html')

@app.route('/video5')
def video5():
    return render_template('video5.html')

@app.route('/dashboard')
def dashboard():
    return render_template('index.html')

@app.route('/login', methods = ["POST", "GET"])
def login():
    msg=''
    if request.method == "POST":        #Only if data has been posted
        print("yo3")
        result = request.form           #Get the data
        email = result["email"]
        password = result["password"]
        try:
            #Try signing in the user with the given information
            user = auth.sign_in_with_email_and_password(email, password)
            verified = auth.get_account_info(user['idToken'])['users'][0]['emailVerified']
            if not verified:
                msg = ("Please verify your email")      
                return render_template('login.html',msg=msg)
            print("Successful!")
            #Insert the user data in the global person
            # global person
            # person["is_logged_in"] = True
            # person["email"] = user["email"]
            # person["uid"] = user["localId"]
            session['loggedin'] = True
            session['uid'] = user['localId']
            session['email'] = user['email']
            #Get the name of the user
            data = db.child("users").get()
            session['name'] = data.val()[session["uid"]]["name"]
            #Redirect to welcome page
            if session:
                # return render_template('index.html')
                return redirect(url_for('dashboard'))
        except:
            #If there is any error, redirect back to login
            msg="Invalid Email or Password"
            return render_template('login.html',msg=msg)
    # else:
    #     if person["is_logged_in"] == True:
    #         return redirect(url_for('welcome'))
    #     else:
    #         return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/video_motion')
def video_motion():
    return Response(gen_motion(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_face')
def video_face():
    return Response(gen_face(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_emotion')
def video_emotion():
    return Response(gen_emotion(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_activity')
def video_activity():
    return Response(gen_activity(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_fire')
def video_fire():
    return Response(gen_fire(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/register", methods = ["POST", "GET"])
def register():
    msg=""
    if request.method == "POST":        #Only listen to POST
        result = request.form           #Get the data submitted
        print("yo2")
        email = result["email"]
        name = result["username"]
        # city = result["city"]
        # pn = result["phone"]
        password = result["password"]
    #     print(email, password)
    #     return render_template('register2.html',msg="Done")
    # else:
    #     return render_template('register2.html')

        try:
            #Try creating the user account using the provided data
            user = auth.create_user_with_email_and_password(email, password)
            auth.send_email_verification(user['idToken'])
            #Login the user
            # user = auth.sign_in_with_email_and_password(email, password)
            #Add data to global person
            # global person
            # person["is_logged_in"] = True
            # person["email"] = user["email"]
            # person["uid"] = user["localId"]
            # person["name"] = name
            # Append data to the firebase realtime database
            data = {"name": name, "email": email}
            db.child("users").child(user["localId"]).set(data)
            #Go to welcome page
            # return redirect(url_for('register'), msg="Please go to your email for verification")
            # return redirect(url_for('register'))
            msg="Please go to your email for verification"
        except:
            #If there is any error, redirect to register
            msg="There is an error please try again"
            # return redirect(url_for('register'))
    
    return render_template('register2.html', msg=msg)
    # else:
    #     if person["is_logged_in"] == True:
    #         return redirect(url_for('welcome'))
    #     else:
    #         return redirect(url_for('register'))

# @app.route("/register", methods = ["POST", "GET"])
# def register():
#     return render_template('register2.html')

@app.route('/details')
def details():
    data = pd.read_csv('log/motion.csv', header=0)
    data_face=pd.read_csv('log/face.csv', header=0)
    data_act=pd.read_csv('log/activity.csv', header=0)
    data_fire=pd.read_csv('log/fire.csv', header=0)
    # myData = list(data.values)
    m1=str(data.tail(5).iloc[4][2])+" "+str(data.tail(5).iloc[4][1])+" "+str(data.tail(5).iloc[4][0])
    m2=str(data.tail(5).iloc[3][2])+" "+str(data.tail(5).iloc[3][1])+" "+str(data.tail(5).iloc[3][0])
    m3=str(data.tail(5).iloc[2][2])+" "+str(data.tail(5).iloc[2][1])+" "+str(data.tail(5).iloc[2][0])
    m4=str(data.tail(5).iloc[1][2])+" "+str(data.tail(5).iloc[1][1])+" "+str(data.tail(5).iloc[1][0])
    m5=str(data.tail(5).iloc[0][2])+" "+str(data.tail(5).iloc[0][1])+" "+str(data.tail(5).iloc[0][0])

    fa1=str(data_face.tail(5).iloc[4][2])+" "+str(data_face.tail(5).iloc[4][1])+" "+str(data_face.tail(5).iloc[4][0])
    fa2=str(data_face.tail(5).iloc[3][2])+" "+str(data_face.tail(5).iloc[3][1])+" "+str(data_face.tail(5).iloc[3][0])
    fa3=str(data_face.tail(5).iloc[2][2])+" "+str(data_face.tail(5).iloc[2][1])+" "+str(data_face.tail(5).iloc[2][0])
    fa4=str(data_face.tail(5).iloc[1][2])+" "+str(data_face.tail(5).iloc[1][1])+" "+str(data_face.tail(5).iloc[1][0])
    fa5=str(data_face.tail(5).iloc[0][2])+" "+str(data_face.tail(5).iloc[0][1])+" "+str(data_face.tail(5).iloc[0][0])

    a1=str(data_act.tail(5).iloc[4][2])+" "+str(data_act.tail(5).iloc[4][1])+" "+str(data_act.tail(5).iloc[4][0])
    a2=str(data_act.tail(5).iloc[3][2])+" "+str(data_act.tail(5).iloc[3][1])+" "+str(data_act.tail(5).iloc[3][0])
    a3=str(data_act.tail(5).iloc[2][2])+" "+str(data_act.tail(5).iloc[2][1])+" "+str(data_act.tail(5).iloc[2][0])
    a4=str(data_act.tail(5).iloc[1][2])+" "+str(data_act.tail(5).iloc[1][1])+" "+str(data_act.tail(5).iloc[1][0])
    a5=str(data_act.tail(5).iloc[0][2])+" "+str(data_act.tail(5).iloc[0][1])+" "+str(data_act.tail(5).iloc[0][0])

    f1=str(data_fire.tail(5).iloc[4][2])+" "+str(data_fire.tail(5).iloc[4][1])+" "+str(data_fire.tail(5).iloc[4][0])
    f2=str(data_fire.tail(5).iloc[3][2])+" "+str(data_fire.tail(5).iloc[3][1])+" "+str(data_fire.tail(5).iloc[3][0])
    f3=str(data_fire.tail(5).iloc[2][2])+" "+str(data_fire.tail(5).iloc[2][1])+" "+str(data_fire.tail(5).iloc[2][0])
    f4=str(data_fire.tail(5).iloc[1][2])+" "+str(data_fire.tail(5).iloc[1][1])+" "+str(data_fire.tail(5).iloc[1][0])
    f5=str(data_fire.tail(5).iloc[0][2])+" "+str(data_fire.tail(5).iloc[0][1])+" "+str(data_fire.tail(5).iloc[0][0])

    return render_template('details.html', str_motion_1=m1, str_motion_2=m2, str_motion_3=m3, str_motion_4=m4, str_motion_5=m5,
                                           str_face_1=fa1, str_face_2=fa2, str_face_3=fa3, str_face_4=fa4, str_face_5=fa5,
                                           str_act_1=a1, str_act_2=a2, str_act_3=a3, str_act_4=a4, str_act_5=a5,
                                           str_fire_1=f1, str_fire_2=f2, str_fire_3=f3, str_fire_4=f4, str_fire_5=f5)


@app.route('/contactus')
def contactus():
    return render_template('contactus.html')

@app.route('/aboutus')
def aboutus():
    return render_template('aboutus.html')

if __name__=="__main__":
    app.run(debug=True)