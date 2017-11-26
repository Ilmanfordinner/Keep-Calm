from __future__ import print_function
import time 
import requests
import cv2
import operator
import numpy as np
from PIL import Image
import io
import urllib, base64
import threading
import copy


_region = 'westus'
_url = 'https://{}.api.cognitive.microsoft.com/emotion/v1.0/recognize'.format(_region)
_key = '823b722df60c4db7bb766c792079f29e'
_maxNumRetries = 10
_requestFrequency = 3.1

_current_frame = None
_lastRequestTime = 0
_latest_response = []
_face_cascade = None

_global_edited_webcam_frame = None


_players_array = None
_players_override = None
_persistent_faces = True

_winnerTime = 8
_lastWin = 0

_point_results = []

def processRequest(json, data, headers, params):

    """
    Helper function to process the request to Project Oxford

    Parameters:
    json: Used when processing images from its URL. See API Documentation
    data: Used when processing image read from disk. See API Documentation
    headers: Used to pass the key information and the data type request
    """

    retries = 0
    result = None

    while True:

        response = requests.request('post', _url, json = json, data = data, headers = headers, params = params )

        if response.status_code == 429: 

            print( "Message: %s" % ( response.json() ) )

            if retries <= _maxNumRetries: 
                time.sleep(1) 
                retries += 1
                continue
            else: 
                print( 'Error: failed after retrying!' )
                break

        elif response.status_code == 200 or response.status_code == 201:

            if 'content-length' in response.headers and int(response.headers['content-length']) == 0: 
                result = None 
            elif 'content-type' in response.headers and isinstance(response.headers['content-type'], str): 
                if 'application/json' in response.headers['content-type'].lower(): 
                    result = response.json() if response.content else None 
                elif 'image' in response.headers['content-type'].lower(): 
                    result = response.content
        else:
            print( "Error code: %d" % ( response.status_code ) )
            print( "Message: %s" % ( response.json() ) )

        break
        
    return result


def BGR2RGB(array):
    return np.roll(array, 1, axis=-1)
            
# Compresses images from np.arrays to JPEGS to make queries to Cognitive API faster
def nparrayToCompressedByteArray(array):
    
    array = BGR2RGB(array)
    raw_data = array.tobytes()
    height, width, channels = array.shape
    
    raw_image = Image.frombytes("RGB", (width, height), raw_data)
    imgByteArr = io.BytesIO()
    raw_image.save(imgByteArr, format='JPEG')
    return imgByteArr.getvalue()

# Image saving function
def sv(array):
    
    array = BGR2RGB(array)
    raw_data = array.tobytes()
    height, width, channels = array.shape
    
    raw_image = Image.frombytes("RGB", (width, height), raw_data)
    raw_image.save(r'C:\tmp\5.png', format='PNG')
    

def sendRequest(array):
    compressed = array
    compressed = nparrayToCompressedByteArray(compressed)
    
    headers = dict()
    headers['Ocp-Apim-Subscription-Key'] = _key
    headers['Content-Type'] = 'application/octet-stream'

    json = None
    
    params = urllib.parse.urlencode({})

    result = processRequest( json, compressed, headers, params )
    
    return result

# Thread for sending requests to Cognitive Services
def requestThread():

    global _lastRequestTime
    global _latest_response
    
    while (True):
        time.sleep(_requestFrequency)
        timeTillNextRequest = _requestFrequency + _lastRequestTime - time.time()
        if timeTillNextRequest > 0:
            time.sleep(timeTillNextRequest)
        
        _lastRequestTime = time.time()
        _latest_response = sendRequest(_current_frame)
        print (_latest_response)
        

# Draws a rectangle on a face
def drawFaceInfo(location, target):
    
    green = (0,255,0)
    #print(location)
    
    x, y, w, h = location
    
    cv2.rectangle(target, (x, y), (x+w, y+h), green, 5)

# Draws a small np.array image onto a bigger one. Takes alpha channels into account.
def drawImage(s_img, l_img, x_offset, y_offset):
    y1, y2 = y_offset, y_offset + s_img.shape[0]
    x1, x2 = x_offset, x_offset + s_img.shape[1]
    
    if len(s_img[0][0]) == 4:
        alpha_s = s_img[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s
    else:
        alpha_s = np.ones((s_img.shape[0], s_img.shape[1]))
        alpha_l = np.zeros((s_img.shape[0], s_img.shape[1]))

    for c in range(0, 3):
        l_img[y1:y2, x1:x2, c] = (alpha_s * s_img[:, :, c] + alpha_l * l_img[y1:y2, x1:x2, c])

# If somebody's state has been changed, it gets updated globally
def updateNearestPlayer(location, emoji):
    x, y, w, h = location
    to_update = 0
    for index in range(1, len(_players_array)):
        dist1 = abs(_players_array[to_update][0] - x)
        dist2 = abs(_players_array[index][0] - x)
        if dist2 < dist1:
            to_update = index

    _players_override[to_update] = emoji

# Draws the apt. emoji on the apt. person
def drawEmoji(info, location, target, override_emoji=''):
        
    best = ('kek', -1)
    for f, s in info['scores'].items():
        if s > best[1]:
            best = (f, s)
    
    emotion = best[0]
    if _persistent_faces and override_emoji != '':
        emotion = override_emoji

    if emotion == 'anger' or emotion == 'contempt' or emotion == 'disgust':
        emoji_raw = cv2.imread("angry_emoji.png", -1)
    elif emotion == 'fear' or emotion == 'surprise':
        emoji_raw = cv2.imread("fear_emoji.png", -1)
    elif emotion == 'happiness':    
        emoji_raw = cv2.imread("laughing_crying_emoji2.png", -1)
    elif emotion == 'sadness':
        emoji_raw = cv2.imread("sad_emoji.png", -1)
    elif emotion == 'winner':
        emoji_raw = cv2.imread('winner_emoji.png', -1)
    else:
        return ''
    
    
    x, y, w, h = location
    emoji = cv2.resize(emoji_raw, (w, h))
    drawImage(emoji, target, x, y)
    return emotion


# Draws all the information we have about the frame (emojis, rects, etc.) on the frame
def drawInfo(frame):
       
    global _lastWin
    global _players_array
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    face_locations = _face_cascade.detectMultiScale(grayscale, scaleFactor = 1.1, minNeighbors = 5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    tmp_response = _latest_response
    
    tmp_response = sorted(tmp_response, key=lambda face: face['faceRectangle']['left'])
    
    face_locations = sorted(face_locations, key=lambda loc: loc[0])
        
    if len(face_locations) == len(_players_array):
        _players_array = face_locations
    
    for index in range(0, min(len(tmp_response), len(face_locations))):
        print(_players_override)
        if index >= len(_players_override):
            kek = ''
        else:
            kek = _players_override[index]
        res = drawEmoji(tmp_response[index], face_locations[index], frame, kek)
        if res != '' and not 'winner' in _players_override:
            updateNearestPlayer(face_locations[index], res)
            emptycnt = 0
            lastIndex = -1
            for i in range(0, len(_players_override)):
                emptycnt += _players_override[i] == ""
                if _players_override[i] == "":
                    lastIndex = i
            if emptycnt == 1:
                _players_override[lastIndex] = 'winner'
                _lastWin = time.time()
                _point_results[lastIndex] += 1
    
        
# Draws rectangles on top of all faces in the frame. Also assists with global variable initialization.
def drawRects(frame):
        
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    face_locations = _face_cascade.detectMultiScale(grayscale, scaleFactor = 1.1, minNeighbors = 5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    #tmp_response = _latest_response
    
    #tmp_response = sorted(tmp_response, key=lambda face: face['faceRectangle']['left'])
    
    face_locations = sorted(face_locations, key=lambda loc: loc[0])

    global _players_array
    global _players_override
    global _point_results
    _players_array = face_locations
    _point_results = [0] * len(_players_array)
    _players_override = [''] * len(face_locations)
    print(_players_override)
    
    for index in range(0, len(face_locations)):
        drawFaceInfo(face_locations[index], frame)

# Webcam Thread for dealing with the webcam and most game logic
def webcamThread():
    
    global _global_edited_webcam_frame
    
    while True:
        if _current_frame is None:
            continue
        
        edited_frame = _current_frame.copy()
        
        drawInfo(edited_frame)
        
        _global_edited_webcam_frame = edited_frame
    
# Initial calibration
def initialize():
    cap = cv2.VideoCapture(0)
    
    global _face_cascade
    global _current_frame
    
    cascPath = "haarcascade_frontalface_default.xml"
    _face_cascade = cv2.CascadeClassifier(cascPath)

    _lastRequestTime = time.time()

    
    while(True):
        # Capture frame-by-frame
        ret, _current_frame = cap.read()
 
        # Our operations on the frame come her
        
        color = cv2.resize(cv2.cvtColor(_current_frame, cv2.IMREAD_COLOR), (0, 0), fx = 0.5, fy = 0.5)
        drawRects(_current_frame)
    
        # Display the resulting frame
        cv2.imshow('edited_frame',_current_frame)
        cv2.moveWindow('edited_frame', 0, 0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    
    
def webcam_main():
    #initialize()

    cap = cv2.VideoCapture(0)
    
    global _face_cascade
    global _current_frame
    global _players_override
    
    # Use cascade classifier for loal face position detection
    cascPath = "haarcascade_frontalface_default.xml"
    _face_cascade = cv2.CascadeClassifier(cascPath)

    _lastRequestTime = time.time()
    
    # Create a thread for polling requests to the Cognitive Services API
    t = threading.Thread(target = requestThread)
    t.setDaemon(True)
    t.start()
    
    # And one for polling the webcam
    webcam_thread = threading.Thread(target = webcamThread)
    webcam_thread.setDaemon(True)
    webcam_thread.start()
    

    
    while(True):
        # Capture frame-by-frame
        ret, _current_frame = cap.read()
        
        # Operations on the frame come here
        
        # Cases where we have an edited frame and not
        if not _global_edited_webcam_frame is None:
            color = cv2.resize(cv2.cvtColor(_global_edited_webcam_frame, cv2.IMREAD_COLOR), (0, 0), fx=0.5, fy=0.5)
        else:
            color = np.zeros((320, 240))
        
        # Case when there was a winner recently
        if time.time() - _lastWin < _winnerTime:
            for index in range(0, len(_players_array)):
                x, y, w, h = _players_array[index]
                x = x//2
                y = y//2
                w = w//2
                h = h//2
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(color,str(_point_results[index]),(x, y), font, 2,(255,255,255),2,cv2.LINE_AA)
        # Case when next round is about to start
        elif time.time() - _lastWin < _winnerTime + 1:
            for i in range(0, len(_players_override)):
                _players_override[i] = ''
    
        # Display the resulting frame
        cv2.imshow('edited_frame',color)
        cv2.moveWindow('edited_frame', 0, 0)
        if cv2.waitKey(16) & 0xFF == ord('q'):
            break
            
    
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def createWebcamMainThread():
    t = threading.Thread(target = webcam_main)
    t.setDaemon(True)
    t.start()
    
    
    return t


#if __name__ == "__main__":
#    main()
