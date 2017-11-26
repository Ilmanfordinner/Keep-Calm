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


_region = 'westus' #Here you enter the region of your subscription
_url = 'https://{}.api.cognitive.microsoft.com/emotion/v1.0/recognize'.format(_region)
_key = 'd0bfdec61a7841609a35fda0c12d7788'
_maxNumRetries = 10
_requestFrequency = 3.1

_current_frame = None
_lastRequestTime = 0
_latest_response = []
_face_cascade = None

_global_edited_webcam_frame = None


_players_array = None
_players_override = None
_persistent_faces = False

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
            
    
def nparrayToCompressedByteArray(array):
    
    array = BGR2RGB(array)
    raw_data = array.tobytes()
    height, width, channels = array.shape
    
    raw_image = Image.frombytes("RGB", (width, height), raw_data)
    imgByteArr = io.BytesIO()
    raw_image.save(imgByteArr, format='JPEG')
    return imgByteArr.getvalue()

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
        
        
def drawFaceInfo(info, location, target):
    neutralness = info['scores']['neutral']
    
    green = (0,255,0)
    #print(location)
    
    x, y, w, h = location
    
    cv2.rectangle(target, (x, y), (x+w, y+h), green, 5)
    
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
        
def updateNearestPlayer(location, emoji):
    x, y, w, h = location
    to_update = 0
    for index in range(1, len(_players_array)):
        dist1 = abs(_players_array[to_update][0] - x)
        dist2 = abs(_players_array[index][0] - x)
        if dist2 < dist1:
            to_update = index

    _players_override[to_update] = emoji

def drawEmoji(info, location, target, override_emoji=''):
        
    best = ('kek', -1)
    for f, s in info['scores'].items():
        if s > best[1]:
            best = (f, s)
    
    #print(type(best[0]))
    emotion = best[0]
    if _persistent_faces and override_emoji != '':
        emotion = override_emoji

    #print(type(best))
    if emotion == 'anger' or emotion == 'contempt' or emotion == 'disgust':
        emoji_raw = cv2.imread("angry_emoji.png", -1)
    elif emotion == 'fear' or emotion == 'surprise':
        emoji_raw = cv2.imread("fear_emoji.png", -1)
    elif emotion == 'happiness':    
        emoji_raw = cv2.imread("laughing_crying_emoji2.png", -1)
    elif emotion == 'sadness':
        emoji_raw = cv2.imread("sad_emoji.png", -1)
    else:
        return ''
    
    
    x, y, w, h = location
    emoji = cv2.resize(emoji_raw, (w, h))
    drawImage(emoji, target, x, y)
    return emotion

        
def drawInfo(frame):
        
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    face_locations = _face_cascade.detectMultiScale(grayscale, scaleFactor = 1.1, minNeighbors = 5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    tmp_response = _latest_response
    
    tmp_response = sorted(tmp_response, key=lambda face: face['faceRectangle']['left'])
    
    face_locations = sorted(face_locations, key=lambda loc: loc[0])
    
    #print(_players_array)
    for index in range(0, min(len(tmp_response), len(face_locations))):
        print(_players_override)
        res = drawEmoji(tmp_response[index], face_locations[index], frame, _players_override[index])
        if res != '':
            updateNearestPlayer(face_locations[index], res)

def drawRects(frame):
        
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    face_locations = _face_cascade.detectMultiScale(grayscale, scaleFactor = 1.1, minNeighbors = 5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    tmp_response = _latest_response
    
    tmp_response = sorted(tmp_response, key=lambda face: face['faceRectangle']['left'])
    
    face_locations = sorted(face_locations, key=lambda loc: loc[0])

    global _players_array
    global _players_override
    _players_array = face_locations
    _players_override = [''] * len(face_locations)
    print(_players_override)
    
    for index in range(0, min(len(tmp_response), len(face_locations))):
        drawFaceInfo(tmp_response[index], face_locations[index], frame)


def webcamThread():
    
    global _global_edited_webcam_frame
    
    while True:
        if _current_frame is None:
            continue
        
        edited_frame = _current_frame.copy()
        
        drawInfo(edited_frame)
        
        _global_edited_webcam_frame = edited_frame
    

def initialize():
    cap = cv2.VideoCapture(0)
    
    global _face_cascade
    global _current_frame
    
    cascPath = "haarcascade_frontalface_default.xml"
    _face_cascade = cv2.CascadeClassifier(cascPath)

    _lastRequestTime = time.time()
    
    t = threading.Thread(target = requestThread)
    t.setDaemon(True)
    t.start()

    
    while(True):
        # Capture frame-by-frame
        ret, _current_frame = cap.read()
 
        # Our operations on the frame come her
        
        color = cv2.resize(cv2.cvtColor(_current_frame, cv2.IMREAD_COLOR), (0, 0), fx = 0.5, fy = 0.5)
        drawRects(_current_frame)
    
        # Display the resulting frame
        cv2.imshow('edited_frame',_current_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


def main():
    initialize()

    cap = cv2.VideoCapture(0)
    videocap = cv2.VideoCapture('1.mp4')
    
    global _face_cascade
    global _current_frame
    
    cascPath = "haarcascade_frontalface_default.xml"
    _face_cascade = cv2.CascadeClassifier(cascPath)

    _lastRequestTime = time.time()
    
    t = threading.Thread(target = requestThread)
    t.setDaemon(True)
    t.start()
    
    webcam_thread = threading.Thread(target = webcamThread)
    webcam_thread.setDaemon(True)
    webcam_thread.start()
    

    
    while(True):
        # Capture frame-by-frame
        ret, _current_frame = cap.read()
        ret, video_frame = videocap.read()
        
    
        # Our operations on the frame come here
        
        
        if not _global_edited_webcam_frame is None:
            color = cv2.resize(cv2.cvtColor(_global_edited_webcam_frame, cv2.IMREAD_COLOR), (0, 0), fx = 0.5, fy = 0.5)
            drawImage(color, video_frame, 0, 0)
        
    
        # Display the resulting frame
        cv2.imshow('edited_frame',video_frame)
        if cv2.waitKey(16) & 0xFF == ord('q'):
            break
            
    
    # When everything done, release the capture
    cap.release()
    videocap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

