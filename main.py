import os
import sys
import pygame
import vlc
import video_handler
import webcam_handler
import threading
import time
import cv2
import numpy as np
import random


# Debug info
def callback(self, player):

	print ('FPS =',  player.get_fps())
	print ('time =', player.get_time(), '(ms)')
	print ('FRAME =', .001 * player.get_time() * player.get_fps())

# Loading screen for when YouTube videos get downloaded
def loadingScreen(targetVideo):
    
    dots = 0
    ticks = 0
    phr_ind = 0
    phrases = [
        "Building walls.",
        "Duding Wednesdays.",
        "Snooping dogs.",
        "Slimming shadies.",
        "Spinning fidgets.",
        "Danking memes.",
        "Downloading videos.",
        "Omaeing wa mou shindeirus.",
        "Writing KYP.",
        "Stringing theories.",
        "Quanting computers.",
        "Moving emojies."
        ]
        
    random.shuffle(phrases)
    
    
    while (targetVideo >= video_handler.getLoadedVideoCount()):
        dottext = phrases[phr_ind]
        for i in range(0, dots):
            dottext = dottext + '.'
        
        frame = np.zeros((200, 600))
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame,dottext,(25, 120), font, 1.25,(255,255,255),2,cv2.LINE_AA)
        
        cv2.imshow("Downloading videos...", frame)
        cv2.moveWindow('Downloading videos...', 900, 550)
        if cv2.waitKey(1000) & 0xFF == ord('q'):
            sys.exit(0)
            
        dots = (dots + 1)%3
        ticks = (ticks + 1) % 6
        if ticks == 0:
            phr_ind = (phr_ind + 1) %len(phrases)
            text = phrases[phr_ind]
        
    cv2.destroyWindow("Downloading videos...")

    
# Function that tells us if we're ingame
def inGame():
    return True
    
def main(args):
    
    # Take playlists from the console
    if len(sys.argv) > 1:
        target_playlist = sys.argv[1]
    else:   # Or use some presets
        #target_playlist = 'https://www.youtube.com/playlist?list=PL81280E14A07C995D'
        target_playlist = 'https://www.youtube.com/playlist?list=PLy3-VH7qrUZ5IVq_lISnoccVIYZCMvi-8'
    
    # Starting the video downloading thread
    video_thread = video_handler.create_playlist_loading_thread(target_playlist)
        
    
    # Starting the webcam and face detection threads
    webcam_handler.initialize()
    
    
    
    # Setting up PyGame for use with VLC
    pygame.init()
    os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (0,0)

    screen = pygame.display.set_mode((1600,900), pygame.NOFRAME)
    pygame.display.get_wm_info()
    
    
    # Some debug info
    print ("Using %s renderer" % pygame.display.get_driver())
    print ('Playing playlist: %s' % target_playlist)
    
    webcam_thread = webcam_handler.createWebcamMainThread()
    
    # Check if video is accessible
    loadingScreen(0)
    movie = video_handler.get_video(0)
    
    # Create instane of VLC and create reference to video.
    vlcInstance = vlc.Instance()
    media = vlcInstance.media_new(movie)
    
    # Create new instance of vlc player
    player = vlcInstance.media_player_new()
    
    # Add a callback
    em = player.event_manager()
    em.event_attach(vlc.EventType.MediaPlayerTimeChanged, \
        callback, player)
    
    # Pass pygame window id to vlc player, so it can render its contents there.
    win_id = pygame.display.get_wm_info()['window']
    if sys.platform == "linux2": # for Linux using the X Server
        player.set_xwindow(win_id)
    elif sys.platform == "win32": # for Windows
        player.set_hwnd(win_id)
    elif sys.platform == "darwin": # for MacOS
        player.set_agl(win_id)
    
    # Load video into vlc player instance
    player.set_media(media)
    
    # Quit pygame mixer to allow vlc full access to audio device
    pygame.mixer.quit()
    
    # Start video playback
    player.play()
    
    current_video = 0
    
    while inGame():
        if player.get_state() == vlc.State.Ended:
            player.stop()
            current_video += 1
            loadingScreen(current_video)
            
            media = vlcInstance.media_new(video_handler.get_video(current_video))
            player.set_media(media)
            player.play()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(2)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    sys.exit(0)      
            
                    
                
if __name__ == "__main__":
    main(sys.argv[-1:])