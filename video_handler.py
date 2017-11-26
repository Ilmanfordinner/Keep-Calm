import sys
import requests
import bs4
from pytube import YouTube
from pprint import pprint
import os
import subprocess
import urllib
import ffmpy
import threading
import logging
import random

_loaded_video_count = 0
_current_playlist_videos = []

# Downloads an entire YouTube playlist from URL
def load_playlist(url):
    
    global _loaded_video_count
    global _current_playlist_videos
    
    
    downloaded_videos = get_video_names('videos\\')
    
    _loaded_video_count = 0
    
    
    _current_playlist_videos = get_playlist_links(url)
    print(_current_playlist_videos)
    
    for vid in _current_playlist_videos:
        if getId(vid) in downloaded_videos:
            _loaded_video_count += 1
    
    for link in _current_playlist_videos:
        if not getId(link) in downloaded_videos:
            download_video(link, 'videos\\')
            _loaded_video_count += 1
            
# Utility function for the other file. Gets the number of available videos.
def getLoadedVideoCount():
    return _loaded_video_count

# Get a random video
def get_video(id):
    if id >= _loaded_video_count:
        raise Exception("Not enough videos")
    
    downloaded_videos = get_video_names('videos\\')
    current_downloaded_videos = []
    for vid in _current_playlist_videos:
        if getId(vid) in downloaded_videos:
            current_downloaded_videos.append(getId(vid))
    
    return ('videos\\' + current_downloaded_videos[random.randrange(0, _loaded_video_count)] + ".mp4")

# Get a list of all the video filenames
def get_video_names(dir):
    names = []
    
    for root, directories, files in os.walk(dir):
        for f in files:
            #If it finds an .mp3 or .mp4 file it saves it without the extension
            if f.endswith('.mp4'):
                names.append(f[:len(f)-4])
    return names

# Gets all YouTube video links from a playlist link
def get_playlist_links(url):
    website = requests.get(url)
    soup = bs4.BeautifulSoup(website.text, 'html.parser')
    
    items = soup.find_all('a', class_='pl-video-title-link')
    
    links = []
    for i in items:
        l = str('https://www.youtube.com' + i.get('href')[:i.get('href').index('&')])
        if l not in links:
            links.append(l)
    return links

# Gets the ID of a Youtube video (=AAAAAAAAAAA)
def getId(link):
    return link[-12:]

#Downloads a video at 720p or less
def download_video(url, dir):
    for i in range(100):
        try:
            yt = YouTube(url)
            video = yt.streams.filter(subtype='mp4').filter(res='720p').first()
            if video is None:
                video = yt.streams.filter(subtype='mp4').filter(res='480p').first()
                if video is None:
                    video = yt.streams.filter(subtype='mp4').filter(res='360p').first()
                    if video is None:
                        video = yt.streams.filter(subtype='mp4').filter(res='240p').first()
                        if video is None:
                            video = yt.streams.filter(subtype='mp4').filter(res='144p').first()
                            if video is None:
                                raise Exception("Succ")
            break
        except urllib.error.URLError:
            print('URLError. Trying again... (try {} of {})'.format(i+1, 100))
    
    #If current song is not already downloaded, attempt to download it
    fname = str(getId(url) + '.mp4')
    video.download(dir)
    os.rename((dir+video.default_filename), (dir+fname))

# Creates a thread for downloading the videos
def create_playlist_loading_thread(playlist_url):
    t = threading.Thread(target = load_playlist, args = (playlist_url, ))
    t.setDaemon(True)
    t.start()
    
    return t