import sys
import requests
import bs4
from pytube import YouTube
from pprint import pprint
import os
import subprocess
import urllib

def get_playlist_videos(url):
	'''Returns list of links to videos in playlist'''
	website = requests.get(url)
	#soup = bs4.BeautifulSoup(website.text, 'lxml')
	soup = bs4.BeautifulSoup(website.text, 'html.parser')

	items = soup.find_all('a', class_='pl-video-title-link')
	
	links = []
	for i in items:
		l = str('https://www.youtube.com' + i.get('href')[:i.get('href').index('&')])
		if l not in links:
			links.append(l)
	return links

def dl_playlist(url, dl_folder):
	songs_in_folder = get_songs_in_folder(dl_folder)

	links = get_playlist_videos(url)

	for l in links:
		print('On item ({} of {})'.format(links.index(l)+1, len(links)))
		dl_video(l, dl_folder, songs_in_folder)

def dl_video(url, dl_folder, songs_in_folder=None):
	if songs_in_folder is None:
		songs_in_folder = get_songs_in_folder(dl_folder)

	print('Downloading: {}'.format(url))
	# Loop because urllib sometimes does not work properly
	for i in range(100):
		try:
			yt = YouTube(url)
			video = yt.streams.filter(subtype='mp4').first()
			break
		except urllib.error.URLError:
			print('URLError. Trying again... (try {} of {})'.format(i+1, 100))
	
	#If current song is not already downloaded, attempt to download it
	if yt.title not in songs_in_folder:
		video.download(dl_folder)
	else:
		print('Already downloaded!')

def get_songs_in_folder(folder):
	songs = []

	#Goes through the folder in which the sript is located
	for root, directories, files in os.walk(folder):
		for f in files:
			#If it finds an .mp3 or .mp4 file it saves it without the extension
			if f.endswith('.mp3') or f.endswith('.mp4'):
				songs.append(f[:len(f)-4])
	return songs


def convert_to_mp3(dl_folder):
	'''DOES NOT WORK'''
	#Checks if there is a converter in the folder
	filepath = dl_folder + '/convert.bat'
	print(filepath)
	if os.path.isfile(filepath):
		os.system(filepath)
	else:
		print('Could not convert to mp3! Missing convert.bat\n')
