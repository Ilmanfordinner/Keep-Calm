import youtube_dl as ytdl
import os

s = input()

ytdl.dl_playlist(s, os.path.dirname(os.path.realpath(__file__)))