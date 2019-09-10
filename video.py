import pygame
import sys
import time
import numpy as np
import cv2
import math
import os
import argparse
from pygame.locals import *

defaultFont = "Courier"
defaultSize = 12
defaultOutput = "output.mp4"
defaultChars = " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~"
defaultTempFile = "temp.mp4"
defaultChunkSize = 1000

parser = argparse.ArgumentParser(description="Turn an image into ASCII art")
parser.add_argument("video", help="video file to use", type=str)
parser.add_argument("-f", "--font", help="path for font to use; if not specified, the system's {} font is used".format(defaultFont), type=str, default=None)
parser.add_argument("-s", "--size", help="font size to use; default is {}".format(defaultSize), type=int, default=defaultSize)
parser.add_argument("-o", "--output", help="path of where to store the output video; default is '{}'".format(defaultOutput), type=str, default=defaultOutput)
parser.add_argument("-d", "--dictionary", help="list of characters to use; default is all printable ASCII chars", type=str, default=defaultChars)
parser.add_argument("-c", "--color", help="draw the image in color", action="store_true")
parser.add_argument("-i", "--frame_invisible", help="do not display each frame as it renders; this is much faster", action="store_true")
parser.add_argument("-v", "--video_invisible", help="do not display each frame after it has rendered", action="store_true")
parser.add_argument("-a", "--audio", help="copy audio from input video", action="store_true")
parser.add_argument("-t", "--temp", help="temporary file; default is {}".format(defaultTempFile), type=str, default=defaultTempFile)
parser.add_argument("-F", "--frame_count", help="number of frames to render; default (-1) is all of them", type=int, default=-1)
parser.add_argument("-n", "--chunk_size", help="number of frames per chunk; default is {}".format(defaultChunkSize), type=int, default=defaultChunkSize)

args = parser.parse_args()

video = args.video
filename = args.font
size = args.size
output = args.output
chars = list(args.dictionary)
color = args.color
invisible = args.frame_invisible
videoInvisible = args.video_invisible
copyAudio = args.audio
temp = args.temp
maxFrameCount = args.frame_count
chunkSize = args.chunk_size

pygame.init()

screen = pygame.display.set_mode((1, 1), 0, 32)

pygame.display.set_caption("ASCII ART")

font = None
if filename is None:
    font = pygame.font.SysFont(defaultFont, size)
else:
    font = pygame.font.Font(filename, size)


blockSize = -1

surfaces = [font.render(char, False, (0, 0, 0), (255, 255, 255)) for char in chars]
grids = [pygame.surfarray.array2d(i).transpose()*255 for i in surfaces]
for i in range(len(grids)):
    top = -1
    bottom = -1
    left = -1
    right = -1
    height, width = grids[i].shape
    for j, row in enumerate(grids[i]):
        if np.sum(row) > 0:
            top = j
            break
    if top < 0:
        top = 0
        bottom = 1
        left = 0
        right = 1
    else:
        for j, row in enumerate(np.flip(grids[i], 0)):
            if np.sum(row) > 0:
                bottom = height-j
                break
        for j, col in enumerate(grids[i].transpose()):
            if np.sum(col) > 0:
                left = j
                break
        for j, col in enumerate(np.flip(grids[i].transpose(), 0)):
            if np.sum(col) > 0:
                right = width-j
                break
    grids[i] = grids[i][top:bottom, left:right]
    surfaces[i] = surfaces[i].subsurface((left, top, right-left, bottom-top))
    if right-left > blockSize:
        blockSize = right-left
    if bottom-top > blockSize:
        blockSize = bottom-top
        
for i, label in enumerate(surfaces):
    surface = pygame.Surface((blockSize, blockSize))
    surface.fill((255, 255, 255))
    surface.blit(label, ((surface.get_width()-label.get_width())/2,
                         (surface.get_height()-label.get_height())/2))
    surfaces[i] = surface
    grids[i] = pygame.surfarray.array2d(surface).transpose()//16777215*255

cap = cv2.VideoCapture(video)
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
length = length if length < maxFrameCount or maxFrameCount < 0 else maxFrameCount
fps = cap.get(cv2.CAP_PROP_FPS)
oldHeight, oldWidth = cap.read()[1].shape[:2]
height = oldHeight//blockSize * blockSize
width = oldWidth//blockSize * blockSize

heightDiff = oldHeight-height
widthDiff = oldWidth-width
cap.release()
chunkCount = math.ceil(length/chunkSize)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(temp if copyAudio else output, fourcc, fps, (width, height))
for chunk in range(chunkCount):
    
    cap = cv2.VideoCapture(video)
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
        break
    frames = []
    colorFrames = []
    frameCount = 0
    while cap.isOpened() and (frameCount < maxFrameCount or maxFrameCount < 0) and len(frames) < chunkSize:
        ret, frame = cap.read()
        frameCount += 1
        if frameCount <= chunk*chunkSize:
            continue
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            if color:
                colorFrames.append(pygame.image.frombuffer(frame.tostring(), frame.shape[1::-1], "RGB"))
        else:
            break
    cap.release()
    if len(frames) == 0:
        break

    frameBlocks = []

    for i, frame in enumerate(frames):
        if heightDiff != 0:
            frames[i] = frame[heightDiff//2:-math.ceil(heightDiff/2), :]

        if widthDiff != 0:
            frames[i] = frames[i][:, widthDiff//2:-math.ceil(widthDiff/2)]


        blocks = (frames[i].reshape(height//blockSize, blockSize, -1, blockSize)
                  .swapaxes(1,2)
                  .reshape(height//blockSize, width//blockSize, blockSize, blockSize))

        frameBlocks.append(blocks)

    if invisible:
        screen = pygame.Surface((width, height))
    else:
        screen = pygame.display.set_mode((width, height), 0, 32)

    
    lastBlocks = None
    for index, blocks in enumerate(frameBlocks):
        print("Chunk [{}/{}], Frame [{}/{}], [{}/{}] seconds".format(chunk+1, chunkCount, index+1, len(frameBlocks), round((chunk*chunkSize+index+1)/fps, 2), round(length/fps, 2)))
        done = False
        for y, row in enumerate(blocks):
            for x, block in enumerate(row):
                if index > 0 and np.array_equal(block, lastBlocks[y][x]):
                    continue
                closest = -1
                minDist = -1
                for event in pygame.event.get():
                    if event.type == QUIT:
                        done = True
                        break
                if done:
                    break
                for i, grid in enumerate(grids):
                    dist = np.sum(np.absolute(block-grid))
                    if closest < 0 or dist < minDist:
                        minDist = dist
                        closest = i
                if color:
                    colorToUse = pygame.transform.average_color(colorFrames[index], (x*blockSize+widthDiff//2, y*blockSize+heightDiff//2, blockSize, blockSize))
                    array = pygame.PixelArray(surfaces[closest].copy())
                    array.replace((0, 0, 0), colorToUse)
                    surface = array.make_surface()
                    array.close()
                    screen.blit(surface, (x*blockSize, y*blockSize))
                else:
                    screen.blit(surfaces[closest], (x*blockSize, y*blockSize))
                if not invisible:
                    pygame.display.update()
            if done:
                break
        frame = pygame.surfarray.array3d(screen)
        frame = cv2.transpose(frame)
        out.write(frame)
        if not videoInvisible:
            cv2.imshow("Frame", frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                done = True
                break
        
        

        lastBlocks = blocks
        if done:
            break
    if done:
        break
    

cap.release()
out.release()
cv2.destroyAllWindows()

if copyAudio:
    os.system("ffmpeg -i {temp} -i {input} -c copy -map 0:v:0 -map 1:a:0 -shortest {output} -y".format(temp=temp, output=output, input=video))
    os.remove(temp)

pygame.quit()
