import pygame
import sys
import time
import numpy as np
import cv2
import math
import argparse
from pygame.locals import *

defaultFont = "Courier"
defaultSize = 12
defaultOutput = "output.png"
defaultChars = " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~"

parser = argparse.ArgumentParser(description="Turn an image into ASCII art")
parser.add_argument("image", help="image file to use", type=str)
parser.add_argument("-f", "--font", help="path for font to use; if not specified, the system's {} font is used".format(defaultFont), type=str, default=None)
parser.add_argument("-s", "--size", help="font size to use; default is {}".format(defaultSize), type=int, default=defaultSize)
parser.add_argument("-o", "--output", help="path of where to store the output image; default is '{}'".format(defaultOutput), type=str, default=defaultOutput)
parser.add_argument("-d", "--dictionary", help="list of characters to use; default is all printable ASCII chars", type=str, default=defaultChars)
parser.add_argument("-c", "--color", help="draw the image in color", action="store_true")
parser.add_argument("-i", "--invisible", help="do not display the image as it renders; this is much faster", action="store_true")

args = parser.parse_args()

image = args.image
filename = args.font
size = args.size
output = args.output
chars = list(args.dictionary)
color = args.color
invisible = args.invisible

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

colorImage = None
if color:
    colorImage = pygame.image.load(image).convert()
image = cv2.imread(image, 0)

oldHeight, oldWidth = image.shape
height = oldHeight//blockSize * blockSize
width = oldWidth//blockSize * blockSize

heightDiff = oldHeight-height
widthDiff = oldWidth-width


if heightDiff != 0:
    image = image[heightDiff//2:-math.ceil(heightDiff/2), :]

if widthDiff != 0:
    image = image[:, widthDiff//2:-math.ceil(widthDiff/2)]

blocks = (image.reshape(height//blockSize, blockSize, -1, blockSize)
          .swapaxes(1,2)
          .reshape(height//blockSize, width//blockSize, blockSize, blockSize))

if invisible:
    screen = pygame.Surface((width, height))
else:
    screen = pygame.display.set_mode((width, height), 0, 32)
screen.fill((255, 255, 255))
done = False
for y, row in enumerate(blocks):
    for x, block in enumerate(row):
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
            colorToUse = pygame.transform.average_color(colorImage, (x*blockSize+widthDiff//2, y*blockSize+heightDiff//2, blockSize, blockSize))
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

pygame.image.save(screen, output)

pygame.quit()
