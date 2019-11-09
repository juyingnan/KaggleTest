from PIL import Image
import os, sys

path = r"C:\Users\bunny\Desktop\ships-in-satellite-imagery\shipsnet\shipsnet/"
dirs = os.listdir(path)


def resize():
    for item in dirs:
        if os.path.isfile(path + item):
            im = Image.open(path + item)
            imResize = im.resize((224, 224), Image.ANTIALIAS)
            imResize.save(path + item, 'PNG', quality=90)


resize()
