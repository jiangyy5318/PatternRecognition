 # -*- coding: UTF-8 -*-
import os
import sys
import getopt
from PIL import Image
import PIL

def ResizePicture(path):
    old = Image.open(path,'r')
    new_img = old.resize((256,256),Image.BILINEAR)
    new_img.save(path)

if __name__ == '__main__':
    CCC = os.getcwd()
    #get argument
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hi:", ["help","input="])
    except getopt.GetoptError:
        sys.exit()

    #resolute argument
    for op, value in opts:
        if op == '-i' or op == '--input':
                A = value
        elif op == '-o' or op == '--output':
                B = value
        elif op == '-h' or op == '--help':
                sys.exit()
        #print value
    A = os.path.join(CCC,A)
    #B = os.path.join(CCC,B)

    #for parent,dirnames,filenames in os.walk(str(A)):
    #    for dirname in filenames:

    #        list=os.path.split(parent)[-1]
    #        print list + "/"+  dirname+ 'has been resized'
    #        ResizePicture(list + "/"+  dirname,)

    list_dirs = os.walk(str(A))
    for root, dirs, files in list_dirs:
        #for d in dirs:
        #    print os.path.join(root, d)
        for f in files:
            print str(os.path.join(root, f))+' has been resized'
            ResizePicture(str(os.path.join(root, f)))

            #print os.path.join(root, f)

            #file.write(list + "/"+  dirname)


