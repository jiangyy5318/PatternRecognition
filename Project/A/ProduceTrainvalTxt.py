 # -*- coding: UTF-8 -*-
import os
import sys
import getopt

def FromDirToTxt(Folderdir,txtdir):

    file = open(txtdir, 'w')
    for parent,dirs,filenames in os.walk(Folderdir):
        for dirname in filenames:
            list=os.path.split(parent)[-1]
            file.write(list + "/"+  dirname)
            file.write('\t'+list[1])
            file.write('\n')
    file.close()
    print Folderdir + ' has been done'



if __name__ == '__main__':
    #get pwd
    CCC = os.getcwd()

    #get argument
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hi:o:", ["help","input=" ,"output="])
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
        print value
    A = os.path.join(CCC,A)
    B = os.path.join(CCC,B)

    #Transfer .txt
    FromDirToTxt(str(A),str(B))