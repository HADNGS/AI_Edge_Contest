#!/usr/bin/env python3
#Split 1216x1936 picture into 16 sheets of 320x480 picture
#2019/01/23 Ver1.0 by M.Y

from glob import glob
import scipy.misc
import numpy
import os.path

def splitPic(inPATH_Pic,outPATH_Pic):
    #Get Pic set
    PATH_img=[]
    PATH_png = glob(inPATH_Pic + '/*.png')
    PATH_jpg = glob(inPATH_Pic + '/*.jpg')

    PATH_img=PATH_png
    if PATH_jpg :
        for PATH_jpg_one in PATH_jpg:
            PATH_img.append(PATH_jpg_one)

    #Set Vertical and Horizontal(I am lazy to let it smart...)
    Vertical_PART=[320,618,917,1216]
    Horizontal_PART=[480,960,1440,1920]

    '''Begin to split pictures'''
    for PATH_img_one in PATH_img:
        img=scipy.misc.imread(PATH_img_one) 
        filename=os.path.basename(PATH_img_one)
        filename=filename.split(".")

        #Check Pic size and resize Pic
        if img.shape!=(1216,1936,3):
            print("the size of ["+PATH_img_one+"] is wrong.Check it.")
            exit()
        else:
            #Cut horizontal: 1936>>1920
            img = numpy.delete(img, numpy.s_[1928:1936:1], axis=1)
            img = numpy.delete(img, numpy.s_[:8:1], axis=1)
            
            i=1
            for Vertical_PART_one in Vertical_PART:
                for Horizontal_PART_one in Horizontal_PART:
                    img_save=img[Vertical_PART_one-320:Vertical_PART_one,Horizontal_PART_one-480:Horizontal_PART_one]

                    filename_save=outPATH_Pic+filename[0]+"_"+str(i)+"."+filename[1]
                    scipy.misc.imsave(filename_save,img_save)
                    i+=1    


if __name__ == '__main__':

    inPATH_Pic=r"D:\Github_Project\_DATA\seg_train_annotations"
    outPATH_Pic=r"D:\Github_Project\AI_Edge_Contest\02_RemodelingSource\Uda_Qibo\data\seg_annotations_train/"

    splitPic(inPATH_Pic,outPATH_Pic)