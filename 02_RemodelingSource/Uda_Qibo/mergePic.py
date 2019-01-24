#!/usr/bin/env python3
#Merge 16 sheets of 320x480 picture into 1216x1936 picture
#2019/01/24 Ver1.0 by M.Y

from glob import glob
import scipy.misc
import numpy
import os.path

def mergePic(inPATH_Pic,outPATH_Pic,type_Pic):
    #Get Pic set
    PATH_img_full=[]
    PATH_img_base=[]
    PATH_img=[]
    PATH_img_full = glob(inPATH_Pic + '/*.'+type_Pic)

    
    #Get basename of Pic
    for _ in PATH_img_full:
        if _.rsplit("_",1)[0] not in PATH_img_base:
            PATH_img_base.append(_.rsplit("_",1)[0])

    #Set Vertical and Horizontal(I am lazy to let it smart...)
    Vertical_PART=[320,618,917,1216]   
    Horizontal_PART=[488,968,1448,1928] 
    side=8

    '''Begin to merge pictures'''
    for PATH_img_base_one in PATH_img_base:
        i=0
        img_save=numpy.ndarray(shape=(1216,1936,3))

        for Vertical_PART_one in Vertical_PART:
            for Horizontal_PART_one in Horizontal_PART:
                i+=1
                filename=PATH_img_base_one+"_"+str(i)+"."+type_Pic

                if os.path.isfile(filename):
                    img=scipy.misc.imread(filename)
                else:
                    exit("Can not found: "+filename+". Check it.")

                #Check Pic size and resize Pic
                if img.shape!=(320,480,3):
                    print("The size of ["+filename+"] is wrong.Check it.")
                    exit()
                else:
                    img_save[Vertical_PART_one-320:Vertical_PART_one,Horizontal_PART_one-480:Horizontal_PART_one]=img
        
        for side_one in range(side):
            img_save[0:1216,side:side]=img_save[0:1216,8:8]
            img_save[0:1216,1936-side:1936-side]=img_save[0:1216,1928:1928]


        scipy.misc.imsave(outPATH_Pic+os.path.basename(PATH_img_base_one)+"."+type_Pic,img_save)




if __name__ == '__main__':

    inPATH_Pic=r"E:\Github\AI_Edge_Contest\02_RemodelingSource\Uda_Qibo\runs\1548311931.3680956"
    outPATH_Pic=r"E:\Github\AI_Edge_Contest\02_RemodelingSource\Uda_Qibo\runs\1548311931.3680956\output/"
    type_Pic="jpg"

    mergePic(inPATH_Pic,outPATH_Pic,type_Pic)