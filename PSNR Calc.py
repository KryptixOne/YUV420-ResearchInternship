import numpy as np
import math
from ntpath import basename
from os.path import getsize
import os 
import pprint

def CalcMSE(OrigY,OrigU,OrigV,ReconY,ReconU,ReconV):
    OrigYRow, OrigYCol, _ =  OrigY.shape
    OrigURow, OrigUCol, _ =  OrigU.shape
    OrigVRow, OrigVCol, _ =  OrigV.shape

    ReconYRow, ReconYCol, _ =  ReconY.shape
    ReconURow, ReconUCol, _ =  ReconU.shape
    ReconVRow, ReconVCol, _ =  ReconV.shape
    
    InnerSumY =0
    InnerSumU=0
    InnerSumV=0
    Innertotal = 0

    for i in range(OrigYRow):
        for j in range(OrigYCol):
            InnerSumY = InnerSumY + ((OrigY[i,j,0]- ReconY[i,j]))**2
    for i in range(OrigURow):
        for j in range(OrigUCol):
            InnerSumU = InnerSumU + ((OrigU[i,j,0]- ReconU[i,j]))**2
    for i in range(OrigURow):
        for j in range(OrigUCol):
            InnerSumV = InnerSumV + ((OrigV[i,j,0]- ReconV[i,j]))**2
    Innertotal = InnerSumY+InnerSumU+InnerSumV

    TotalMSE = (Innertotal * (1/(OrigYRow*OrigYCol +OrigURow*OrigUCol*2)))
    MSEY =(InnerSumY * (1/(OrigYRow*OrigYCol)))
    MSEU =(InnerSumU * (1/(OrigURow*OrigUCol)))
    MSEV =(InnerSumV * (1/(OrigURow*OrigUCol)))
    
    return MSEY,MSEU,MSEV

            

def YUV_Imread (filename):
    #filen1ame should be a path, i.e .\user\name\directory\FILENAME.YUV
    #filename example : BasketballDrill_832x480_50fps_8bit_420.yuv
    YUVFile = basename(filename)
    
    FileMetrics = YUVFile.split('_')
    
    YUVResolution = FileMetrics[1].split('x')
    YUVFPS = FileMetrics[2]
    YUVBit_sign = FileMetrics[3]
    width = int(YUVResolution[0])
    height = int( YUVResolution[1])
    Total = width*height
    FileByteSize = int(getsize(filename))

    if (YUVBit_sign == '8bit'):
        Frames = int( FileByteSize/(width*height*1.5))
        
        read_bit_str = np.uint8
    else:
        Frames = int(FileByteSize/(width*height*1.5)/2)
        read_bit_str = np.uint16

    Y = np.zeros([height,width,Frames])
    U = np.zeros([int(height/2),int(width/2),Frames])
    V = np.zeros([int(height/2),int(width/2),Frames])
        
    with open (filename, 'rb') as fid:
        for i in range(Frames):
            FullFileArray = np.fromfile(fid, read_bit_str)
            #print (FullFileArray)
            counter =0;
            for  j in range(height):
                for k in range(width):
                    Y[j,k,i] = FullFileArray[counter]
                    counter +=1
            for  j in range(int(height/2)):
                for k in range(int(width/2)):
                    U[j,k,i] = FullFileArray[counter]
                    counter +=1
            for  j in range(int(height/2)):
                for k in range(int(width/2)):
                    V[j,k,i] = FullFileArray[counter]
                    counter +=1
                    
    #Y=torch.from_numpy(Y)
    #U=torch.from_numpy(U)
    #V=torch.from_numpy(V)
    return Y, U, V

def CalcPSNR(OrigImgAdd,ReconImageAdd):
    OrigY,OrigU,OrigV = YUV_Imread(OrigImgAdd)
    ReconY,ReconU,ReconV = YUV_Imread(ReconImageAdd)
    MSEY,MSEU,MSEV = CalcMSE(OrigY,OrigU,OrigV,ReconY,ReconU,ReconV)
 
    PSNRY = round(10*math.log10((((2**8)-1)**2)/MSEY),2)
    PSNRU = round(10*math.log10((((2**8)-1)**2)/MSEU),2)
    PSNRV = round(10*math.log10((((2**8)-1)**2)/MSEV),2)
    #print(PSNRY,' ' ,PSNRU,' ' ,PSNRV)
    return [PSNRY,PSNRU,PSNRV]





#Note, Everything below works only when the Reconstructed Image filename has the following:
# "Lambda"+LambdaValue+"OriginalImageName"

lambdalist = [0.1]
FolderNameList = [r'C:\Path\To\Folder\Containing\Reconstructed\BQTerrace',
    r'C:\Path\To\Folder\Containing\Reconstructed\FourPeople',
    r'C:\Path\To\Folder\Containing\Reconstructed\Johnny',
    r'C:\Path\To\Folder\Containing\Reconstructed\KristenAndSara',
    r'C:\Path\To\Folder\Containing\Reconstructed\PartyScene',
    r'C:\Path\To\Folder\Containing\Reconstructed\RaceHorses1',
    r'C:\Path\To\Folder\Containing\Reconstructed\RaceHorses2',
    r'C:\Path\To\Folder\Containing\Reconstructed\SlideEditing',
    r'C:\Path\To\Folder\Containing\Reconstructed\SlideShow',
    r'C:\Path\To\Folder\Containing\Reconstructed\ArenaOfValor',
    r'C:\Path\To\Folder\Containing\Reconstructed\BasketballDrill',
    r'C:\Path\To\Folder\Containing\Reconstructed\BasketballDrillText',
    r'C:\Path\To\Folder\Containing\Reconstructed\BasketballDrive',
    r'C:\Path\To\Folder\Containing\Reconstructed\BasketballPass',
    r'C:\Path\To\Folder\Containing\Reconstructed\BlowingBubbles',
    r'C:\Path\To\Folder\Containing\Reconstructed\BQMall',
    r'C:\Path\To\Folder\Containing\Reconstructed\BQSquare']

#pprint.pprint(FolderNameList)
originalNameList=['BQTerrace_1920x1080_60fps_8bit_420_00.yuv',
                  'FourPeople_1280x720_60fps_8bit_420_00.yuv',
                  'Johnny_1280x720_60fps_8bit_420_00.yuv',
                  'KristenAndSara_1280x720_60fps_8bit_420_00.yuv',
                  'PartyScene_832x480_50fps_8bit_420_00.yuv',
                  'RaceHorses_416x240_30fps_8bit_420_00.yuv',
                  'RaceHorses_832x480_30fps_8bit_420_00.yuv',
                  'SlideEditing_1280x720_30fps_8bit_420_00.yuv',
                  'SlideShow_1280x720_20fps_8bit_420_00.yuv',
                  'ArenaOfValor_1920x1080_60fps_8bit_420_00.yuv',
                  'BasketballDrill_832x480_50fps_8bit_420_00.yuv',
                  'BasketballDrillText_832x480_50fps_8bit_420_00.yuv',
                  'BasketballDrive_1920x1080_50fps_8bit_420_00.yuv',
                  'BasketballPass_416x240_50fps_8bit_420_00.yuv',
                  'BlowingBubbles_416x240_50fps_8bit_420_00.yuv',
                  'BQMall_832x480_60fps_8bit_420_00.yuv',
                  'BQSquare_416x240_60fps_8bit_420_00.yuv']

Index =0

PSNRVALS = {}

for orig in originalNameList  :
    OrigImgAddr= os.path.join(r'C:\Path\To\Folder\Containing\original\Image\\',orig)
    FileMetrics = orig.split('_')
    nameofit = FileMetrics[0]
    #Create the address of the image
    
    #ReconImgAddr0_01 = FolderNameList[Index]+r'\Lambda0.01'+orig
    #ReconImgAddr0_025 = FolderNameList[Index]+r'\Lambda0.025'+orig
    #ReconImgAddr0_03 = FolderNameList[Index]+r'\Lambda0.03'+orig
    #ReconImgAddr0_035 = FolderNameList[Index]+r'\Lambda0.035'+orig
    #ReconImgAddr0_04 = FolderNameList[Index]+r'\Lambda0.04'+orig
    #ReconImgAddr0_05 = FolderNameList[Index]+r'\Lambda0.05'+orig
    ReconImgAddr0_1 = FolderNameList[Index]+r'\Lambda0.1'+orig

    #print(ReconImgAddr0_01)
    #print(orig)

    #PSNR Values to Dictionary
    #PSNRVALS[nameofit +' Lam0.01']= CalcPSNR(OrigImgAddr,ReconImgAddr0_01)
    #PSNRVALS[nameofit + ' Lam0.025']= CalcPSNR(OrigImgAddr,ReconImgAddr0_025)
    #PSNRVALS[nameofit +' Lam0.03']= CalcPSNR(OrigImgAddr,ReconImgAddr0_03)
    #PSNRVALS[nameofit +' Lam0.035']= CalcPSNR(OrigImgAddr,ReconImgAddr0_035)
    #PSNRVALS[nameofit +' Lam0.04']= CalcPSNR(OrigImgAddr,ReconImgAddr0_04)
    #PSNRVALS[nameofit +' Lam0.05']= CalcPSNR(OrigImgAddr,ReconImgAddr0_05)
    PSNRVALS[nameofit +' Lam0.1']= CalcPSNR(OrigImgAddr,ReconImgAddr0_1)

    #print(PSNRVALS)
    Index = Index +1
pprint.pprint(PSNRVALS)
