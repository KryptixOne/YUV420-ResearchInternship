
# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from compressai.transforms.functional import yuv_420_to_444, yuv_444_to_420
from ntpath import basename
from os.path import getsize
import numpy as np
import torch
from math import log2, ceil
class ImageFolder(Dataset):
    """Load an image folder database. Training and testing image samples
    are respectively stored in separate directories:

    .. code-block::

        - rootdir/
            - train/
                - img000.png
                - img001.png
            - test/
                - img000.png
                - img001.png

    Args:
        root (string): root directory of the dataset
        transform (callable, optional): a function or transform that takes in a
            PIL image and returns a transformed version
        split (string): split mode ('train' or 'val')
    """

    def __init__(self, root, transform=None, split="train"):
        splitdir = Path(root) / split

        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')

        self.samples = [f for f in splitdir.iterdir() if f.is_file()]

        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        
        img = Image.open(self.samples[index]).convert("RGB")
        
        if self.transform:
            newone = self.transform(img)
            print(newone.shape)
            return newone
        return img

    def __len__(self):
        return len(self.samples)


##YUV FUNCTIONS*************************************************************
class YUVImageFolder(Dataset):
    """Load an image folder database. Training and testing image samplespython
    are respectively stored in separate directories:

    .. code-block::

        - rootdir/
            - train/
                - img000.png
                - img001.png
            - test/
                - img000.png
                - img001.png

    Args:
        root (string): root directory of the dataset
        transform (callable, optional): a function or transform that takes in a
            PIL image and returns a transformed version
        split (string): split mode ('train' or 'val')
    """

    def __init__(self, root, transform=None, split="train",Training_Type = 1):

        splitdir = Path(root) / split

        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')

        self.samples = [f for f in splitdir.iterdir() if f.is_file()]

        self.transform = transform
        self.TrainingType = Training_Type
        #print('self xform is',self.transform )


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        print('Training Type is: ',self.TrainingType)

        if self.TrainingType == 1 : #training type = seperate paths
            #imread -> 420 to 444 -> RandCrop -> model(RandCrop) -> 444to420 ->  Y,U,V -> Altered_network

            imgY, imgU, imgV = YUV_Imread(self.samples[index]) #get numpyarray img read

            Yt, Ut, Vt = YUV_NumpyToTensor(imgY, imgU, imgV)
            Yt = Yt.unsqueeze(0)
            Ut = Ut.unsqueeze(0)
            Vt = Vt.unsqueeze(0)
            img = yuv_420_to_444((Yt,Ut,Vt)).squeeze(0) #upsampled image to 444
            if self.transform:
                #print('xform')
                Newone = self.transform(img)#.squeeze(0)
                #print('Newone',Newone.shape)
                return Newone

            return img #paddedDim0Flag ,paddedDim1Flag)

        if self.TrainingType == 2: # Pixel Shuffle Creates Quarter images

            imgY, imgU, imgV = YUV_Imread(self.samples[index])
            New_Y = Shuffler(imgY)
            #print('New_Y',New_Y.shape)
            Yt, Ut, Vt = YUV_NumpyToTensor(New_Y, imgU, imgV)
            #print('Yt', Yt.shape)
            Yt = Yt.unsqueeze(0)
            Ut = Ut.unsqueeze(0)
            Vt = Vt.unsqueeze(0)
            #print('Yt_after unsqueeze',Yt.shape)
            img = (torch.cat((Yt, Ut, Vt), dim=1)).squeeze(0)
            #print('img',img.shape)

            if self.transform:
                Newone = self.transform(img)#.squeeze(0)

                return Newone

            return img #paddedDim0Flag ,paddedDim1Flag)

            return

    def __len__(self):
        return len(self.samples)



def DeShuffler(Y0,Y1,Y2,Y3):
    Y0 = Y0.numpy()
    Y1 = Y1.numpy()
    Y2 = Y2.numpy()
    Y3 = Y3.numpy()

    RowSize = Y0.shape[0]
    ColSize = Y0.shape[1]
    NewRow = RowSize*2
    NewCol = ColSize*2

    #print('New Row Size', NewRow)
    #print('New Col Size', NewCol)

    ReconY = np.zeros([NewRow,NewCol,1])
    for i in range(int(NewRow)):
        for j in range(int(NewCol)):
            #print(i,' and ',j)
            if j % 2 == 0 and i % 2 == 0:
                ReconY[i, j,:] = Y0[int(i / 2), int(j / 2),:]
            elif j % 2 == 1 and i % 2 == 0:
                ReconY[i, j,:] = Y1[int(i / 2), int(j / 2),:]
            elif j % 2 == 0 and i % 2 == 1:
                ReconY[i, j,:] = Y2[int(i / 2), int(j / 2),:]
            elif j % 2 == 1 and i % 2 == 1:
                ReconY[i, j,:] = Y3[int(i / 2), int(j / 2),:]
            else:
                continue

    return ReconY



def Shuffler(YComp): #numpy input

    A = YComp
    RowSize = A.shape[0]
    ColSize = A.shape[1]


    A_new = np.zeros([int(RowSize / 2), int(ColSize / 2)])
    B_new = np.copy(A_new)
    C_new = np.copy(A_new)
    D_new = np.copy(A_new)

    # Quarter resolution 1
    for i in range(int(RowSize)):
        for j in range(int(ColSize)):
            if j % 2 == 0 and i % 2 == 0:
                A_new[int(i / 2), int(j / 2)] = A[i, j]
            elif j % 2 == 1 and i % 2 == 0:
                B_new[int(i / 2), int(j / 2)] = A[i, j]
            elif j % 2 == 0 and i % 2 == 1:
                C_new[int(i / 2), int(j / 2)] = A[i, j]
            elif j % 2 == 1 and i % 2 == 1:
                D_new[int(i / 2), int(j / 2)] = A[i, j]
            else:
                continue
    """
    # Quarter resolution 2
    for i in range(int(RowSize)):
        for j in range(int(ColSize)):
            if j % 2 == 1 and i % 2 == 0:
                B_new[int(i / 2), int(j / 2)] = A[i, j]
            else:
                continue
    # Quarter resolution 3
    for i in range(int(RowSize)):
        for j in range(int(ColSize)):
            if j % 2 == 0 and i % 2 == 1:
                C_new[int(i / 2), int(j / 2)] = A[i, j]
            else:
                continue
    # Quarter resolution 4
    for i in range(int(RowSize)):
        for j in range(int(ColSize)):
            if j % 2 == 1 and i % 2 == 1:
                D_new[int(i / 2), int(j / 2)] = A[i, j]
            else:
                continue
    """
    Y_new = np.zeros([int(RowSize / 2), int(ColSize / 2),4])
    Y_new[:, :, 0] = A_new
    Y_new[:, :, 1] = B_new
    Y_new[:, :, 2] = C_new
    Y_new[:, :, 3] = D_new
    return Y_new


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


def YUV_NumpyToTensor(Y,U,V):
    #Y U V need to be unaltered pure range values.
    #i.e Y [16 235] annd U,V [16,240]
    #Y_tensor = torch.from_numpy((Y-16)*(1/219))
    #U_tensor = torch.from_numpy((U-16)*(1/224))
    #V_tensor = torch.from_numpy((V-16)*(1/224))
    Y_tensor = torch.from_numpy(Y * (1 / 255))
    U_tensor = torch.from_numpy(U * (1 / 255))
    V_tensor = torch.from_numpy(V * (1 / 255))
    Y_tensor = Y_tensor.permute(2,0,1)
    U_tensor = U_tensor.permute(2,0,1)
    V_tensor = V_tensor.permute(2,0,1)
    
    return Y_tensor, U_tensor, V_tensor


def imWrite_YUV (Y,U,V,filename,bit_flag):

    #Y, U, V should be torch Tensors of rowXcolXFrames
    #bit_flag should be uint8 or uint16
    #filename should be a fullpath r'C:\Users\Daniel\Desktop\TestFolder\WrittenFile_832x480_50fps_8bit_420_00.yuv'

    if bit_flag =='uint8':
        write_str = np.uint8
    else:
        write_str = np.uint16

    Y=((torch.clamp(Y,min=0, max =1)).numpy())*255
    U=((torch.clamp(U,min=0, max =1)).numpy())*255
    V=((torch.clamp(V,min=0, max =1)).numpy())*255
    
    with open (filename,'wb') as FileToWrite:
            
        np.asarray(Y, dtype=write_str).tofile(FileToWrite)
        np.asarray(U, dtype=write_str).tofile(FileToWrite)
        np.asarray(V, dtype=write_str).tofile(FileToWrite)


def ZeroPad3DNumpyArray(npArray, NZ_Height, NZ_Width):
    dim0 = int(npArray.shape[0])
    dim1 = int(npArray.shape[1])
    dim2 = int(npArray.shape[2])
    NZ_Height= int(NZ_Height)
    NZ_Width=int(NZ_Width)
    #print(dim0,dim1,dim2,NZ_Width,NZ_Height)
    PaddedArray = np.zeros([
                            int(dim0 + 2 * NZ_Height),
                            int(dim1 + 2 * NZ_Width),
                            dim2])
    PaddedArray[NZ_Height:NZ_Height + dim0, NZ_Width: NZ_Width + dim1,:] = npArray

    return PaddedArray

def Num_Zero_Calc (LumArray, U_Array):
    Ydim1 = LumArray.shape[0]
    Ydim2 = LumArray.shape[1]
    Udim1 = U_Array.shape[0]
    Udim2 = U_Array.shape[1]

    NumZerosHeight = (Ydim1 - Udim1) * (1 / 2)
    NumZerosWidth = (Ydim2 - Udim2) * (1 / 2)

    return NumZerosHeight, NumZerosWidth

def PadrForNonSquare(DataTensor,TrainingType,deviceInput): #Quality Test Version
    if TrainingType ==1:
        #Convert back to 420
        img420 = yuv_444_to_420(DataTensor)
        #moves tensor back to (rowxColXchannel format)
        imgY = (((img420[0].squeeze(0)).permute(1,2,0)).cpu()).numpy()
        imgU = (((img420[1].squeeze(0)).permute(1,2,0)).cpu()).numpy()
        imgV = (((img420[2].squeeze(0)).permute(1,2,0)).cpu()).numpy()


        Ydim0 = int(imgY.shape[0])
        print('Yshape', imgY.shape)

        Ydim1 = int(imgY.shape[1])
        Ydim2 = int(imgY.shape[2])
        Udim0 = int(imgU.shape[0])
        Udim1 = int(imgU.shape[1])
        Udim2 = int(imgU.shape[2])

        paddedDim0Flag = 0
        paddedDim1Flag = 0
        paddedDim =3
        padValue = 0.5
        if Ydim0 > Ydim1:  # keep rows same, padd col with zeros
            paddedDim = 1
            if ceil(log2(Ydim0)) == log2(Ydim0):
                Ydim0_new = Ydim0

            else:
                Ydim0_new = 2 ** (ceil(log2(Ydim0)))

            num_dim0_zeros_Y = int((Ydim0_new - Ydim0) * 1 / 2)
            num_dim0_zeros_UV = int(num_dim0_zeros_Y * 1 / 2)
            numYZeros = int((Ydim0_new - Ydim1) * 1 / 2)
            numUVZeros = int(numYZeros * 1 / 2)

            PaddedArrayY = np.zeros([
                int(Ydim0_new),
                int(Ydim1 + 2 * numYZeros),
                Ydim2]) +padValue
            PaddedArrayY[num_dim0_zeros_Y: num_dim0_zeros_Y + Ydim0, numYZeros: numYZeros + Ydim1, :] = imgY
            imgY = PaddedArrayY

            PaddedArrayU = np.zeros([
                int(Ydim0_new / 2),
                int(Udim1 + 2 * numUVZeros),
                Udim2]) +padValue
            PaddedArrayU[num_dim0_zeros_UV:num_dim0_zeros_UV + Udim0, numUVZeros: numUVZeros + Udim1, :] = imgU
            imgU = PaddedArrayU

            PaddedArrayV = np.zeros([
                int(Ydim0_new / 2),
                int(Udim1 + 2 * numUVZeros),
                Udim2]) +padValue
            PaddedArrayV[num_dim0_zeros_UV:num_dim0_zeros_UV + Udim0, numUVZeros: numUVZeros + Udim1, :] = imgV
            imgV = PaddedArrayV

            paddedDim0Flag = num_dim0_zeros_Y
            paddedDim1Flag = numYZeros

            #turn back to tensor and upsample again.


            Y_tensor = torch.from_numpy(imgY)
            U_tensor = torch.from_numpy(imgU)
            V_tensor = torch.from_numpy(imgV)
            Yt = Y_tensor.permute(2, 0, 1)
            Ut = U_tensor.permute(2, 0, 1)
            Vt = V_tensor.permute(2, 0, 1)



            Yt = Yt.unsqueeze(0)
            Ut = Ut.unsqueeze(0)
            Vt = Vt.unsqueeze(0)

            Paddedimg = yuv_420_to_444((Yt, Ut, Vt))  # upsampled padded image back to 444

        if Ydim1 > Ydim0:  # keep col same, pad row with zeros
            paddedDim = 0
            if ceil(log2(Ydim1)) == log2(Ydim1):
                Ydim1_new = Ydim1
            else:
                Ydim1_new = 2 ** (ceil(log2(Ydim1)))
                #print('Ydim1 new', Ydim1)

            num_dim1_zeros_Y = int((Ydim1_new - Ydim1) * 1 / 2)
            num_dim1_zeros_UV = int(num_dim1_zeros_Y * 1 / 2)
            numYZeros = int((Ydim1_new - Ydim0) * 1 / 2)
            numUVZeros = int(numYZeros * 1 / 2)

            PaddedArrayY = np.zeros([
                int(Ydim0 + 2 * numYZeros),
                int(Ydim1_new),
                Ydim2]) +padValue
            PaddedArrayY[numYZeros: numYZeros + Ydim0, num_dim1_zeros_Y: num_dim1_zeros_Y + Ydim1, :] = imgY
            imgY = PaddedArrayY

            PaddedArrayU = np.zeros([
                int(Udim0 + 2 * numUVZeros),
                int(Ydim1_new / 2),
                Udim2]) +padValue
            PaddedArrayU[numUVZeros: numUVZeros + Udim0, num_dim1_zeros_UV:num_dim1_zeros_UV + Udim1, :] = imgU
            imgU = PaddedArrayU

            PaddedArrayV = np.zeros([
                int(Udim0 + 2 * numUVZeros),
                int(Ydim1_new / 2),
                Udim2]) +padValue
            PaddedArrayV[numUVZeros: numUVZeros + Udim0, num_dim1_zeros_UV:num_dim1_zeros_UV + Udim1, :] = imgV
            imgV = PaddedArrayV

            paddedDim0Flag = numYZeros
            paddedDim1Flag = num_dim1_zeros_Y

            Y_tensor = torch.from_numpy(imgY)
            U_tensor = torch.from_numpy(imgU)
            V_tensor = torch.from_numpy(imgV)
            Yt = Y_tensor.permute(2, 0, 1)
            Ut = U_tensor.permute(2, 0, 1)
            Vt = V_tensor.permute(2, 0, 1)

            Yt = Yt.unsqueeze(0)
            Ut = Ut.unsqueeze(0)
            Vt = Vt.unsqueeze(0)

            Paddedimg = yuv_420_to_444((Yt, Ut, Vt))  # upsampled padded image back to 444



        #recombine images and turn back to tensor.
        if deviceInput =='cuda':
            Paddedimg =Paddedimg.cuda()

        return Paddedimg,paddedDim ,paddedDim0Flag, paddedDim1Flag
    elif TrainingType ==2:
        # moves tensor back to (rowxColXchannel format)
        npDataTensor = (((DataTensor.squeeze(0)).permute(1, 2, 0)).cpu()).numpy()
        #print(npDataTensor.shape)

        imgY0= np.expand_dims(npDataTensor[:,:,0],axis = 2)
        imgY1 = np.expand_dims(npDataTensor[:, :, 1],axis = 2)
        imgY2 = np.expand_dims(npDataTensor[:, :, 2],axis = 2)
        imgY3 = np.expand_dims(npDataTensor[:, :, 3],axis = 2)
        imgU = np.expand_dims(npDataTensor [:,:,4],axis = 2)
        imgV = np.expand_dims(npDataTensor[:,:,5],axis = 2)

        Ydim0 = int(imgY0.shape[0])
        print('Yshape', imgY0.shape)


        Ydim1 = int(imgY0.shape[1])
        Ydim2 = int(imgY0.shape[2])
        Udim0 = int(imgU.shape[0])
        Udim1 = int(imgU.shape[1])
        Udim2 = int(imgU.shape[2])

        paddedDim0Flag = 0
        paddedDim1Flag = 0
        paddedDim = 3
        padValue = 0.5
        if Ydim0 > Ydim1:  # keep rows same, padd col with zeros
            paddedDim = 1
            if ceil(log2(Ydim0)) == log2(Ydim0):
                Ydim0_new = Ydim0

            else:
                Ydim0_new = 2 ** (ceil(log2(Ydim0)))

            num_dim0_zeros_Y = int((Ydim0_new - Ydim0) * 1 / 2)
            #num_dim0_zeros_UV = int(num_dim0_zeros_Y * 1 / 2)
            numYZeros = int((Ydim0_new - Ydim1) * 1 / 2)
            #numUVZeros = int(numYZeros * 1 / 2)

            PaddedArrayY0 = np.zeros([
                int(Ydim0_new),
                int(Ydim1 + 2 * numYZeros),
                Ydim2]) + padValue
            PaddedArrayY0[num_dim0_zeros_Y: num_dim0_zeros_Y + Ydim0, numYZeros: numYZeros + Ydim1, :] = imgY0
            imgY0 = PaddedArrayY0

            PaddedArrayY1 = np.zeros([
                int(Ydim0_new),
                int(Ydim1 + 2 * numYZeros),
                Ydim2]) + padValue
            PaddedArrayY1[
            num_dim0_zeros_Y: num_dim0_zeros_Y + Ydim0, numYZeros: numYZeros + Ydim1,:] = imgY1
            imgY1 = PaddedArrayY1

            PaddedArrayY2 = np.zeros([
                int(Ydim0_new),
                int(Ydim1 + 2 * numYZeros),
                Ydim2]) + padValue
            PaddedArrayY2[
            num_dim0_zeros_Y: num_dim0_zeros_Y + Ydim0, numYZeros: numYZeros + Ydim1,:] = imgY2
            imgY2 = PaddedArrayY2

            PaddedArrayY3 = np.zeros([
                int(Ydim0_new),
                int(Ydim1 + 2 * numYZeros),
                Ydim2]) + padValue
            PaddedArrayY3[
            num_dim0_zeros_Y: num_dim0_zeros_Y + Ydim0, numYZeros: numYZeros + Ydim1,:] = imgY3
            imgY3 = PaddedArrayY3

            PaddedArrayU = np.zeros([
                int(Ydim0_new),
                int(Udim1 + 2 * numUVZeros),
                Udim2]) + padValue
            PaddedArrayU[num_dim0_zeros_Y:num_dim0_zeros_Y + Udim0, numYZeros: numYZeros + Udim1, :] = imgU
            imgU = PaddedArrayU

            PaddedArrayV = np.zeros([
                int(Ydim0_new),
                int(Udim1 + 2 * numUVZeros),
                Udim2]) + padValue
            PaddedArrayV[num_dim0_zeros_Y:num_dim0_zeros_Y + Udim0, numYZeros: numYZeros + Udim1, :] = imgV
            imgV = PaddedArrayV

            paddedDim0Flag = num_dim0_zeros_Y
            paddedDim1Flag = numYZeros

            # turn back to tensor

            Y_tensor0 = torch.from_numpy(imgY0)
            Y_tensor1 = torch.from_numpy(imgY1)
            Y_tensor2 = torch.from_numpy(imgY2)
            Y_tensor3 = torch.from_numpy(imgY3)
            U_tensor = torch.from_numpy(imgU)
            V_tensor = torch.from_numpy(imgV)
            Yt0 = Y_tensor0.permute(2, 0, 1)
            Yt1 = Y_tensor1.permute(2, 0, 1)
            Yt2 = Y_tensor2.permute(2, 0, 1)
            Yt3 = Y_tensor3.permute(2, 0, 1)
            Ut = U_tensor.permute(2, 0, 1)
            Vt = V_tensor.permute(2, 0, 1)

            Yt0 = Yt0.unsqueeze(0)
            Yt1 = Yt1.unsqueeze(0)
            Yt2 = Yt2.unsqueeze(0)
            Yt3 = Yt3.unsqueeze(0)
            Ut = Ut.unsqueeze(0)
            Vt = Vt.unsqueeze(0)
            Paddedimg = (torch.cat((Yt0,Yt1,Yt2,Yt3,Ut,Vt), dim=1))



            #Paddedimg = yuv_420_to_444((Yt, Ut, Vt))  # upsampled padded image back to 444

        if Ydim1 > Ydim0:  # keep col same, pad row with zeros
            paddedDim = 0
            if ceil(log2(Ydim1)) == log2(Ydim1):
                Ydim1_new = Ydim1
            else:
                Ydim1_new = 2 ** (ceil(log2(Ydim1)))
                # print('Ydim1 new', Ydim1)

            num_dim1_zeros_Y = int((Ydim1_new - Ydim1) * 1 / 2)
            #num_dim1_zeros_UV = int(num_dim1_zeros_Y * 1 / 2)
            numYZeros = int((Ydim1_new - Ydim0) * 1 / 2)
            #numUVZeros = int(numYZeros * 1 / 2)

            PaddedArrayY0 = np.zeros([
                int(Ydim0 + 2 * numYZeros),
                int(Ydim1_new),
                Ydim2]) + padValue
            PaddedArrayY0[numYZeros: numYZeros + Ydim0, num_dim1_zeros_Y: num_dim1_zeros_Y + Ydim1, :] = imgY0
            imgY0 = PaddedArrayY0

            PaddedArrayY1 = np.zeros([
                int(Ydim0 + 2 * numYZeros),
                int(Ydim1_new),
                Ydim2]) + padValue
            PaddedArrayY1[numYZeros: numYZeros + Ydim0, num_dim1_zeros_Y: num_dim1_zeros_Y + Ydim1, :] = imgY1
            imgY1 = PaddedArrayY1

            PaddedArrayY2 = np.zeros([
                int(Ydim0 + 2 * numYZeros),
                int(Ydim1_new),
                Ydim2]) + padValue
            PaddedArrayY2[numYZeros: numYZeros + Ydim0, num_dim1_zeros_Y: num_dim1_zeros_Y + Ydim1, :] = imgY2
            imgY2 = PaddedArrayY2

            PaddedArrayY3 = np.zeros([
                int(Ydim0 + 2 * numYZeros),
                int(Ydim1_new),
                Ydim2]) + padValue
            PaddedArrayY3[numYZeros: numYZeros + Ydim0, num_dim1_zeros_Y: num_dim1_zeros_Y + Ydim1, :] = imgY3
            imgY3 = PaddedArrayY3

            PaddedArrayU = np.zeros([
                int(Udim0 + 2 * numYZeros),
                int(Ydim1_new ),
                Udim2]) + padValue
            PaddedArrayU[numYZeros: numYZeros + Udim0, num_dim1_zeros_Y:num_dim1_zeros_Y + Udim1, :] = imgU
            imgU = PaddedArrayU

            PaddedArrayV = np.zeros([
                int(Udim0 + 2 * numYZeros),
                int(Ydim1_new ),
                Udim2]) + padValue
            PaddedArrayV[numYZeros: numYZeros + Udim0, num_dim1_zeros_Y:num_dim1_zeros_Y + Udim1, :] = imgV
            imgV = PaddedArrayV

            paddedDim0Flag = numYZeros
            paddedDim1Flag = num_dim1_zeros_Y

            # turn back to tensor

            Y_tensor0 = torch.from_numpy(imgY0)
            Y_tensor1 = torch.from_numpy(imgY1)
            Y_tensor2 = torch.from_numpy(imgY2)
            Y_tensor3 = torch.from_numpy(imgY3)
            U_tensor = torch.from_numpy(imgU)
            V_tensor = torch.from_numpy(imgV)
            Yt0 = Y_tensor0.permute(2, 0, 1)
            Yt1 = Y_tensor1.permute(2, 0, 1)
            Yt2 = Y_tensor2.permute(2, 0, 1)
            Yt3 = Y_tensor3.permute(2, 0, 1)
            Ut = U_tensor.permute(2, 0, 1)
            Vt = V_tensor.permute(2, 0, 1)

            Yt0 = Yt0.unsqueeze(0)
            Yt1 = Yt1.unsqueeze(0)
            Yt2 = Yt2.unsqueeze(0)
            Yt3 = Yt3.unsqueeze(0)
            Ut = Ut.unsqueeze(0)
            Vt = Vt.unsqueeze(0)
            Paddedimg = (torch.cat((Yt0, Yt1, Yt2, Yt3, Ut, Vt), dim=1))

        # recombine images and turn back to tensor.
        if deviceInput =='cuda':
            Paddedimg =Paddedimg.cuda()
        return Paddedimg, paddedDim, paddedDim0Flag, paddedDim1Flag


def DepadTensor(DataTensor, paddedDim,paddedDim0Flag,paddedDim1Flag,TrainingType):
    if TrainingType ==1:
        # Convert back to 420
        img420 = yuv_444_to_420(DataTensor)
        # moves tensor back to (rowxColXchannel format)
        imgY = (((img420[0].squeeze(0)).permute(1, 2, 0)).cpu()).numpy()
        imgU = (((img420[1].squeeze(0)).permute(1, 2, 0)).cpu()).numpy()
        imgV = (((img420[2].squeeze(0)).permute(1, 2, 0)).cpu()).numpy()

        Ydim0 = int(imgY.shape[0])
        Ydim1 = int(imgY.shape[1])
        Ydim2 = int(imgY.shape[2])
        Udim0 = int(imgU.shape[0])
        Udim1 = int(imgU.shape[1])
        Udim2 = int(imgU.shape[2])

        if paddedDim ==3:
            return imgY,imgU,imgV

        elif paddedDim ==1: #means that Col got heavily padded
            numYZeros = paddedDim1Flag
            numUVZeros = int(paddedDim1Flag*1/2)
            num_dim0_zeros_Y = paddedDim0Flag
            num_dim0_zeros_UV = int(num_dim0_zeros_Y *1/2)
            #determine old dimensions
            Ydim1_new = int(Ydim1 - 2*numYZeros)
            Ydim0_new = int(Ydim0 - 2*num_dim0_zeros_Y)
            UVdim1_new = int(Ydim1_new * 1/2)
            UVdim0_new = int(Ydim0_new * 1/2)

            UnPaddedArrayY= np.zeros ([Ydim0_new,Ydim1_new,Ydim2])
            UnPaddedArrayY[:,:,:]= imgY[num_dim0_zeros_Y:Ydim0-num_dim0_zeros_Y,numYZeros: Ydim1- numYZeros,:]

            UnPaddedArrayU = np.zeros([UVdim0_new,UVdim1_new,Udim2])
            UnPaddedArrayU[:, :, :] = imgU[num_dim0_zeros_UV:Udim0 - num_dim0_zeros_UV, numUVZeros: Udim1 - numUVZeros, :]

            UnPaddedArrayV = np.zeros([UVdim0_new, UVdim1_new, Udim2])
            UnPaddedArrayV[:, :, :] = imgV[num_dim0_zeros_UV:Udim0 - num_dim0_zeros_UV, numUVZeros: Udim1 - numUVZeros, :]


        elif paddedDim ==0: # means that Row got heavily padded
            numYZeros = paddedDim0Flag
            numUVZeros = int(paddedDim0Flag * 1 / 2)
            num_dim1_zeros_Y = paddedDim1Flag
            num_dim1_zeros_UV = int(num_dim1_zeros_Y * 1 / 2)

            #determine old dimensions
            Ydim0_new = int(Ydim0 - 2*numYZeros)
            Ydim1_new = int(Ydim1 - 2*num_dim1_zeros_Y)
            UVdim0_new = int(Ydim0_new * 1/2)
            UVdim1_new = int(Ydim1_new * 1/2)

            UnPaddedArrayY= np.zeros ([Ydim0_new,Ydim1_new,Ydim2])
            UnPaddedArrayY[:,:,:]= imgY[numYZeros:Ydim0-numYZeros,num_dim1_zeros_Y: Ydim1- num_dim1_zeros_Y,:]

            UnPaddedArrayU = np.zeros([UVdim0_new,UVdim1_new,Udim2])
            UnPaddedArrayU[:, :, :] = imgU[numUVZeros:Udim0 - numUVZeros, num_dim1_zeros_UV: Udim1 - num_dim1_zeros_UV, :]

            UnPaddedArrayV = np.zeros([UVdim0_new, UVdim1_new, Udim2])
            UnPaddedArrayV[:, :, :] = imgV[numUVZeros:Udim0 - numUVZeros, num_dim1_zeros_UV: Udim1 - num_dim1_zeros_UV, :]

        imgY_new = torch.from_numpy(UnPaddedArrayY)
        imgU_new = torch.from_numpy(UnPaddedArrayU)
        imgV_new = torch.from_numpy(UnPaddedArrayV)

        return imgY_new,imgU_new,imgV_new

    elif TrainingType ==2:
        # Convert back to 420
        npDataTensor = (((DataTensor.squeeze(0)).permute(1, 2, 0)).cpu()).numpy()
        # print(npDataTensor.shape)
        #Row X Col X Channel
        imgY0 = np.expand_dims(npDataTensor[:, :, 0], axis=2)
        imgY1 = np.expand_dims(npDataTensor[:, :, 1], axis=2)
        imgY2 = np.expand_dims(npDataTensor[:, :, 2], axis=2)
        imgY3 = np.expand_dims(npDataTensor[:, :, 3], axis=2)
        imgU = np.expand_dims(npDataTensor[:, :, 4], axis=2)
        imgV = np.expand_dims(npDataTensor[:, :, 5], axis=2)

        Ydim0 = int(imgY0.shape[0])
        Ydim1 = int(imgY0.shape[1])
        Ydim2 = int(imgY0.shape[2])
        Udim0 = int(imgU.shape[0])
        Udim1 = int(imgU.shape[1])
        Udim2 = int(imgU.shape[2])

        if paddedDim == 3:
            return imgY, imgU, imgV

        elif paddedDim == 1:  # means that Col got heavily padded
            numYZeros = paddedDim1Flag
            #numUVZeros = int(paddedDim1Flag * 1 / 2)
            num_dim0_zeros_Y = paddedDim0Flag
            #num_dim0_zeros_UV = int(num_dim0_zeros_Y * 1 / 2)
            # determine old dimensions
            Ydim1_new = int(Ydim1 - 2 * numYZeros)
            Ydim0_new = int(Ydim0 - 2 * num_dim0_zeros_Y)
            #UVdim1_new = int(Ydim1_new * 1 / 2)
            #UVdim0_new = int(Ydim0_new * 1 / 2)

            UnPaddedArrayY0 = np.zeros([Ydim0_new, Ydim1_new, Ydim2])
            UnPaddedArrayY0[:, :, :] = imgY0[num_dim0_zeros_Y:Ydim0 - num_dim0_zeros_Y, numYZeros: Ydim1 - numYZeros, :]

            UnPaddedArrayY1 = np.zeros([Ydim0_new, Ydim1_new, Ydim2])
            UnPaddedArrayY1[:, :, :] = imgY1[num_dim0_zeros_Y:Ydim0 - num_dim0_zeros_Y, numYZeros: Ydim1 - numYZeros, :]

            UnPaddedArrayY2 = np.zeros([Ydim0_new, Ydim1_new, Ydim2])
            UnPaddedArrayY2[:, :, :] = imgY2[num_dim0_zeros_Y:Ydim0 - num_dim0_zeros_Y, numYZeros: Ydim1 - numYZeros, :]

            UnPaddedArrayY3 = np.zeros([Ydim0_new, Ydim1_new, Ydim2])
            UnPaddedArrayY3[:, :, :] = imgY3[num_dim0_zeros_Y:Ydim0 - num_dim0_zeros_Y, numYZeros: Ydim1 - numYZeros, :]

            UnPaddedArrayU = np.zeros([UVdim0_new, UVdim1_new, Udim2])
            UnPaddedArrayU[:, :, :] = imgU[num_dim0_zeros_Y:Udim0 - num_dim0_zeros_Y, numYZeros: Udim1 - numYZeros,
                                      :]

            UnPaddedArrayV = np.zeros([UVdim0_new, UVdim1_new, Udim2])
            UnPaddedArrayV[:, :, :] = imgV[num_dim0_zeros_Y:Udim0 - num_dim0_zeros_Y, numYZeros: Udim1 - numYZeros,
                                      :]


        elif paddedDim == 0:  # means that Row got heavily padded
            numYZeros = paddedDim0Flag
            numUVZeros = int(paddedDim0Flag * 1 / 2)
            num_dim1_zeros_Y = paddedDim1Flag
            num_dim1_zeros_UV = int(num_dim1_zeros_Y * 1 / 2)

            # determine old dimensions
            Ydim0_new = int(Ydim0 - 2 * numYZeros)
            Ydim1_new = int(Ydim1 - 2 * num_dim1_zeros_Y)
            UVdim0_new = int(Ydim0_new * 1 / 2)
            UVdim1_new = int(Ydim1_new * 1 / 2)

            UnPaddedArrayY0 = np.zeros([Ydim0_new, Ydim1_new, Ydim2])
            UnPaddedArrayY0[:, :, :] = imgY0[numYZeros:Ydim0 - numYZeros, num_dim1_zeros_Y: Ydim1 - num_dim1_zeros_Y, :]

            UnPaddedArrayY1 = np.zeros([Ydim0_new, Ydim1_new, Ydim2])
            UnPaddedArrayY1[:, :, :] = imgY1[numYZeros:Ydim0 - numYZeros, num_dim1_zeros_Y: Ydim1 - num_dim1_zeros_Y, :]

            UnPaddedArrayY2 = np.zeros([Ydim0_new, Ydim1_new, Ydim2])
            UnPaddedArrayY2[:, :, :] = imgY2[numYZeros:Ydim0 - numYZeros, num_dim1_zeros_Y: Ydim1 - num_dim1_zeros_Y, :]

            UnPaddedArrayY3 = np.zeros([Ydim0_new, Ydim1_new, Ydim2])
            UnPaddedArrayY3[:, :, :] = imgY3[numYZeros:Ydim0 - numYZeros, num_dim1_zeros_Y: Ydim1 - num_dim1_zeros_Y, :]

            UnPaddedArrayU = np.zeros([Ydim0_new, Ydim1_new, Udim2])
            UnPaddedArrayU[:, :, :] = imgU[numYZeros:Udim0 - numYZeros, num_dim1_zeros_Y: Udim1 - num_dim1_zeros_Y,
                                      :]

            UnPaddedArrayV = np.zeros([Ydim0_new, Ydim1_new, Udim2])
            UnPaddedArrayV[:, :, :] = imgV[numYZeros:Udim0 - numYZeros, num_dim1_zeros_Y: Udim1 - num_dim1_zeros_Y,
                                      :]

        imgY0_new = torch.from_numpy(UnPaddedArrayY0)
        imgY1_new = torch.from_numpy(UnPaddedArrayY1)
        imgY2_new = torch.from_numpy(UnPaddedArrayY2)
        imgY3_new = torch.from_numpy(UnPaddedArrayY3)
        imgU_new = torch.from_numpy(UnPaddedArrayU)
        imgV_new = torch.from_numpy(UnPaddedArrayV)

        imgY_new = DeShuffler(imgY0_new,imgY1_new,imgY2_new,imgY3_new)
        imgY_new = torch.from_numpy(imgY_new)
        #print(imgY_new.shape)

        return imgY_new, imgU_new, imgV_new

