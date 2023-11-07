import cv2
import numpy as np
import math
from PIL import Image

lena = Image.open("lena.bmp")
lena_ld = lena.load()
n,m = lena.size[0],lena.size[1]

index1 = (1,2)
index2 = (2,0)

out = np.zeros((lena.size[0] , lena.size[1]), np.uint8)
for i in range(n): # for every pixel:
    for j in range(m):
        p = lena_ld[i,j]
        out[j][i]=p

def extract_8_block(img,row, column):
    block = np.zeros((8,8), np.uint8)
    for i in range(8):
        if i+row >= n:
            break
        for j in range(8):
            if j+column >= m:
                break
            block[i][j] = img[i+row,j+column]
    return block

def DCT(block):
    assert len(block) == 8 and len(block[0]) == 8
    Block = block

    DCT_BLOCK = np.zeros((8,8))
    for u in range(8):
        for v in range(8):
            for x in range(8):
                for y in range(8):
                    DCT_BLOCK[u][v] += Block[x][y] * math.cos((2*x+1)*u*math.pi/16) * math.cos((2*y+1)*v*math.pi/16)

            if u == 0:
                DCT_BLOCK[u][v] /= math.sqrt(2)

            if v == 0:
                DCT_BLOCK[u][v] /= math.sqrt(2)

            DCT_BLOCK[u][v] /= 4
    return DCT_BLOCK

def inverseDCT(DCT_BLOCK):

    BLOCK = np.zeros((8,8))

    for x in range(8):
        for y in range(8):
            for u in range(8):
                for v in range(8):

                    if u == 0:
                        Cu = 1/math.sqrt(2)
                    else:
                        Cu = 1

                    if v == 0:
                        Cv = 1/math.sqrt(2)
                    else:
                        Cv = 1

                    BLOCK[x][y] += Cu * Cv * DCT_BLOCK[u][v] * math.cos((2*x+1)*u*math.pi/16) * math.cos((2*y+1)*v*math.pi/16)
            BLOCK[x][y] = round(BLOCK[x][y]/4)

    return BLOCK

def char2bin_len8(char):
    return ("00000000" + bin(ord(char))[2:])[-8:]

def QuantizationHiding(block, bit):
    matrix = np.array([[16, 11, 10, 16, 24, 40, 51, 61],[12, 12, 14, 19, 26, 58, 60, 55],[14, 13, 16, 24, 40, 57, 69, 56],[14, 17, 22, 29, 51, 87, 80, 62],[18, 22, 37, 56, 68, 109, 103, 77],[24, 35, 55, 64, 81, 104, 113, 92],[49, 64, 78, 87, 103, 121, 120, 101],[72, 92, 95, 98, 112, 100, 103, 99]])


    Q = np.zeros((8,8),dtype=np.int32)
    for i in range(8):
        for j in range(8):
            Q[i][j] = round(block[i][j]/matrix[i][j])

    if int(bit) == 0:
        if Q[index1] >= Q[index2]:
            if Q[index1] == Q[index2]:
                Q[index2]+=3
            Q[index1], Q[index2] = Q[index2], Q[index1]

    else:
        if Q[index1] < Q[index2]:
            # if Q[index1] == Q[index2]:
                # Q[index2]+=3
            Q[index1], Q[index2] = Q[index2], Q[index1]
    return Q ,int(bit)

def Quantization(block):
    matrix = np.array([[16,11,10,16,24,40,51,61],
                    [12,12,14,19,26,58,60,55],
                    [14,13,16,24,40,57,69,56],
                    [14,17,22,29,51,87,80,62],
                    [18,22,37,56,68,109,103,77],
                    [24,35,55,64,81,104,113,92],
                    [49,64,78,87,103,121,120,101],
                    [72,92,95,98,112,100,103,99]])  
    Q = np.zeros((8,8),dtype=np.int32)
    for i in range(8):
        for j in range(8):
            Q[i][j] = round(block[i][j]/matrix[i][j])

    if Q[index1]==0 and Q[index2]==0:
        return Q , -1

    return Q ,1




def Extract(Q):

    if Q[index1]==0 and Q[index2]==0:
        return -1



    bit = "0"
    if Q[index1] >= Q[index2]:
        bit = "1"
    return  bit

def deQuantizationExtract(Q):

    matrix = np.array([[16,11,10,16,24,40,51,61],
                    [12,12,14,19,26,58,60,55],
                    [14,13,16,24,40,57,69,56],
                    [14,17,22,29,51,87,80,62],
                    [18,22,37,56,68,109,103,77],
                    [24,35,55,64,81,104,113,92],
                    [49,64,78,87,103,121,120,101],
                    [72,92,95,98,112,100,103,99]])   

    block = np.zeros((8,8),dtype=np.int32)
    block += Q*matrix

    if block[index2]==0 and block[index1]==0:
        return block , -1
    bit = "0"
    if block[index1] >= block[index2]:
        bit = "1"
    return block, bit

def ReplaceBlock(block, row, column):
    # n, m = image.shape
    for i in range(8):
        if i+row >= n :
            break
        for j in range(8):
            if j+column >= m:
                break
            out[i+row][j+column] = block[i][j]
    return out 

                
            
            # A[l]+=1

def HidingText(image, Texte):

    n,m = image.size[0],image.size[1]
    image = image.load()
    Texte = ''.join(format(ord(x), 'b') for x in Texte)
    print(Texte)
    # Text_index = 0
    # Bin_index = 0

    # char2bin = char2bin_len8(Texte[Text_index])


    Hiding = np.copy(image)
    print(Hiding)

    cnt=0
    for i in range(0,8*(n//8),8):
        for j in range(0,8*(m//8),8):
            Block = extract_8_block(Hiding, i, j)
            # if Bin_index == 8:
            #     Bin_index = 0
            #     Text_index += 1
            #     if Text_index == len(Texte):
            #         return Hiding
            #     char2bin = char2bin_len8(Texte[Text_index])


            Block -= 128*np.ones((8,8), dtype = np.uint)
            DCT_BLOCK = DCT(Block)
            Q = QuantizationHiding(DCT_BLOCK, Texte[cnt])
            Block = deQuantizationExtract(Q)[0]
            Block = inverseDCT(Block)
            Block += 128*np.ones((8,8), dtype = np.uint)
            Hiding=ReplaceBlock(Hiding, Block, i, j)
            cnt+=1
            if (cnt==len(Texte)):
                return Hiding
    return Hiding

def GetText(Hiding):

    n,m = image.size[0],image.size[1]
    Text = ""
    char2bin = ""
    for i in range(0,8*n//8,8):
        for j in range(0,8*m//8,8):
            Block = extract_8_block(Hiding, i, j)
            Block -= 128*np.ones((8,8), dtype = np.uint)
            DCT_BLOCK = DCT(Block)
            # Q = Quantization(DCT_BLOCK)
            char2bin = char2bin + Extract(DCT_BLOCK)
            if len(char2bin) == 8:
                Text = Text + chr(int(char2bin,2))
                char2bin = ""
    return Text


if __name__ == '__main__':

    ########## Hinding ######################################################################
    Text="Hello"
    Text=''.join(format(ord(x), 'b') for x in Text)
    cnt=0
    flag=0
    inn = ""
    for i in range(4,n,8):
        if flag :
            break
        for j in range(10,m,8):
            block = extract_8_block(lena_ld , i ,j)
            block -= 128*np.ones((8,8),dtype=np.uint8)
            cff = DCT(block)
            if Quantization(cff)[1]==-1:
                continue
            g=QuantizationHiding(cff  , Text[cnt])[0]
            r ,oo =deQuantizationExtract(g)
            inn += oo
            cff2=inverseDCT(r)
            block2 = cff2+ 128*np.ones((8,8),dtype=np.uint8)
            ReplaceBlock(block2,i,j)
            cnt+=1
            if cnt == len(Text):
                flag=1
                break

    print(inn)
    pil_image=Image.fromarray(out)
    im1 = pil_image.save("dctout.bmp")
    # print(Quantization(DCT(extract_8_block(lena_ld,4,9) - 128*np.ones((8,8),dtype=np.uint8))))

    #
    # ########## Extracting ######################################################################
    ans=""
    source = Image.open("dctout.bmp")
    n,m = source.size[0],source.size[1]
    source = source.load()
    # print('\n\n',extract_8_block(source,0,7))
    print('\n')
    # print(Quantization(DCT(extract_8_block(source,4,9) - 128*np.ones((8,8),dtype=np.uint8))) )
    for i in range(4,n,8):
        for j in range(10,m,8):
            block = extract_8_block(source , i ,j)
            block -= 128*np.ones((8,8),dtype=np.uint8)
            cff = DCT(block)
            if Extract(cff) == -1:
                continue
            a=Extract(cff)
            # g = Quantization(cff)
            # if deQuantizationExtract(g)[1] == -1 :
            #     continue
            # a=deQuantizationExtract(g)[1]
            ans+=a
    print(ans)
    print(''.join([chr(int(ans[i:i+8] , 2)) for i in range(0,len(ans),8)]))

    # print(128*np.ones((8,8),dtype=np.int32) + inverseDCT(deQuantizationExtract(Quantization(DCT(extract_8_block(lena_ld,0,10)) - 128*np.ones((8,8),dtype=np.int32)))[0]))
