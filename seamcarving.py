import sys
import cv2
import numpy as np
img = cv2.imread("dali.jpg")
MAXIMO = 100000000

def sob(img):
    vertical = cv2.Sobel(img,cv2.CV_32F,0,1);
    horizontal = cv2.Sobel(img,cv2.CV_32F,1,0);
    final = cv2.sqrt(cv2.pow(vertical,2)+cv2.pow(horizontal,2))
    return cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)


'''
a=np.array([1,2,3,4,5,6,7,8,9,10])
def deletar_np(arr,a,b):
    print(arr[0:a])
    print(arr[a+1:])
deletar_np(a,1,0)
'''
def energy_map(img):
    
    direcao = [[0 for i in range(len(img[0]))] for i in range(len(img))]
    dp = [[MAXIMO for i in range(len(img[0]))] for i in range(len(img))]
    for i in range(len(img[0])):
        dp[0][i]=img[0][i]
    for i in range(1,len(img)):
        for j in range(len(img[0])):
            opt3=dp[i-1][j]
            
            if (j-1>=0):
                opt1=dp[i-1][j-1]
            else:
                opt1= MAXIMO

            if (j+1<len(img[0])):
                opt2 = dp[i-1][j+1]
            else:
                opt2=MAXIMO

            menor = min(opt1,opt2,opt3);
            if (menor==opt1):
                direcao[i][j]=-1
            elif (menor==opt2):
                direcao[i][j]=1
            else:
                direcao[i][j]=0
                
            dp[i][j]=img[i][j]+menor
    
    return dp,direcao

def deletar(arr,i):
    #print(len(arr))
    lnovo=[]
    for j in range(len(arr)):
        if (j!=i):
            lnovo.append(arr[j])
    #print(len(lnovo))
    return np.array(lnovo)

def remove_min_energy(img,img_to_remove):
    dp,direcao=energy_map(img)
    menor=MAXIMO
    melhorindex=0
    for i in range(len(img[0])):
        if (menor>dp[len(img)-1][i]):
            menor=dp[len(img)-1][i]
            melhorindex=i
    y=len(img)-1
    x=melhorindex
    img_new=[[] for i in range(y+1)]
    while(y>=0):
        #print(deletar(img_to_remove[y],x))
        img_new[y]=(deletar(img_to_remove[y],x))
        if (direcao[y][x]==1):
            x+=1
        elif (direcao[y][x]==-1):
            x-=1
        y-=1
    return(np.array(img_new))

def seam(img,n):
    for i in range(n):
        print(i)
        img1=sob(img)
        img=remove_min_energy(img1,img)
    cv2.imwrite("seamed.jpg",img)
cv2.imwrite("teste.jpg",img)
print("c")
seam(img,200)
print("e")


#seam(img,20)


