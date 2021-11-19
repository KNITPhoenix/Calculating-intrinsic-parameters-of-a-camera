###############
##Design the function "calibrate" to  return 
# (1) intrinsic_params: should be a list with four elements: [f_x, f_y, o_x, o_y], where f_x and f_y is focal length, o_x and o_y is offset;
# (2) is_constant: should be bool data type. False if the intrinsic parameters differed from world coordinates. 
#                                            True if the intrinsic parameters are invariable.
#It is ok to add other functions if you need
###############
import numpy as np
from cv2 import imread, cvtColor, COLOR_BGR2GRAY, TERM_CRITERIA_EPS, TERM_CRITERIA_MAX_ITER, \
    findChessboardCorners, cornerSubPix, drawChessboardCorners

criteria = (TERM_CRITERIA_EPS + TERM_CRITERIA_MAX_ITER, 30, 0.001)

def calibrate(imgname):
    worldcoor=[]   #array for real world from right side of image to left, top to bottom to top
    k=1
    for i in range(1,5):
        for j in range(4,0,-1):
            worldcoor.append([j*10,0,i*10])
        worldcoor.append([0, 0, k * 10])
        for j in range(1,5):
            worldcoor.append([0,j*10,i*10])
        k=k+1

    ima1=imread(imgname)
    ima=cvtColor(ima1,COLOR_BGR2GRAY)
    r,imgcoor= findChessboardCorners(ima,(9,4),None)

    if r==True:
        corners2=cornerSubPix(ima,imgcoor,(9,4),(-1,-1),criteria)
        imgcoor=corners2
        img=drawChessboardCorners(ima1,(9,4),imgcoor,r)
        a3=[]
        imgcoor=imgcoor.reshape(-1,2).tolist()
        for i in range(len(worldcoor)):
            a1 = [worldcoor[i][0], worldcoor[i][1], worldcoor[i][2], 1, 0, 0, 0, 0, (-imgcoor[i][0] * worldcoor[i][0]), (-imgcoor[i][0] * worldcoor[i][1]), (-imgcoor[i][0] * worldcoor[i][2]), -imgcoor[i][0]]
            a2 = [0, 0, 0, 0,worldcoor[i][0], worldcoor[i][1], worldcoor[i][2], 1, (-imgcoor[i][1] * worldcoor[i][0]), (-imgcoor[i][1] * worldcoor[i][1]), (-imgcoor[i][1] * worldcoor[i][2]), -imgcoor[i][1]]
            a3.append(a1)
            a3.append(a2)

        a3=np.matrix(a3)  #principle matrix of statement 2
        u,s,vt=np.linalg.svd(a3,full_matrices=True)    #here vt is v transpose
        vt=vt.tolist()

        x=vt[len(vt)-1]

        x=[[x[0],x[1],x[2],x[3]],[x[4],x[5],x[6],x[7]],[x[8],x[9],x[10],x[11]]]    #here we get x, now if we calculate lambda we can multiply it to get m, uptil here everything is correct
        r3=np.matrix([x[2][0],x[2][1],x[2][2]])
        lda=1/(np.sqrt(np.matmul(r3,r3.T))).item([0][0]) #value of lambda
        x=np.matrix(x)
        m=lda*x    #m matrix
        m=m.tolist()
        m1 = np.matrix([[m[0][0],m[0][1],m[0][2]]]).T
        m2 = np.matrix([m[1][0], m[1][1], m[1][2]]).T
        m3 = np.matrix([m[2][0], m[2][1], m[2][2]]).T
        m4 = np.matrix([m[0][3], m[1][3], m[2][3]]).T

        o_x=np.matmul(m1.T,m3).item([0][0])
        o_y=np.matmul(m2.T,m3).item([0][0])
        f_x=np.sqrt(((np.matmul(m1.T,m1))-(o_x**2)))
        f_y=np.sqrt(((np.matmul(m2.T,m2))-(o_y**2)))

        return [f_x,f_y,o_x,o_y],False


if __name__ == "__main__":
    intrinsic_params, is_constant = calibrate('checkboard.png')
    print(intrinsic_params)
    print(is_constant)