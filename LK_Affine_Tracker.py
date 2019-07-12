import glob
import cv2 as cv
import numpy as np


def get_coordinates_array(x_range,y_range):
    a=(x_range[1]-x_range[0]) + 1
    b=(y_range[1]-y_range[0]) + 1
    coordinates=np.zeros((3,a*b))
    count=0
    for y in range(y_range[0],y_range[1]+1):
        for x in range(x_range[0],x_range[1]+1):
            coordinates[0,count]=x
            coordinates[1,count]=y
            coordinates[2,count]=1
            count+=1            
    return coordinates

def point_selector(T_x_image):
    points = cv.selectROI(T_x_image)
    
    x_start = points[0]
    x_end = points[0]+points[2]
    y_start = points[1]
    y_end = points[1]+ points[3]
    x_range=[x_start,x_end]
    y_range=[y_start,y_end]
    
    return x_range,y_range

def get_new_coord(T_x_coordinates,p,x_range,y_range):
    x1,x2=x_range
    y1,y2=y_range
    vertex_array=np.array([[x1,x1,x2,x2],[y1,y2,y2,y1],[1,1,1,1]])

    affine_mat =np.zeros((2,3))
    count =0
    for i in range(3):
        for j in range(2):
            affine_mat[j,i]= p[count,0] 
            count =count+1 
    affine_mat[0,0]+=1
    affine_mat[1,1]+=1
    new_vertex_array=np.dot(affine_mat,vertex_array)
    new_coordinates=np.dot(affine_mat,T_x_coordinates)
    new_coordinates = new_coordinates.astype(int)
    return new_coordinates,new_vertex_array

def get_pixel_array(image,coordinates):
    img_array = np.zeros((1,coordinates.shape[1]))
    img_array[0,:]=image[coordinates[1,:],coordinates[0,:]]  
    return img_array

def compute_error(img_array1,img_array2):
    error=img_array1-img_array2
    return error   

def get_T_x_array(Template_image, x_range, y_range):
    T_x_coordinates=get_coordinates_array(x_range,y_range)
    p_T_x=np.array([[0,0,0,0,0,0]]).T
    new_coordinates,new_ver=get_new_coord(T_x_coordinates,p_T_x,x_range,y_range)
    T_x_img_arr = get_pixel_array(Template_image,new_coordinates)
    return T_x_img_arr

def convert_lab(image):
   clahe = cv.createCLAHE(clipLimit=1., tileGridSize=(1,1))
   lab = cv.cvtColor(image, cv.COLOR_BGR2LAB)
   l, a, b = cv.split(lab)  
   l2 = clahe.apply(b)  
   lab = cv.merge((l,a,l2))  
   img2 = cv.cvtColor(lab, cv.COLOR_LAB2BGR)
   return img2

    
def array_to_img(arrayimage,x_range,y_range):
    a=(x_range[1]-x_range[0]) + 1
    b=(y_range[1]-y_range[0]) + 1
    img=arrayimage.astype(np.uint8)
    image=np.reshape(img,(b,a))
    return image 

def sobel(img): 
    # sobelx = cv.Scharr(img,cv.CV_64F,1,0)   
    # sobely = cv.Scharr(img,cv.CV_64F,0,1)
    sobelx = cv.Sobel(img,cv.CV_64F,1,0,ksize=3)   
    sobely = cv.Sobel(img,cv.CV_64F,0,1,ksize=3)
    return sobelx,sobely

def jacobian(x,y):
    mat=np.array(([x,0,y,0,1,0],[0,x,0,y,0,1]))  
    return mat

def get_delta_p(err_array,steep_descent):
    SDParam = np.dot(steep_descent.T, err_array.T)
    Hessian = np.dot(steep_descent.T, steep_descent)
    Hessian_Inv = np.linalg.pinv(Hessian)
    delta_p = np.dot(Hessian_Inv, SDParam)
    return delta_p

def update_p (p,delta_p):
    p = np.reshape(p,(6,1))
    p = p+delta_p
    return p 

def sanitycheck(coorinate_array, img) :
    min_xy=np.amin(coorinate_array,axis=1)
    max_xy=np.amax(coorinate_array,axis=1)
    if min_xy[0] < 0 or max_xy[0] >= img.shape[1] or min_xy[1] < 0 or max_xy[1] >= img.shape[0]:
        T = False
    else :
        T = True
    return T
    

def draw_rectangle(image,pts):    
    
    cv.polylines(image,  np.int32([pts.T]),  1,  (0, 255, 0),  2)   
    return image

def get_norm(del_p):
    return np.linalg.norm(del_p)
   
def get_steep_descent(sobelx,sobely,new_coordinates,old_coordinates):
    sobel_delx_array=get_pixel_array(sobelx,new_coordinates)
    sobel_dely_array=get_pixel_array(sobely,new_coordinates)
    img1=sobel_delx_array*old_coordinates[0,:]
    img2=sobel_dely_array*old_coordinates[0,:]
    img3=sobel_delx_array*old_coordinates[1,:]
    img4=sobel_dely_array*old_coordinates[1,:]
    steepest_descent_image=np.vstack((img1,img2,img3,img4, sobel_delx_array, sobel_dely_array)).T
    return steepest_descent_image


    

def affineLKtracker(T_x_coordinates,T_x_array,gray_image,x_range, y_range, p, sobelx, sobely):
    new_img_coordinates,new_vertex_array=get_new_coord(T_x_coordinates,p,x_range,y_range)
    if(sanitycheck(new_img_coordinates,gray_image)):
        sanity =True
        img_arr = get_pixel_array(gray_image,new_img_coordinates)
        error_array = compute_error(T_x_array,img_arr)  
        steep_descent=get_steep_descent(sobelx,sobely,new_img_coordinates,T_x_coordinates)
        del_p=get_delta_p(error_array,steep_descent)
        p_norm = get_norm(del_p)
        p = update_p(p,del_p)
    else:
        sanity =False
        del_p = np.array([[0,0,0,0,0,0]]).T
        p_norm = get_norm(del_p)
    return p, del_p, p_norm,new_img_coordinates,new_vertex_array,sanity


# Starting
print("\n\nPlease enter 0 for Car, 1 for human, 2 for vase : ")
choice = int(input())
#threshold  = 0.01
if choice ==0 :
    path = "data/car/*.jpg"
    path1 = "output/car/car"
    threshold  = 0.03
if choice ==1 :
     path = "data/human/*.jpg"
     path1 = "output/human/human"
     threshold  = 0.05
if choice ==2 :
     path = "data/vase/*.jpg"
     path1 = "output/vase/vase"
     threshold  = 0.005

   
filenames = [img for img in glob.glob(path)]
filenames.sort() # ADD THIS LINE  
T_x_image=cv.imread(filenames[0])  
T_x_image=convert_lab(T_x_image)
T_x_image=cv.cvtColor(T_x_image,cv.COLOR_BGR2GRAY)
t_x_mean=np.mean(T_x_image)

if choice==0 :
    print("Enter 0 for best result, enter 1 to choose your own ROI ")
    ch = int(input())
    if ch==0:
        x_range=[136,338]        
        y_range=[108,275]
    elif ch==1:
        x_range,y_range=point_selector(T_x_image) 
        
if choice==1 :
    print("Enter 0 for best result, enter 1 to choose your own ROI ")
    ch = int(input())
    if ch==0:
        x_range=[264,283]
        y_range=[292,360]
    elif ch==1:
        x_range,y_range=point_selector(T_x_image)   

if choice==2 :
    print("Enter 0 for best result, enter 1 to choose your own ROI ")
    ch = int(input())
    if ch==0:
        x_range=[127,172]
        y_range=[92,147]
    elif ch==1:
        x_range,y_range=point_selector(T_x_image) 

print(x_range,y_range) 
#x_range=[264,283]   ### for Human best result
#y_range=[292,360]   ### for Human best result

T_x_array = get_T_x_array(T_x_image,x_range,y_range)
T_x_coordinates=get_coordinates_array(x_range,y_range)

p=np.array([[0,0,0,0,0,0]]).T
count = 0

for img in filenames:
    
#    print(count)        
    image=cv.imread(img)
    image=convert_lab(image)
    cv.waitKey(1)
    gray=cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    i_x_mean=np.mean(gray)
    gray=(gray*((t_x_mean/i_x_mean))).astype(float)
    sobelx,sobely = sobel(gray)
    count = count+1
    
    while True:
        p, del_p, p_norm,new_img_coordinates,new_vertex,sane= affineLKtracker(T_x_coordinates,T_x_array,gray,x_range,y_range,p,sobelx,sobely)     
        if p_norm <= threshold or sane == False :
            break

    rect_img = draw_rectangle(image, new_vertex)
    cv.imshow('rect',rect_img)
    cv.imwrite(path1+'%s.jpg'%count,rect_img)    
    
    
cv.destroyAllWindows()    
