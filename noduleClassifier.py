#!/usr/bin/env python
# coding: utf-8
# Python file that was generates from a Jupyter notebook.
# In[1]:


import os
import dicom
import pickle
import fnmatch
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import scipy.ndimage as nd
import skimage.feature as ft

from skimage import data
from skimage import exposure
from skimage import data, img_as_float
from skimage.feature import hog
from skimage.segmentation import clear_border
from skimage.measure import label,regionprops, perimeter
from skimage.morphology import ball, octagon, disk, dilation, binary_erosion, remove_small_objects 
from skimage.morphology import erosion, closing, reconstruction, binary_closing
from skimage.filters import roberts, sobel
from scipy import ndimage as ndi
from sklearn.preprocessing import normalize


# In[2]:


def vote_sys(pos , vote_threshold):
    '''
    This function applies a voting scheme to assign a malignant label (1) or 
    a benign label (0) to an image, which depends on the vote_threshold. A
    vote_threshold = 1 requires a majority vote to be classified as malignant.
    Subsequently a vote_threshold = 0 only requires a split vote to be classified
    as malignant.
    '''
    #print(pos)
    # Extract the columns for the malignant counts and benign counts.
    non_nodule_check = pos[:,1:3]
    #print(non_nodule_check)
    
    # Sum the malignant and benign counts for each image slice.
    non_nodule_check = np.sum(non_nodule_check, axis = 1)
    #print(non_nodule_check)
    
    # Generates a list of indices for images that have malignant and/or benign counts.
    nodule_inds = [ind for ind,val in enumerate(non_nodule_check) if val > 1 and val < 5]
    #print(nodule_inds)
    
    # The rows of the pos matrix that have indices with malignant and/or benign counts
    # is returned, which eliminates images that in the xml files that correspond to 
    # non-nodules or nodules less than 3mm.
    pos = pos[nodule_inds,:]
    #print(pos)
    
    # Converts the benign counts to negative values.
    pos[:,2] = -1*pos[:,2]
    #print(pos)
    
    # Sums the malignant and benign counts to obtain a net voting value.
    pos[:,1] = pos[:,1] + pos[:,2]
    #print(pos[:,1])
    
    # converts column 2 values to zero and eliminates column 2.
    pos[:,2] = 0*pos[:,2]
    pos = np.delete(pos , (2), axis = 1)
    #print(pos)
    # Applies vote_threshold to assign labels.
    for c in range(len(pos)):
        if pos[c,1] >= vote_threshold:
            pos[c,1] = 1
        else:
            pos[c,1] = 0
    
    
    return pos


# In[3]:


def folder_check(dirname):
    '''
    This function tests the files in the path dirname to determine if the
    files are CT or x-ray images. If the file contains CT images then a
    decision_val = 1 is returned, if the file contains x-ray images then
    a decision_val = -1
    '''
    
    # Initialize decision_val.
    decision_val = -1
    
    # Check if folder contains CT or x-ray images by using modality tag.
    for filename in os.listdir(dirname):
        dash_check = fnmatch.fnmatch(filename, '._*')
        
        if ".dcm" in filename.lower() and dash_check == False:
            scan_check = dicom.read_file( dirname + "/" + filename )
            
            if scan_check.Modality == 'CT':
                decision_val = 1
                #print('ct')
                
                return decision_val
                
            if scan_check.Modality == 'DX':
                decision_val = -1
                #print('dx')
                
                return decision_val
            
            if scan_check.Modality == 'CR':
                decision_val = -1
                #print('cr')
                
                return decision_val
                
            if scan_check.Modality == 'CXR':
                decision_val = -1
                #print('cxr')
                
                return decision_val
    
    return decision_val


# In[4]:


def z_check(dirname , pos):
    '''
    This function determines if any of the z-positions from the annotations
    correspond to the z-positions in the slices.
    '''
    skip_val = -1
    
    # Sort files numerically in a list.
    file_ls = os.listdir(dirname)
    file_ls.sort()
    
    # If folder contains CT images then all slices are loaded into a list.
    slices = [ dicom.read_file(
        dirname + "/" + filename ) for filename in file_ls if ".dcm" in filename.lower() and fnmatch.fnmatch(filename,'._*') == False ]
    
    # Convert slices to a list of z positions.
    z_list = [slices[ele].SliceLocation for ele in range(len(slices))]
    z_list = [float(f) for f in z_list]
    
    # Dimensions of the pos matrix.
    row , col = pos.shape

    # Iterate throught pos matrix and determine if the 
    # z positions correspond to the positions in z_list.
    for i in range(row):
        z_pos = pos[i,0]
        
        if z_pos in z_list:
            skip_val = 1
    
    
    return skip_val



# In[5]:


def ct_scan_slices(dirname , pos):
    '''
    This function sorts the DICOM files by numerical label, then reads and organizes the
    files as a list called slices. The image in slices that correspond to the image z
    positions in the pos matrix are extracted and processed. The function outputs the 
    concatenated images and a label vector.
    '''
    # Save data directories.
    #img_prefix = '/Volumes/SEAGATE2/CS680_project/cs680_np_imgs14/'
    #img_prefix = '/Volumes/SEAGATE2/CS680_project/cs680_cnn/'
    matrix_prefix = '/Volumes/SEAGATE2/CS680_project/cs680_matrices14/'
    label_prefix = '/Volumes/SEAGATE2/CS680_project/cs680_labels14/'
    
    global img_iter
    global matrix_iter
    global label_iter
    
    # Sort files numerically in a list.
    file_ls = os.listdir(dirname)
    file_ls.sort()
    
    # If folder contains CT images then all slices are loaded into a list.
    slices = [ dicom.read_file(
        dirname + "/" + filename ) for filename in file_ls if ".dcm" in filename.lower() and fnmatch.fnmatch(filename,'._*') == False ]
    
    # Sort the slices by instance number
    #slices.sort(key = lambda x: int(x.SliceLocation))
    #print(slices)
    
    #z_list = [slices[ele].SliceLocation for ele in range(len(slices))]
    #z_list = [float(f) for f in z_list]
    
    # Extract the DICOM images in slices that correspond to the images of interest specified by the 
    # pos matrix.
    row , col = pos.shape
    
    # Initialize empty array for images.
    imgs = np.zeros((32,32))
    
    # Iterate through pos matrix and slices.
    for i in range(row):
        z_pos = pos[i,0]
        
        for j in range(len(slices)):
            
            if z_pos == slices[j].SliceLocation: 
                # Convert image to numpy array and zeros out of bound elements.
                im = slices[j].pixel_array
                im[im == -1024] = 0
                '''
                img_iter += 1
                img_num = format(img_iter,'06d')
                np.save(img_prefix + 'img' + img_num + '.npy',im)
                '''
                # Calculated the image boundaries to reshape the images to
                # a dimension of 64x64.
                #'''
                x_left = int( pos[i,6] - 16 )
                x_right = int( pos[i,6] + 16 )
                y_up = int( pos[i,7] - 16 )
                y_down = int( pos[i,7] + 16 )
                
                # Segment the image using the function seg.
                im = seg(im , y_up , y_down , x_left , x_right , plot = False)
                
                # Resize image based on the nodules obtained from the pos matrix
                # such that the image is 64x64 and concatenate to imgs.
                im = im[y_up:y_down , x_left:x_right]
                #img_iter += 1
                #img_num = format(img_iter,'06d')
                #np.save(img_prefix + 'img' + img_num + '.npy',im)
                #'''
                '''
                matrix_iter += 1
                array_num = format(matrix_iter,'06d')
                np.save(matrix_prefix + 'array' + array_num + '.npy',im)
                '''
                '''
                if pos[i,1] == 1 and im[16,16] == 0:
                    im = np.zeros((32,32))
                    pos[i,1] = 0
                '''
                    
                    
                imgs = np.dstack((imgs,im))
                
        
    # Delete the initial zero matrix of imgs.
    imgs = np.delete(imgs , (0) , axis = 2)
    
    # Reshape the label column of pos into a nx1 vector.
    labels = np.reshape(pos[:,1] , (len(pos[:,1]),1))
    '''
    label_iter += 1
    label_num = format(label_iter,'06d')
    np.save(label_prefix + 'label' + label_num + '.npy',labels)
    '''
    
    return imgs , labels



# In[6]:


def seg(im, y_up , y_down , x_left , x_right , plot = False):
    '''
    This funtion segments the lungs from the given 2D slice.
    ''' 
    
    if plot == True:
        f, plots = plt.subplots(5, 2, figsize=(8, 20))
        
    if plot == True:
        plots[0,0].axis('off')
        plots[0,0].imshow(im,cmap=plt.cm.bone)
    '''
    Step 1: Convert into a binary image. 
    '''
    binary = im < 604
    if plot == True:
        plots[0,1].axis('on')
        plots[0,1].imshow(binary, cmap=plt.cm.bone) 
    '''
    Step 2: Remove the blobs connected to the border of the image.
    '''
    cleared = clear_border(binary)
    if plot == True:
        plots[1,0].axis('off')
        plots[1,0].imshow(cleared, cmap=plt.cm.bone) 
    '''
    Step 3: Label the image.
    '''
    label_image = label(cleared)
    if plot == True:
        plots[1,1].axis('on')
        plots[1,1].imshow(label_image, cmap=plt.cm.bone) 
    '''
    Step 4: Keep the labels with 2 largest areas.
    '''
    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    if len(areas) > 2:
        for region in regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:                
                       label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0
    if plot == True:
        plots[2,0].axis('on')
        plots[2,0].imshow(binary, cmap=plt.cm.bone) 
    '''
    Step 5: Erosion operation with a disk of radius 2. This operation is 
    seperate the lung nodules attached to the blood vessels.
    '''
    selem = disk(2)
    binary = binary_erosion(binary, selem)
    if plot == True:
        plots[2,1].axis('on')
        plots[2,1].imshow(binary, cmap=plt.cm.bone) 
    '''
    Step 6: Closure operation with a disk of radius 10. This operation is 
    to keep nodules attached to the lung wall.
    '''
    selem = disk(10)
    binary = binary_closing(binary, selem)
    if plot == True:
        plots[3,0].axis('on')
        plots[3,0].imshow(binary, cmap=plt.cm.bone) 
    '''
    Step 7: Fill in the small holes inside the binary mask of lungs.
    '''
    edges = roberts(binary)
    binary = ndi.binary_fill_holes(edges)
    if plot == True:
        plots[3,1].axis('on')
        plots[3,1].imshow(binary, cmap=plt.cm.bone) 
    '''
    Step 8: Superimpose the binary mask on the input image.
    '''
    get_high_vals = binary == 0
    im[get_high_vals] = 0
    if plot == True:
        plots[4,0].axis('on')
        plots[4,0].imshow(im, cmap=plt.cm.bone) 
        
    im[im < 604] = 0
    if plot == True:
        plots[4,1].axis('on')
        plots[4,1].imshow(im[y_up:y_down , x_left:x_right])
        
    #plt.savefig('seg.png',bbox_inches = 'tight')

        
    return im


# In[7]:


def local_binary_pattern(im , r):
    '''
    This function takes as input an image, im and a neighborhood radius, r and outputs a 1-D
    array consisting of LBP values for all pixels in the image that have nonzero intensity 
    and more 5 neighbors with nonzero intensity. These conditions are applied to ignore 
    non-nodule pixels egdge nodule pixels.
    '''
    
    # Row and column size of image.
    row , col = im.shape
    
    img = np.zeros((row,col))
    
    counter = 0
    lbp_vals = []
    for i in np.arange(1,row-1,1):
        
        for j in np.arange(1,col-1,1):
            
            if im[i,j] == 0:
                continue
            else:
                neighbors = np.array( [im[i-r,j] , im[i-r,j+r] , im[i,j+r] , im[i+r,j+r] ,
                                       im[i+r,j] , im[i+r,j-r] , im[i,j-r] , im[i-r,j-r] ] )
                #print(neighbors)
                #print(np.count_nonzero(neighbors))
                if np.count_nonzero(neighbors) > 5:
                    counter += 1
                    p_0 = 1 if im[i-r,j] >= im[i,j] else 0
                    p_1 = 1 if im[i-r,j+r] >= im[i,j] else 0
                    p_2 = 1 if im[i,j+r] >= im[i,j] else 0
                    p_3 = 1 if im[i+r,j+r] >= im[i,j] else 0
                    p_4 = 1 if im[i+r,j] >= im[i,j] else 0
                    p_5 = 1 if im[i+r,j-r] >= im[i,j] else 0
                    p_6 = 1 if im[i,j-r] >= im[i,j] else 0
                    p_7 = 1 if im[i-r,j-r] >= im[i,j] else 0
                    #binary = ''.join([element_8 , element_7 , element_6 , element_5,
                              #element_4 , element_3 , element_2 , element_1])
                    #im[i,j] = int(binary,2)
                    lbp = (p_0*(2**0) + p_1*(2**1) + p_2*(2**2) + p_3*(2**3) +
                           p_4*(2**4) + p_5*(2**5) + p_6*(2**6) + p_7*(2**7) )
                    lbp_vals.append(lbp)
                    #h = np.hstack([h , lbp]) if h.size else lbp
                    img[i,j] = lbp
                    #print(lbp)
                    
    # Array of bins for a greyscale image.
    bins = np.arange(0,256,1)
    
    # Array of lbp values for im.
    lbp_vals = np.array(lbp_vals)
    
    # Histogram given given by lbp_vals and bins.
    hist = np.histogram(lbp_vals , bins)
    
    # Feature of lbp values.
    lbp_features = hist[0]
    
    # Reshape lbp_features into a nx1 vector.
    lbp_features = np.reshape(lbp_features , (1,len(lbp_features)))
    
    
    return lbp_features


# In[8]:


def HoG(image):
    '''
    This function take an image as input and generates a 1_D array consisting of the values for
    the histogram of oriented gradients.
    '''

    hog_features = hog(image , orientations = 8, pixels_per_cell=(4, 4) , cells_per_block=(1, 1) , 
                       visualize = False , feature_vector = True , multichannel = False)
    
    hog_features = np.reshape(hog_features , (1,len(hog_features)))
    
    
    return hog_features 


# In[9]:


def xml_read(input_folder , mb_grade):
    '''
    This function navigates the folders to extract and read the appropriate
    xml files. The imageZpositions, malignancy assessments, and image position
    information is obtained and compiled in the pos matrix, which is given as
    the out of the function.
    
    pos matrix structure:
    
    | imageZposition | Malignant Count | Benign Count | x_min | x_max | y_min | y_max | x-Center | y-Center |
    '''
    
    for dirname, subdirList, fileList in os.walk(input_folder):
        # Walk through of input folder and printout of their contents.
        #print(dirname)
        #print(subdirList)
        #print(fileList)
        #print(len(fileList))
        #print('End of loop walk')
        
        # If fileList is greater than 1, then iterate through files.
        if len(fileList) > 25:
        
            for filename in os.listdir(dirname):
                dash_check = fnmatch.fnmatch(filename , '._*')
                # Print filenames.
                #print(filename)
        
                # Determine if filename has '.xml' extension.
                if ".xml" in filename.lower() and dash_check == False:
                    # If filename is an XML file, then print directory and name.
                    #print(dirname + "/" + filename)
                    # Define tree structure and obtain root node.
                    tree = ET.parse(dirname + "/" + filename)
                    root = tree.getroot()
                    
                    # The pre-fix '{http://www.nih.gov}' is unique to CT scans whereas
                    # chest x-rays use a different  URL pre-fix.
                    file_type = tree.findall('.//{http://www.nih.gov}TaskDescription')

                    #print(file_type)
                    #print('len',file_type)
                        
                    # If length of file_type is greater than 1 then the XML file belongs
                    # to a CT scan, if not then the XML file belongs to a chest x-ray.
                    if len(file_type) > 0:
                        #xml_file = file_type[0].text
                        xml_file = dirname + "/" + filename
                        #print('xml',xml_file)
                        break
                    else:
                        continue
    
    
    # Define XML as a tree structure.
    tree = ET.parse(xml_file)
    
    # Root node.
    root = tree.getroot()


    #####################################################################
    #
    #
    #Images = tree.findall('.//{http://www.nih.gov}imageSOP_UID')
    #
    #images = []
    #for im in range(len(Images)):
    #    images.append(Images[im].text)
    #
    #
    #imgs=[ii for n,ii in enumerate(images) if ii not in images[:n]]
    #
    #
    #####################################################################


    Position = tree.findall('.//{http://www.nih.gov}imageZposition')

    position = []
    for p in range(len(Position)):
        position.append(Position[p].text)


    position = [format(float(i),'.7f') for i in position]
    position = [n.strip() for n in position]

    pos = [ii for n,ii in enumerate(position) if ii not in position[:n]]
    pos1 = [ii for n,ii in enumerate(position) if ii not in position[:n]]

    #print(pos)
    #print(len(pos))
    
    pos = [float(i) for i in pos]
    pos1 = [float(i) for i in pos1]

    pos = np.array(pos)
    pos_check = pos1

    pos = np.hstack( ( np.reshape(pos,(len(pos),1)) , np.zeros((len(pos),8)) ) )
    counter = np.zeros((len(pos),1))

    
    #print(pos)
    #print(pos_check)
    #####################################################################


    grade = 0
    z_pos = 0
    x_coord = []
    y_coord = []
    for element in root.iter():
        if element.tag == '{http://www.nih.gov}imageZposition' and z_pos != 0:
            #print(x_coord)
            #print('--------------------------------------end-------------------------------------------------')
            x_coord = np.array(x_coord)
            y_coord = np.array(y_coord)
            x_min = np.amin(x_coord)
            y_min = np.amin(y_coord)
            x_max = np.amax(x_coord)
            y_max = np.amax(y_coord)
            #print(x_min)
            #print(x_max)
            #print(y_min)
            #print(y_max)
        
            # Column 3 contains x_min values.
            pos[ind,3] = pos[ind,3] + x_min
            # Column 4 contains x_max values.
            pos[ind,4] = pos[ind,4] + x_max
            # Column 5 contains y_min values.
            pos[ind,5] = pos[ind,5] + y_min
            # Column 6 contains y_max values.
            pos[ind,6] = pos[ind,6] + y_max
            # Counter is a vector of counters for each image slice.
            counter[ind,0] = counter[ind,0] + 1

            #print(pos)
            #print(counter)
            z_pos = 0
            x_coord = []
            y_coord = []
        
        if element.tag == '{http://www.nih.gov}unblindedReadNodule' and z_pos != 0:
            #print(x_coord)
            #print('----------------------------------------end-----------------------------------------------')
            x_coord = np.array(x_coord)
            y_coord = np.array(y_coord)
            x_min = np.amin(x_coord)
            y_min = np.amin(y_coord)
            x_max = np.amax(x_coord)
            y_max = np.amax(y_coord)
            #print(x_min)
            #print(x_max)
            #print(y_min)
            #print(y_max)
        
            # Column 3 contains x_min values.
            pos[ind,3] = pos[ind,3] + x_min
            # Column 4 contains x_max values.
            pos[ind,4] = pos[ind,4] + x_max
            # Column 5 contains y_min values.
            pos[ind,5] = pos[ind,5] + y_min
            # Column 6 contains y_max values.
            pos[ind,6] = pos[ind,6] + y_max
            # Counter is a vector of counters for each image slice.
            counter[ind,0] = counter[ind,0] + 1
        
            #print(pos)
            #print(counter)
            z_pos = 0
            x_coord = []
            y_coord = []
        
        if element.tag == '{http://www.nih.gov}nonNodule' and z_pos != 0:
            #print(x_coord)
            #print('----------------------------------------end-----------------------------------------------')
            x_coord = np.array(x_coord)
            y_coord = np.array(y_coord)
            x_min = np.amin(x_coord)
            y_min = np.amin(y_coord)
            x_max = np.amax(x_coord)
            y_max = np.amax(y_coord)
            #print(x_min)
            #print(x_max)
            #print(y_min)
            #print(y_max)
        
            # Column 3 contains x_min values.
            pos[ind,3] = pos[ind,3] + x_min
            # Column 4 contains x_max values.
            pos[ind,4] = pos[ind,4] + x_max
            # Column 5 contains y_min values.
            pos[ind,5] = pos[ind,5] + y_min
            # Column 6 contains y_max values.
            pos[ind,6] = pos[ind,6] + y_max
            # Counter is a vector of counters for each image slice.
            counter[ind,0] = counter[ind,0] + 1
            
            #print(pos)
            #print(counter)
            z_pos = 0
            x_coord = []
            y_coord = []
        
      
        if element.tag == '{http://www.nih.gov}unblindedReadNodule' or element.tag == '{http://www.nih.gov}nonNodule':
            grade = 0
            z_pos = 0
            #print('----------------------------------------Reset-----------------------------------------------')
            #print(grade)
    
        if element.tag == '{http://www.nih.gov}malignancy':
            grade = float(element.text)
            #print('----------------------------------------Update-----------------------------------------------')
            #print(grade)
    
        if element.tag == '{http://www.nih.gov}imageZposition' and grade > 0:
            z_pos = float(element.text)
            #print('z_pos',z_pos)
        
            ind = pos_check.index(z_pos)
        
            if grade >= mb_grade:
                pos[ind,1] = pos[ind,1] + 1
                #print(pos)
            
            if grade < mb_grade:
                pos[ind,2] = pos[ind,2] + 1
                #print(pos)
                
        if element.tag == '{http://www.nih.gov}xCoord' and grade > 0:
            xcoord = float(element.text)
            #print(xcoord)
            x_coord.append(xcoord)
        
        if element.tag == '{http://www.nih.gov}yCoord' and grade > 0:
            ycoord = float(element.text)
            #print(ycoord)
            y_coord.append(ycoord)
    
        #print element.tag , element.text 
    
    
    if z_pos != 0:
        #print('--------------------------------------end-------------------------------------------------')
        x_coord = np.array(x_coord)
        y_coord = np.array(y_coord)
        x_min = np.amin(x_coord)
        y_min = np.amin(y_coord)
        x_max = np.amax(x_coord)
        y_max = np.amax(y_coord)
        #print(x_min)
        #print(x_max)
        #print(y_min)
        #print(y_max)
    
        # Column 3 contains x_min values.
        pos[ind,3] = pos[ind,3] + x_min
        # Column 4 contains x_max values.
        pos[ind,4] = pos[ind,4] + x_max
        # Column 5 contains y_min values.
        pos[ind,5] = pos[ind,5] + y_min
        # Column 6 contains y_max values.
        pos[ind,6] = pos[ind,6] + y_max
        # Counter is a vector of counters for each image slice.
        counter[ind,0] = counter[ind,0] + 1
    
        #print(pos)
        #print(counter)
        z_pos = 0
        x_coord = []
        y_coord = []

    
    counter[counter == 0] = 1
    for index in range(len(counter)):
        pos[index,3:-1] = pos[index,3:-1]/counter[index]
        pos[index,7] = 0.5*( pos[index,4] + pos[index,3] )
        pos[index,8] = 0.5*( pos[index,6] + pos[index,5] )
        
    #print(pos)
    
    pos = vote_sys(pos , vote_threshold = 1)
    
    return pos



'''
-------------------------Initialization of the input folder.-------------------------
'''
#patients = os.listdir(input_folder)
#patients.sort()

# Initialize save data counters.
img_iter = 0
matrix_iter = 0
label_iter = 0

# Initialize arrays for images and labels.
img_stack = np.zeros((32,32))
labels_array = np.array([])

path = '/Volumes/SEAGATE2/CS680_project/LIDC-IDRI/'

# Initialize string for input_folder.
prefix = 'LIDC-IDRI-' 

fc = 0
skip_folders = [238,585,703,834]
# Initialize input_folder name.
for i in np.arange(3,4,1):
    
    if i in skip_folders:
        continue
    
    suffix = format(i,'04d')
    input_folder = path + prefix + suffix
    #print(input_folder)
    
    # Call the function xml_read and obtain the pos matrix for the input_folder.
    pos = xml_read(input_folder , mb_grade = 4)
    print('++++++++++++++++++++++++++++++++++++++')
    print(pos)
    print('++++++++++++++++++++++++++++++++++++++')
    
    if pos.size == 0:
        continue
    
    for dirname , subdirList , fileList in os.walk(input_folder):
        #print(dirname)
        #print(subdirList)
        #print(fileList)
        #print('End of walk')
    
        if len(subdirList) == 0:
            #print(dirname)
            
            decision_val = folder_check(dirname)
            
            if decision_val == 1:
                
                skip_val = z_check(dirname , pos)
                
                if skip_val > 0:
                    print(dirname)
                    imgs , labels = ct_scan_slices(dirname , pos)
                    img_stack = np.dstack((img_stack , imgs))
                    labels_array = np.vstack([labels_array , labels]) if labels_array.size else labels
                    #print('stack',img_stack.shape)
                    #print('label_array',labels_array.shape)
                    fc += 1
                    print('Folder Count',fc)
                    
                else:
                    continue
                
            else:
                continue
            
        else:
            #print('List not empty')
            continue


# Delete the initial zero array.
img_stack = np.delete(img_stack , (0) , axis = 2)
print('img_stack',img_stack.shape)

print('labels_array',labels_array.shape)
#print(img_stack.shape)
#print(labels_array)
#'''
#fig = plt.figure(figsize = (10, 10))
#plt.imshow(img_stack[:,:,0])
sfdfsdf
cc=0
features_array = np.array([])
for i in range(len(labels_array)):
    cc +=1
    #print(cc)
    lbp_array = local_binary_pattern(img_stack[:,:,i] , 1)
    #print('lbp',lbp_array.shape)
    hog_array = HoG(img_stack[:,:,i])
    #print(hog_array.shape)
    features_vector = np.hstack((lbp_array , hog_array))
    #print('both',features_vector.shape)
    features_array = np.vstack([features_array , features_vector]) if features_array.size else features_vector
    #print('f',features_array.shape)
    #print('Feature Counter',cc)
#'''

print('Features array obtained')
#'''
# Save numpy array.
#np.save('/Volumes/SEAGATE2/CS680_project/features_matrix14.npy',features_array)
#np.save('/Volumes/SEAGATE2/CS680_project/labels_vector14.npy',labels_array)

# Create function, f to open a text file and write to it.
f = open('texture_features14post','w')

# Dump data in features_array using the function, f.
pickle.dump(features_array,f)

# Close function.
f.close()

# Create function, f to open a text file and write to it.
f2 = open('labels_vector14post','w')

# Dump data in features_array using the function, f.
pickle.dump(labels_array,f2)

# Close function.
f2.close()

print('Complete')
#'''


# In[2]:


from scipy.sparse.linalg import eigs
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


# In[3]:


def calc_metrics(label , pred):
    '''
    This function calculates various measures commonly employed
    for machine learning analysis.
    '''
    # label 1 indicates malignant and label 0 indicates benign.
    # Initialize quantities to 0.
    TP = TN = FP = FN = 0
    
    for true_label , pred_label in zip(label , pred):
        if true_label == pred_label:
            if true_label == 1:
                TP += 1
            else:
                TN += 1
        else:
            if true_label == 1:
                FP += 1
            else:
                FN += 1
                
    assert((TP + FP + TN + FN) == len(pred))
    
    # Calculate the desired quantities.
    precision = float(TP)/(TP + FP)
    recall = float(TP)/(TP + FN)
    accuracy = (float(TP + TN))/(TP + FP + TN + FN)
    sensitivity = float(TP)/(TP + FN)
    specificity = float(TN)/(TN + FP)
    
    
    return precision , recall , accuracy , sensitivity , specificity


def cross_validate(k , U_train , y_train):
    '''
    This function applies 10-fold cross-validation.
    '''
    kf = KFold(10, True)
    
    iter = 0
    ave_precision = []
    ave_recall = []
    ave_accuracy = []
    ave_sensitivity = []
    ave_specificity = []
    
    for train_ind , test_ind in kf.split(U_train):
        val_U_train , val_y_train = U_train[train_ind] , y_train[train_ind]
        val_U_test , val_y_test = U_train[test_ind] , y_train[test_ind]
        
        clf_k = KNeighborsClassifier(n_neighbors = k)
        clf_k.fit(val_U_train , val_y_train)
        pred_k = clf_k.predict(val_U_test)
        
        precision , recall , accuracy , sensitivity , specificity = calc_metrics(val_y_test , pred_k)
        
        ave_precision.append(precision)
        ave_recall.append(recall)
        ave_accuracy.append(accuracy)
        ave_sensitivity.append(sensitivity)
        ave_specificity.append(specificity)
        
        iter +=1
        
    precision = round(np.average(ave_precision) , 4)
    recall = round(np.average(ave_recall) , 4)
    accuracy = round(np.average(ave_accuracy) , 4)
    sensitivity = round(np.average(ave_sensitivity) , 4)
    specificity = round(np.average(ave_specificity) , 4)
    
    
    return precision , recall , accuracy , sensitivity , specificity
    


# In[17]:


def pca_fun(X_data):
    '''
    
    '''
    global x_mean
    global e_vec
    global U
    
    # Extract dimensions of X_data
    row , col = X_data.shape
    
    # Obtain the mean of each feature for all images.
    x_mean = np.mean(X_data , axis = 0)
    
    # Clone the mean vector into a matrix.
    X_mean = np.array([x_mean,]*row)
    
    # Obtain the mean-subtracted matrix.
    X_corrected = np.subtract(X_data , X_mean)
    
    # Obtain the covariance matrix.
    X_cov = np.cov(X_data.transpose())
    
    # Determine eigenvalues and eigentvectors of X_cov and sort based on largest magnitude.
    e_val , e_vec = eigs(X_cov , k=100 , which = 'LM')

    # Return only the real components.
    e_val = np.real(e_val)
    e_vec = np.real(e_vec)

    # Return eigenvalues greater than or equal to 1.0.
    e_val = e_val[e_val >= 1.0]
    e_vec = e_vec[: , range(len(e_val))]
    
    U = np.matmul(X_corrected , e_vec)
    
    return U
    
'''
f_texture = open('texture_features4post', 'r')
feat_matrix = pickle.load(f_texture)
f_texture.close()


f_label = open('labels_vector4post', 'r')
labels_vector = pickle.load(f_label)
f_label.close()

print(feat_matrix.shape)
print(labels_vector.shape)

X = feat_matrix
y = labels_vector
'''
X = q_prime
y = w
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Perform PCA on X_train.
U = pca_fun(X_train)


#'''
precisions = []
recalls = []
accuracies = []
sensitivities = []
specificities = []

#k_test = range(1,30)
k_test = np.arange(1,50,1)

for i in k_test:
    precision , recall , accuracy , sensitivity , specificity = cross_validate(i , U , y_train)
    
    print("Number of neighbors", i)
    print('\t precision - ', round(precision, 4)) 
    print('\t recall - ', round(recall, 4)) 
    print('\t accuracy - ', round(accuracy, 4)) 
    print('\t sensitivity - ', round(sensitivity, 4)) 
    print('\t specificity - ', round(specificity, 4))
    print(' ')
    
    precisions.append(precision)
    recalls.append(recall)
    accuracies.append(accuracy)
    sensitivities.append(sensitivity)
    specificities.append(specificity)
#'''



# In[20]:


# Plot of accuracies as a function of contributing neighbors.
def plot_knn():
    fig = plt.figure()
    plt.plot(k_test , accuracies,'m-', label='Accuracy')
    plt.plot(k_test , sensitivities,'g-', label='Sensitivity')
    plt.plot(k_test , specificities,'b-', label='Specificity')
    plt.title('Performance Measures as a \n Function of Contributing Neighbor \n Examples with Split Vote', size = 20)
    plt.xlabel('Number of Contributing Neighbors',size = 20)
    plt.ylabel('Fraction',size = 20)
    plt.legend()#loc='lower right')
    plt.grid()

    plt.show()

    fig.savefig('10fold_cv_split_vote',bbox_inches = 'tight')

    
    return

plot_knn()


# In[19]:


def pca_knn_clf(X_test , y_test , e_vec):
    '''
    This function applies principles component analysis
    for k-nearest neighbours.
    '''
    test_row , test_col = X_test.shape
    
    #x_test_m = np.mean(X_test, axis = 0)
    
    X_test_mean = np.array([x_mean,]*test_row)
    
    X_correct = X_test - X_test_mean
    
    #e_vec = np.real(e_vec)
    
    U_test = np.matmul(X_correct , e_vec)
        
    #print(X_test.shape)
    knn_predict = knn_clf.predict(U_test)
    

    return knn_predict


# Classification of test data with KNN classifier using optimal k value.
k_opt = np.argmax(accuracies) + 1
print(k_opt)

# Define classifier for optimal k values.
knn_clf = KNeighborsClassifier(n_neighbors = k_opt)

# Fit classifier with optimal k value.
knn_clf.fit(U, y_train)

knn_predict = pca_knn_clf(X_test , y_test , e_vec)

# Calculate metrics for test data.
precision , recall , accuracy , sensitivity , specificity = calc_metrics(y_test , knn_predict)

print("< Test Set Metrics >")
print('\tprecision - ', round(precision, 4)) 
print('\trecall - ', round(recall, 4)) 
print('\taccuracy - ', round(accuracy, 4)) 
print('\tsensitivity - ', round(sensitivity, 4)) 
print('\tspecificity - ', round(specificity, 4))

