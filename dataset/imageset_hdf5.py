import glob
from dataset import CXRdataset as CXR
import numpy as np
import h5py
from PIL import Image

# Set the path to hdf5 file
hdf5_path = 'D:/Project workspace/CXR_Project/dataset/imageset.hdf5'
cxr_images_path = 'D:/Project workspace/CXR_Project/dataset/images/*.png'

# Read images addresses and all labels
address = glob.glob(cxr_images_path)
label_list = CXR.load_csv_label() 
labels  = CXR.change_multi_label(label_list)

# Divide the data into 60% train, 20% validation, 20% test
train_addr   = address[0 : int(0.6 * len(address))]
train_labels =  labels[0 : int(0.6 * len(address))]

validation_addr   = address[int(0.6 * len(address)) : int(0.8 * len(address))]
validation_labels =  labels[int(0.6 * len(address)) : int(0.8 * len(address))]

test_addr   = address[int(0.8 * len(address)) :]
test_labels =  labels[int(0.8 * len(address)) :]


# Data shape to save images (img_num, height, width, depth)
# Data shape to save labels (labels_num, height, width)
train_img_shape      = (len(train_addr)     , 1024, 1024)
validation_img_shape = (len(validation_addr), 1024, 1024)
test_img_shape       = (len(test_addr)      , 1024, 1024)

train_label_shape      = (len(train_labels)     , 15)
validation_label_shape = (len(validation_labels), 15)
test_label_shape       = (len(test_labels)      , 15)


# Open hdf5 file
# Load and save labels directly
with h5py.File(hdf5_path, mode = 'w') as hdf:

    hdf.create_dataset('train_images'     , train_img_shape     , dtype = np.int8)
    hdf.create_dataset('validation_images', validation_img_shape, dtype = np.int8)
    hdf.create_dataset('test_images'      , test_img_shape      , dtype = np.int8)
    
    hdf.create_dataset('train_labels'     , train_label_shape     , dtype = np.int8)
    hdf['train_labels'][...]      = train_labels
    
    hdf.create_dataset('validation_labels', validation_label_shape, dtype = np.int8)
    hdf['validation_labels'][...] = validation_labels
    
    hdf.create_dataset('test_labels'      , test_label_shape      , dtype = np.int8)
    hdf['test_labels'][...]       = test_labels


    # Load images -> Transform to numpy arrays -> Save the arrays
    # Train images
    for i in range(len(train_addr)):
        if i % 1000 == 0 and i > 1:
            print( 'Saved/Total(Train): ', i, len(train_addr))
            
        addr = train_addr[i]
        img =  Image.open(addr)
        img_arr = np.array(img)
        
        if img_arr.ndim != 2:
            img_arr = img_arr[ : , :, 0]
        
        hdf['train_images'][i, ] = img_arr
    
    # Validation images
    for i in range(len(validation_addr)):
        if i % 1000 == 0 and i > 1:
            print( 'Saved/Total(Validation): ', i, len(validation_addr))
            
        addr = validation_addr[i]
        img =  Image.open(addr)
        img_arr = np.array(img)
        
        if img_arr.ndim != 2:
            img_arr = img_arr[ : , :, 0]
            
        hdf['validation_images'][i, ] = img_arr
    
    # Test images
    for i in range(len(test_addr)):
        if i % 1000 == 0 and i > 1:
            print( 'Saved/Total(Test): ', i, len(test_addr))
            
        addr = test_addr[i]
        img =  Image.open(addr)
        img_arr = np.array(img)
        
        if img_arr.ndim != 2:
            img_arr = img_arr[ : , :, 0]
            
        hdf['test_images'][i, ] = img_arr