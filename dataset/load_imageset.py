import numpy as np
import h5py

'''Load the hdf5 file and return dataset'''
def load_hdf5_file(total_data_num, normalize = True, validate = False):
    hdf5_path = 'D:/Project workspace/CXR_Project/dataset/imageset.hdf5'
    dataset = {}
    
    train_num = int(0.6 * total_data_num)
    valid_num = int(0.2 * total_data_num)
    test_num  = int(0.2 * total_data_num)
    
    try:
        with h5py.File(hdf5_path, mode = 'r') as hdf:
            dataset['train_images'] = hdf['train_images'][0:train_num, ...]
            dataset['train_labels'] = hdf['train_labels'][0:train_num, ...]
            dataset['test_images']  = hdf['test_images'][0:test_num, ...]
            dataset['test_labels']  = hdf['test_labels'][0:test_num, ...]
            
            if validate:
                dataset['validation_images'] = hdf['validation_images'][0:valid_num, ...]
                dataset['validation_labels'] = hdf['validation_labels'][0:valid_num, ...]

    except Exception as e:  
        print(e) 
    
    if normalize:
        for key in ('train_images', 'test_images',):
            dataset[key] = dataset[key].astype(np.float16)
            dataset[key] /= 255.0
            
        if validate:
            dataset['validation_images'] = dataset['validation_images'].astype(np.float16)
            dataset['validation_images'] /= 255.0

            return (
                    dataset['train_images'] , dataset['train_labels'], 
                    dataset['validation_images'], dataset['validation_labels'], 
                    dataset['test_images'], dataset['test_labels']
                    )
    
    return dataset['train_images'] , dataset['train_labels'], dataset['test_images'], dataset['test_labels']