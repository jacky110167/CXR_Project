import os.path
import numpy as np

# Create a dictionary ( folder_name : folder_path )
# One image folder and one csv file
_images_dict = {
                'images'  : '/images/', 
                'labels'  : '/labels/Data_Entry_2017.csv',
               }

all_labels = (
            'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 
            'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
            'Nodule', 'No Finding', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
             )

# Absolute path to dataset
_dataset_dir = os.path.dirname(os.path.abspath(__file__))
print('Absolute path to dataset:\n ', _dataset_dir)

total_img_num = 112120
train_img_num = 67272
validation_img_num = 22424
test_img_num = 22424

img_dim = (1024, 1024)
img_size = 1048576 # 1024*1024


# Extract labels from csv files
def load_csv_label():
    print('Load labels in csv file......') 
    
    csv_path = _dataset_dir + _images_dict['labels']
 
    try:
        _ , labels , _ , _ , _ , _, _ , _ , _ , _ , _ = np.loadtxt(csv_path, 
                                                                   delimiter = ',',
                                                                   unpack = True,
                                                                   dtype = 'str')
    except Exception as e:  
        print(e) 
           
    print('Completed')

    return labels


# Change original labels to multi-hot labels -> [1, 0, 0, 1, 0, ......]
def change_multi_label(labels):
    change_labels = np.zeros((len(labels), 15))
    
    print('Change the form of the original labels......') 
    
    for i in range(0, len(labels)):
        '''Seperate the multiple labels by "|" '''
        spl = list(labels[i].split('|'))
        
        for j in range(0, 15):
            if all_labels[j] in spl: 
                change_labels[i][j] = 1
                
                if(len(spl) == 1):  break
                
    print('Completed')
    
    return change_labels