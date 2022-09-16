import os
from torchvision.transforms import transforms
from torch.utils.data import Dataset
import cv2
import random
import matplotlib.pyplot as plt


def get_file_list(data_path):
    """find data pathes base on extentions

    Args:
        data_path : base path with it has Folders OR files which contatin that format of data
    Raises:
        FileNotFoundError: No data was found

    Returns:
        list: it returns list of data path(es)
    """
    data_list = []
    if os.path.isfile(data_path):
        data_list.append(data_path)
    else:
        for subdir, dirs, files in os.walk(data_path):
            for file in files:
                if file.lower().endswith((".jpg")):
                    data_list.append(os.path.join(subdir, file))           
    data_list.sort()
    if not data_list:
        raise FileNotFoundError("No data was found")
    return data_list

def split_data(list_of_celebs_path,percentage_of_train=70):
    percentage_of_train /= 100
    last_of_train_index = round(len(list_of_celebs_path)*percentage_of_train)
    list_of_celebs_path_train = list_of_celebs_path[:last_of_train_index]
    list_of_celebs_path_val = list_of_celebs_path[last_of_train_index:]
    return list_of_celebs_path_train,list_of_celebs_path_val


class myDataset(Dataset):
    def __init__(self,images_directory,transform):
        self.img_files = images_directory
        self.transform = transform

    def __len__(self):
        return len(self.img_files)
        
    def __getitem__(self,index):
        img = cv2.imread(self.img_files[index])
        if self.transform:
            return self.transform(img)
        else: 
            return img
def transform_celeba():
    data_transform = transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.CenterCrop((148,148)),
                                            transforms.Resize((64,64)),
                                            transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),  
                                        ])
    return data_transform


def plot_input_sample(dataloader,
                      data_transform,
                      batch_size,
                      ):
    random_index = random.randint(0,batch_size-1)
    if data_transform:
        plt.imshow(next(iter(dataloader))[random_index].numpy().transpose((1, 2, 0))[:,:,[2,1,0]])
    else:
        plt.imshow(next(iter(dataloader))[random_index][:,:,[2,1,0]])