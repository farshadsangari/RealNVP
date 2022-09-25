
import data_utilities
from torch.utils.data import DataLoader




def dataloader(base_raw_data_path,
               batch_size):

    list_of_celebs_path = data_utilities.get_file_list(base_raw_data_path)
    list_of_celebs_path_train,list_of_celebs_path_val = data_utilities.split_data(list_of_celebs_path)
    data_transform = data_utilities.transform_celeba()

    train_data = data_utilities.myDataset(list_of_celebs_path_train,data_transform)
    val_data = data_utilities.myDataset(list_of_celebs_path_val,data_transform)

    train_dataloader = DataLoader(dataset = train_data , batch_size = batch_size,shuffle=True)
    val_dataloader = DataLoader(dataset = val_data , batch_size = batch_size,shuffle=True)
    
    return train_dataloader,val_dataloader