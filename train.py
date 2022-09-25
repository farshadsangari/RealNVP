import math
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

######  Local packages  ######
import learning as learning
import models as models
import loss_function as loss_function
import data as data
import utils as utils
import optim as optim


def main(args):
    cuda = True if args.gpu and torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
    
        
    list_of_celebs_path = data.get_file_list(args.base_raw_data_path)
    list_of_celebs_path_train,list_of_celebs_path_val = data.split_data(list_of_celebs_path)
    data_transform = data.transform_celeba()

    train_data = data.myDataset(list_of_celebs_path_train,data_transform)
    val_data = data.myDataset(list_of_celebs_path_val,data_transform)

    train_dataloader = DataLoader(dataset = train_data , batch_size = args.batch_size,shuffle=True)
    val_dataloader = DataLoader(dataset = val_data , batch_size = args.batch_size,shuffle=True)
    
    model = models.realNVP(num_scales = args.num_scales,
                    current_scale_idx = 0,
                    input_channels = args.input_channels,
                    middle_channels = args.middle_channels,
                    data_constraint = args.data_constraint,
                    num_blocks = args.num_blocks,
                    use_resnet=args.use_resnet)
    
    criterion = loss_function.RealNVPLoss(distribution_mean=args.distribution_mean,
                                          distribution_std=args.distribution_std,
                                          k=args.possible_values_in_each_input_dimension)
    
    
    optimizer = optim.adam_optim(
                params=model.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay)
    
    # Loading Model
    if args.ckpt_load_path is not None:
        print("******  Loading Model   ******")
        model, optimizer = utils.load_model(
                            ckpt_path=args.ckpt_load_path,
                            model=model,
                            optimizer=optimizer)
        
    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()
    
    # Schedular
    num_train_steps = math.ceil(len(train_dataloader) / args.batch_size)*args.num_epochs

    lr_scheduler = get_linear_schedule_with_warmup(
                                        optimizer=optimizer,
                                        num_warmup_steps=0,
                                        num_training_steps=num_train_steps)
    
    
    
    # Train the model(Train and Validation Steps)
    model,optimizer = learning.Train_mode(model = model,
                                cuda=cuda,
                                train_dataloader = train_dataloader,
                                val_dataloader = val_dataloader,
                                criterion = criterion,
                                optimizer = optimizer,
                                lr_scheduler = lr_scheduler,
                                num_epochs = args.num_epochs,
                                max_grad_norm = args.max_grad_norm,
                                saving_checkpoint_path=args.ckpt_save_path,
                                saving_prefix=args.ckpt_prefix,
                                saving_checkpoint_freq=args.ckpt_save_freq,
                                report_path = args.report_path)
    
    return model



if __name__ == "__main__":
    args = utils.get_args()
    main(args)


