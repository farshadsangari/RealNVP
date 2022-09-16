
import torch
import pandas as pd
from tqdm import tqdm
import os
import models as models
import utils as utils




def Train_mode(model,
               cuda,
               train_dataloader,
               val_dataloader,
               criterion,
               optimizer,
               lr_scheduler,
               num_epochs,
               max_grad_norm,
               saving_checkpoint_path,
               saving_prefix,
               saving_checkpoint_freq,
               report_path):
    
    global avg_loss_per_epoch,reverse,report
    
    
    report = pd.DataFrame(columns=["mode", "epoch", "batch_index","learning_rate", "loss_batch","avg_epoch_loss_till_current_batch"])

    reverse =False
    avg_loss_per_epoch = [[] for i in range(2)]        # First sublist = Train loss / Second sublist = val loss 

    ###################################    Training mode     ##########################################
    for epoch in range(1,num_epochs+1):
            avg_train_loss = utils.AverageMeter()
            avg_val_loss = utils.AverageMeter()
            mode = 'train'
            model.train()
            # Loop for train batches
            loop_train = tqdm(enumerate(train_dataloader, 1),total=len(train_dataloader),desc='train')
            for batch_index,inputs in loop_train:
                if cuda:
                    inputs = inputs.cuda()
                optimizer.zero_grad()
                outputs,sldj = model(inputs)
                loss = criterion(outputs, sldj)
                optimizer.zero_grad()
                loss.backward()
                avg_train_loss.update(loss.item(), inputs.size(0))
                
                report = report.append({
                    'mode': mode,
                    'epoch':epoch,
                    'batch_index':batch_index,
                    'learning_rate':optimizer.param_groups[0]["lr"],
                    'avg_epoch_loss_till_current_batch':avg_train_loss.avg
                                        })
                

                utils.clip_grad_norm(optimizer, max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                
                    
                loop_train.set_description(f'Train mode - epoch : {epoch}')
                loop_train.set_postfix(Loss_Train = "{:.4f}".format(avg_train_loss.avg),refresh=True)
                if (epoch % saving_checkpoint_freq) == 0 : 
                    models.save(
                        file_path=saving_checkpoint_path,
                        file_name=f"{saving_prefix}{epoch}.ckpt",
                        model=model,
                        optimizer=optimizer)
            avg_loss_per_epoch[0].append(avg_train_loss.avg)
            


    ################################    Validation mode   ##############################################
            model.eval()
            mode = 'validation'
            with torch.no_grad():
            
                # Loop for val batches
                loop_val = tqdm(enumerate(val_dataloader, 1),total=len(val_dataloader),desc='val')
                for batch_index,inputs in loop_val:
                    if cuda:  
                        inputs = inputs.cuda()
                    optimizer.zero_grad()
                    outputs,sldj = model(inputs)
                    loss = criterion(outputs, sldj)
                    avg_val_loss.update(loss.item(), inputs.size(0))
                    
                    
                    
                    report = report.append({
                            'mode': mode,
                            'epoch':epoch,
                            'batch_index':batch_index,
                            'learning_rate':optimizer.param_groups[0]["lr"],
                            'avg_epoch_loss_till_current_batch':avg_val_loss.avg
                                          })
                    
                    optimizer.zero_grad()
                    loop_val.set_description(f'Validation mode - epoch : {epoch}')
                    loop_val.set_postfix(Loss_val = "{:.4f}".format(avg_val_loss.avg),refresh=True)
                avg_loss_per_epoch[1].append(avg_val_loss.avg)
    
    
    report.to_csv(os.path.join(report_path, f"report_training.csv"))
    return model,optimizer


def Inference_mode(model,
                   cuda,
                   test_dataloader,
                   criterion,
                   optimizer,
                   ):
    
    
            global reverse
            reverse =False
            model.eval()
            with torch.no_grad():
                avg_test_loss = utils.AverageMeter()
                # Loop for test batches
                loop_test = tqdm(enumerate(test_dataloader, 1),total=len(test_dataloader),desc='val')
                for i,inputs in loop_test:
                    if cuda:  
                        inputs = inputs.cuda()
                    optimizer.zero_grad()
                    outputs,sldj = model(inputs)
                    loss = criterion(outputs, sldj)
                    optimizer.zero_grad()
                    avg_test_loss.update(loss.item(), inputs.size(0))
                    
                    loop_test.set_description(f'Test mode')
                    loop_test.set_postfix(Loss_test = "{:.4f}".format(avg_test_loss.avg),refresh=True)
                print(f'@the end of training, Test loss value is : {avg_test_loss.avg}')
                
                
                
                
def Sample(model, batch_size, device):
    """Sample from RealNVP model.
    Args:
        model (torch.nn.DataParallel): The RealNVP model wrapped in DataParallel.
        batch_size (int): Number of samples to generate.
        device (torch.device): Device to use.
    """
    z = torch.randn((batch_size, 3, 32, 32), dtype=torch.float32, device=device)
    x, _ = model(z, reverse=True)
    x = torch.sigmoid(x)

    return x