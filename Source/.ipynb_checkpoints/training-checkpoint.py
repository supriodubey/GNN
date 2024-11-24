# F
# Routines for training and testing the GNNs
# 

from Source.constants import *
import os

# Training step
def train(loader, model, num_params, optimizer, scheduler):

    model.train()
    loss_tot = 0
    

    # Iterate in batches over the training dataset
    for data in loader:  

        data.to(device)
        optimizer.zero_grad()  # Clear gradients.
        out = model(data)  # Perform a single forward pass
        # print(f"out is {out} \n")
        # print(f"output shape is: {out.shape}")
       
      

        # Perform likelihood-free inference to predict also the standard deviation
        # Take mean and standard deviation of the output
        y_out, err_out = out[:,: num_params], out[:, num_params:2* num_params]    
        #print(f"Predicted: {y_out}, True: {data.y}")
        # print(f"y_out is {y_out}")
        # print(f"err_out is {err_out}")


        # Compute loss as sum of two terms for likelihood-free inference
        loss_mse = torch.mean(torch.sum((y_out - data.y)**2., axis=1) , axis=0)
        loss_lfi = torch.mean(torch.sum(((y_out - data.y)**2. - err_out**2.)**2., axis=1) , axis=0)
        loss = torch.log(loss_mse) + torch.log(loss_lfi)
        
        # Derive gradients
        loss.backward()  

        # Update parameters based on gradients
        optimizer.step()  
        scheduler.step() 
        loss_tot += loss.item()

    return loss_tot/len(loader)


# Testing/validation step
def test(loader, model, num_params):

    model.eval()

    trueparams = np.zeros((1,  num_params))
    outparams = np.zeros((1, num_params))
    outerrparams = np.zeros((1, num_params))

    errs = []
    chi2s = []
    loss_tot = 0

    # Iterate in batches over the training/test dat
    for data in loader:  
        with torch.no_grad():

            data.to(device)
            out = model(data)  # prediction of the model
            
            # perform likelihood-free inference to predict also the standard deviation
            # Take mean and standard deviation of the output
            y_out, err_out = out[:,: num_params], out[:, num_params:2* num_params]

            # Compute loss as sum of two terms for likelihood-free inference
            loss_mse = torch.mean(torch.sum((y_out - data.y)**2., axis=1) , axis=0)
            loss_lfi = torch.mean(torch.sum(((y_out - data.y)**2. - err_out**2.)**2., axis=1), axis=0)
            loss = torch.log(loss_mse) + torch.log(loss_lfi)
            
            # absolute error
            err = (y_out - data.y)
            errs.append(np.abs(err.detach().cpu().numpy()).mean() )

            # chi2
            # chi2 = (y_out - data.y)**2 / err_out**2
            # chi2s.append((chi2.detach().cpu().numpy()).mean())
            
            loss_tot += loss.item()

            # Append true values and predictions
            trueparams = np.append(trueparams, data.y.detach().cpu().numpy(), 0)
            # print(f"True : {trueparams}\n")
            outparams = np.append(outparams, y_out.detach().cpu().numpy(), 0)
            # print(f"Predicted : {outparams}\n")
            outerrparams = np.append(outerrparams, err_out.detach().cpu().numpy(), 0)
            # print(f"Error : {err}\n")
    os.makedirs('/mnt/Outputs', exist_ok=True)
    # Save true values and predictions (for plotting)
    np.save("/mnt/Outputs/true_values.npy", trueparams)
    np.save("/mnt/Outputs/predicted_values.npy", outparams)
    np.save("/mnt/Outputs/errors_predicted.npy", outerrparams)

    return loss_tot/len(loader), np.array(errs).mean(axis=0)#, np.array(chi2s).mean()


# Training procedure
def training_routine(model, train_loader, valid_loader, test_loader, hparams, num_params, verbose=True):

    # Define optimizer and learning rate scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams.learning_rate, weight_decay=hparams.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = hparams.T_max, eta_min = 0, last_epoch = -1)
    
    # initializing losses and errors 
    train_losses, valid_losses = [], []
    valid_loss_min, err_min = 1000., 1000.
    # chi2_min = 1e6
    counter = 0
    # Training loop
    for epoch in range(1, hparams.n_epochs+1):
        train_loss = train(train_loader, model, num_params, optimizer, scheduler)

        valid_loss, err = test(valid_loader, model, num_params)
        # test_loss, err = test(test_loader, model, num_params)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        # chi2s.append(chi2)

        # Save model if it has improved 
        if valid_loss <= valid_loss_min - hparams.min_delta:
            if verbose: 
                print(f"Validation loss decreased ({valid_loss_min:.3f} --> {valid_loss:.3f}). Saving model ...") 
            torch.save(model.state_dict(), "Models/best_model_from_training")
            valid_loss_min = valid_loss
            err_min = err
            counter = 0 # Resetting Counter if there is an improvement 
        else:
            counter += 1
        
        # EARLY STOPPING
        if  counter >= hparams.patience:
            if verbose: 
                print(f" No improvement in loss ### Early Stopping ### after {epoch} epochs.")
            break
        if verbose:
            print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.3f}, Validation Loss: {valid_loss:.3f}, Error: {err:.3f}')

    return train_losses, valid_losses

