import os
import torch

# Save weights
def save(net, 
         logger, 
         model_save_dir,
         optimizer,
         scheduler,
         epoch,
         loss, 
         exp="best"):
    # path to save model according best or last
    path = os.path.join(model_save_dir, exp)

    # checkpoint
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'logs': logger.get_logs(),
    }

    # save checkpoint
    torch.save(checkpoint, path + ".pth")

# Load trained weights
def restore(path, net, logger, optimizer, scheduler):
    """ Load back the model and logger from a given checkpoint
        epoch detailed in hps['restore_epoch'], if available"""

    if os.path.exists(path):
        try:
            checkpoint = torch.load(path)
            logger.restore_logs(checkpoint['logs'])
            net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            epoch = checkpoint['epoch']
            loss = checkpoint['loss']
            net.train()
            print("Network Restored!")
            return {'epoch':epoch, 
                    'loss': loss, 
                    'optimizer': optimizer, 
                    'net': net, 
                    'logger': logger}

        except Exception as e:
            print("Restore Failed! Training from scratch.")
            print(e)
            return {'epoch': -1}
    else:
        print("Restore point unavailable. Training from scratch.")
        return {'epoch': -1}



def load_features(model, params):
    """ Load params into all layers of 'model'
        that are compatible, then freeze them"""
        
    model_dict = model.state_dict()

    imp_params = {k: v for k, v in params.items() if k in model_dict}

    # Load layers
    model_dict.update(imp_params)
    model.load_state_dict(imp_params)

    # Freeze layers
    for name, param in model.named_parameters():
        param.requires_grad = False
