
import torch
import torch.nn as nn
import torch.optim as optim

from easydict import EasyDict as edict

from dataset import get_dataloader
from model import Encoder, Decoder
from monai.utils import set_determinism
import numpy as np
from PIL import Image

import os



# Logging
import logging
import mlflow
import psutil
from torch.utils.tensorboard import SummaryWriter
import tempfile
# Adicionado 08/01/2026 - teste de sliding window
import torch.nn as nn
from tqdm import tqdm
from monai.inferers import SlidingWindowInferer


def log_memory_usage(logger, stage):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    logger.info(f"{stage} - Memory usage: {mem_info.rss / 1024 ** 2} MB")


def log_disk_usage(logger, stage):
    disk_usage = psutil.disk_usage('/')
    read_count, write_count = psutil.disk_io_counters()[:2]
    logger.info(f"{stage} - Disk usage: {disk_usage.percent}%, Read count: {read_count}, Write count: {write_count}")


def initialize_logging(cfg):
    logging.basicConfig(
        filename=os.path.join(
            cfg.log_to_dir, f"log_{cfg.best_metric}_model_{cfg.backbone}_run_{cfg.run_number}.txt"
        ),
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
    )
    return logging.getLogger()


def initialize(cfg):
    set_determinism(seed=0)
    logger = initialize_logging(cfg)

    #activations_fn = Activations(sigmoid=True)

    train_loader, val_loader = get_dataloader(cfg, mode="train")


    # Quando permitir loading descomenta
    # if cfg.resume_train and cfg.model_path != None:
    #     cfg.model_path = os.path.join(cfg.save_model_dir, f"best_AUC_{cfg.backbone}_run_{int(cfg.run_number)}.pth")
    #     model = torch.load(cfg.model_path, map_location=torch.device(cfg.device))

    # else:
    #     model = Classifier(cfg)
    #     model.to(cfg.device)

    #optimizer, scheduler = get_optimizer(cfg, model, loss_fn, train_loader, val_loader)

    return (
        SummaryWriter(
            log_dir=os.path.join(cfg.log_to_dir, f"Train_{cfg.backbone}_{cfg.run_number}")
        ),
        SummaryWriter(
            log_dir=os.path.join(cfg.log_to_dir, f"Validation_{cfg.backbone}_{cfg.run_number}")
        ),
        train_loader, val_loader,
        logger,
    )


# Adicionado 08/01/2026 - teste de sliding window
class AutoEncoderWrapper(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)




#def train( h, w, z, device_ids, epochs):
def train_model(cfg, device_ids):
    
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.mlflow_experiment_name)
    #mlflow.set_tracking_uri("sqlite:///mlflow.db")
    
    with mlflow.start_run(run_name=cfg.mlflow_run_name):
        mlflow.set_tag("model_name", cfg.backbone)
        mlflow.log_params(vars(cfg))


    try:
        (train_writer,
        val_writer,
        train_loader,
        val_loader,
        logger,) = initialize(cfg)


        
        print('Carregando arquitetura')
        # Adicionado 08/01/2026 - teste de sliding window
        encoder = Encoder(cfg.img_size[0], cfg.img_size[1], cfg.img_size[2], z_dim=512).to(cfg.device)
        decoder = Decoder(cfg.img_size[0], cfg.img_size[1], cfg.img_size[2], z_dim=512).to(cfg.device)
    
        ae_loss = nn.MSELoss()
    
        optimizer_ae = optim.Adam([{'params': encoder.parameters()},
                                {'params': decoder.parameters()}], lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        if cfg.pretrained:
            step = 0
            print('Retomando treinamento a partir do checkpoint:')
            checkpoint = torch.load(cfg.checkpoint_path, weights_only=True)
            
            encoder.load_state_dict(checkpoint['encoder'])
            decoder.load_state_dict(checkpoint['decoder'])
            optimizer_ae.load_state_dict(checkpoint['optimizer_ae'])
            #optimizer_dec.load_state_dict(checkpoint['optimizer_dec'])
            
            start_epoch = checkpoint['epoch'] + 1  # continua da próxima época
            best_loss   = checkpoint['loss']
            print(f'Checkpoint carregado. Retomando da época {start_epoch}.')
            # enc_path = 'reports/encoder_best_pre.pth'
            # dec_path = 'reports/decoder_best_pre.pth'
            # encoder = Encoder(cfg.img_size[0], cfg.img_size[1], cfg.img_size[2], z_dim=512).to(cfg.device)
            # decoder = Decoder(cfg.img_size[0], cfg.img_size[1], cfg.img_size[2], z_dim=512).to(cfg.device)
            # enc_state_dict = torch.load(enc_path, weights_only = True)['encoder']
            # dec_state_dict = torch.load(dec_path, weights_only = True)['decoder']
            # encoder.load_state_dict(enc_state_dict)
            # decoder.load_state_dict(dec_state_dict)
        else:
            start_epoch = 0
            step = 0
            best_loss = 100
            
            
            


        

        # 3. Apply DataParallel if using multiple GPUs
        if torch.cuda.device_count() > 1:
            encoder = nn.DataParallel(encoder).to(cfg.device)
            decoder = nn.DataParallel(decoder).to(cfg.device)
        
        #model_wrapper = AutoEncoderWrapper(encoder, decoder)
        #inferer = SlidingWindowInferer(roi_size=roi_size, sw_batch_size=4, overlap=0.5)
        


        # if distribution
        # encoder = nn.DataParallel(encoder, device_ids=device_ids).to(cfg.device)
        # decoder = nn.DataParallel(decoder, device_ids=device_ids).to(cfg.device)



        #tensorboard_path, saved_model_path, log_path = form_results(f'{h}-{w}-{z}', z_dim)
        #writer = SummaryWriter(tensorboard_path)





        for epoch in tqdm(range(start_epoch,cfg.num_epochs)):
            print("-" * 10)
            print(f"epoch {epoch + 1}/{cfg.num_epochs}")
            encoder.train()
            decoder.train()

            autoencoder_loss_epoch = 0.0

            for data in train_loader:
                img = data['im'].to(cfg.device)
                # ==========forward=========
                z = encoder(img)
                x_hat = decoder(z)

                # ==========compute the loss and backpropagate=========

                encoder_decoder_loss = ae_loss(x_hat, img)

                optimizer_ae.zero_grad()
                encoder_decoder_loss.backward()
                optimizer_ae.step()

                # ========METRICS===========
                autoencoder_loss_epoch += encoder_decoder_loss.item()
                
                train_writer.add_scalar("Encoder loss", encoder_decoder_loss, step)
                mlflow.log_metric("Encoder Train Loss", encoder_decoder_loss.item(), step=epoch)


                step += 1



            train_loss = autoencoder_loss_epoch / len(train_loader)


            val_loss = val(cfg, val_loader, encoder, decoder)

            print('train_loss: {:.4f}'.format(train_loss))
            print('val_loss: {:.4f}'.format(val_loss))
            train_writer.add_scalars('train and val loss per epoch', {'train_loss': train_loss,
                                                                'val_loss': val_loss
                                                                }, epoch + 1)

            mlflow.log_metric("Encoder Val Loss", val_loss, step=epoch)        

            # Log images to MLflow
            if (epoch + 1) % 10 == 0:  # control frequency to avoid bloating MLflow
                def tensor_to_mlflow_image(tensor):
                    
                    vol = tensor[0, 0]
                    mid = vol.shape[-1] // 2
                    slice_2d = vol[..., mid].cpu().float()
                    slice_2d = (slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min() + 1e-8)
                    return slice_2d.detach().numpy()  # ← add .detach()
                                
               


                

                orig_slice = tensor_to_mlflow_image(img)
                recon_slice = tensor_to_mlflow_image(x_hat)
                
                comparison = np.concatenate([orig_slice, recon_slice], axis=1)
                comparison_img = Image.fromarray((comparison * 255).astype(np.uint8))
                
                # Save to temp file and log as artifact
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    comparison_img.save(tmp.name)
                    mlflow.log_artifact(tmp.name, artifact_path="recon")
                    os.remove(tmp.name)
                orig_slice = tensor_to_mlflow_image(img)
                recon_slice = tensor_to_mlflow_image(x_hat)
            

            if (epoch + 1) % 10 == 0 or (epoch + 1) == cfg.num_epochs:
                
                
                torch.save({
                        'epoch': epoch,
                        'encoder': encoder.state_dict(),
                        'decoder': decoder.state_dict(),
                        'optimizer_ae': optimizer_ae.state_dict(),
                        'loss': val_loss,
                    }, cfg.save_model_dir + f'/ae_checkpoint_{epoch + 1}.pth')
                # torch.save({
                #     'epoch': epoch + 1,
                #     'encoder': encoder.state_dict(),
                #     'optimizer_enc': optimizer_enc.state_dict(),
                #     'loss': loss,
                # }, cfg.save_model_dir + f'/encoder_{epoch + 1}.pth')

                # torch.save({
                #     'epoch': epoch + 1,
                #     'decoder': decoder.state_dict(),
                # }, cfg.save_model_dir + f'/decoder_{epoch + 1}.pth')

            if val_loss < best_loss:
                
                best_loss = val_loss
                torch.save({
                        'epoch': epoch,
                        'encoder': encoder.state_dict(),
                        'decoder': decoder.state_dict(),
                        'optimizer_ae': optimizer_ae.state_dict(),
                        'loss': val_loss,
                    }, cfg.save_model_dir + f'/ae_best.pth')

        
                # torch.save({
                #     'epoch': epoch + 1,
                #     'encoder': encoder.state_dict(),
                # }, cfg.save_model_dir + f'/encoder_best.pth')

                # torch.save({
                #     'epoch': epoch + 1,
                #     'decoder': decoder.state_dict(),
                # }, cfg.save_model_dir + f'/decoder_best.pth')
                # print(f'saved best model in epoch: {epoch+1}')
        
        train_writer.close()
        mlflow.end_run()
    
    except Exception as e:
        logger.error(f"Error during training epoch {epoch}: {e}")
        raise


def val(cfg, dataloader, encoder, decoder):
    encoder.eval()
    decoder.eval()

    #model_wrapper = AutoEncoderWrapper(encoder, decoder)
    # Use sw_batch_size=1 to be safe with memory
    #inferer = SlidingWindowInferer(roi_size=cfg.roi_size, sw_batch_size=1, overlap=0.5)
    
    ae_loss = nn.MSELoss()
    autoencoder_loss = 0.0

    with torch.no_grad():
        for data in dataloader:
            img = data['im'].to(cfg.device)
            # ==========forward=========
            z = encoder(img)
            x_hat = decoder(z)
            #x_hat = inferer(img, model_wrapper)
            
            # ==========compute the loss=========
            encoder_decoder_loss = ae_loss(x_hat, img)
            autoencoder_loss += encoder_decoder_loss.item()

        tol_loss = autoencoder_loss / len(dataloader)

    return tol_loss


