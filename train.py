
import torch
import torch.nn as nn
import torch.optim as optim

from easydict import EasyDict as edict

from dataset import get_dataloader
from vae import Model, Encoder, Decoder
from monai.utils import set_determinism
from utils import loss_function

import os



# Logging
import logging
import mlflow
import psutil
from torch.utils.tensorboard import SummaryWriter

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


        if cfg.pretrained:

            print('Carregando rede pretreinada:')
            enc_path = 'reports/encoder_best_pre.pth'
            dec_path = 'reports/decoder_best_pre.pth'
            encoder = Encoder(cfg.roi_size[0], cfg.roi_size[1], cfg.roi_size[2], z_dim=512).to(cfg.device)
            decoder = Decoder(cfg.roi_size[0], cfg.roi_size[1], cfg.roi_size[2], z_dim=512).to(cfg.device)
            enc_state_dict = torch.load(enc_path, weights_only = True)['encoder']
            dec_state_dict = torch.load(dec_path, weights_only = True)['decoder']
            encoder.load_state_dict(enc_state_dict)
            decoder.load_state_dict(dec_state_dict)
        
        else:
            print('Carregando arquitetura')
            # Adicionado 08/01/2026 - teste de sliding window
            encoder = Encoder(input_dim=cfg.x_dim, hidden_dim=cfg.hidden_dim, latent_dim=cfg.latent_dim)
            decoder = Decoder(latent_dim=cfg.latent_dim, hidden_dim = cfg.hidden_dim, output_dim = cfg.x_dim)
            model = Model(encoder = encoder, decoder = decoder).to(cfg.device)

        # 3. Apply DataParallel if using multiple GPUs
        # if torch.cuda.device_count() > 1:
        #     encoder = nn.DataParallel(encoder).to(cfg.device)
        #     decoder = nn.DataParallel(decoder).to(cfg.device)


        BCE_loss = nn.BCELoss()

        optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

        #tensorboard_path, saved_model_path, log_path = form_results(f'{h}-{w}-{z}', z_dim)
        #writer = SummaryWriter(tensorboard_path)

        step = 0
        best_loss = 100


        for epoch in tqdm(range(cfg.num_epochs)):
            model.train()
            print("-" * 10)
            print(f"epoch {epoch + 1}/{cfg.num_epochs}")


            overall_loss  = 0.0

            for data in train_loader:
                img = data['im'].to(cfg.device)
                # ==========forward=========
                x_hat, mean, log_var = model(img)
                loss = loss_function(img, x_hat, mean, log_var)

                # ==========compute the loss and backpropagate=========

                overall_loss += loss.item()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # ========METRICS===========
                



                step += 1



            train_loss = overall_loss / len(train_loader)
            
            #train_writer.add_scalar("Overall loss", loss.item(), step)
            mlflow.log_metric("Average loss", train_loss, step=epoch)

            val_loss = val(cfg, val_loader, model)

            print('train_loss: {:.4f}'.format(train_loss))
            print('val_loss: {:.4f}'.format(val_loss))
            train_writer.add_scalars('train and val loss per epoch', {'train_loss': train_loss,
                                                                'val_loss': val_loss
                                                                }, epoch + 1)

            mlflow.log_metric("Model BCE Loss", val_loss, step=epoch)        

            #plot_2d_or_3d_image(img, epoch + 1, writer, index=0, frame_dim=-1, tag='image')
            #plot_2d_or_3d_image(x_hat, epoch + 1, writer, index=0, frame_dim=-1, tag='recon image')

            if (epoch + 1) % 50 == 0 or (epoch + 1) == cfg.num_epochs:
                torch.save({
                    'epoch': epoch + 1,
                    'model': model.state_dict(),
                }, cfg.save_model_dir + f'/encoder_{epoch + 1}.pth')


            if val_loss < best_loss:
                best_loss = val_loss
                torch.save({
                    'epoch': epoch + 1,
                    'model': model.state_dict(),
                }, cfg.save_model_dir + f'/encoder_best.pth')

                print(f'saved best model in epoch: {epoch+1}')
        
        train_writer.close()
        mlflow.end_run()
    
    except Exception as e:
        logger.error(f"Error during training epoch {epoch}: {e}")
        raise


def val(cfg, dataloader, model):
    model.eval()


    autoencoder_loss = 0.0

    with torch.no_grad():
        for data in dataloader:
            img = data['im'].to(cfg.device)
            # ==========forward=========
            #z = encoder(img)
            #x_hat = decoder(z)
            x_hat = model(img)
            
            # ==========compute the loss=========
            encoder_decoder_loss = loss_function(x_hat, img)
            autoencoder_loss += encoder_decoder_loss.item()

        tol_loss = autoencoder_loss / len(dataloader)

    return tol_loss


