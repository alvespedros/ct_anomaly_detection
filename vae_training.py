import torch
from monai.networks.nets import VarAutoEncoder
from tqdm import trange
from dataset import get_dataloader
from utils import save_model
import mlflow
import torch.nn as nn
import os 
import torchvision


torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

# Referência: https://hunterheidenreich.com/posts/modern-variational-autoencoder-in-pytorch/

# VAE with reconstruction error (MSE - log likelihood under normal model)
# and KL divergence. kl_weight serves to avoid posterior colapse (high KL and low MSE)
def loss_function(recon_x, x, mu, logvar, kl_weight=1e-4):
    # 1. Reconstruction Loss (MSE)

    recon_loss = nn.functional.binary_cross_entropy(recon_x, x, reduction = 'sum') / (recon_x.size(0))
    #mse_loss = nn.MSELoss( reduction="sum")
    #recon_loss = mse_loss(recon_x, x,)

    logvar_clamped = torch.clamp(logvar, min=-25.0, max=5.0)
    # 2. KL Divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.sum(1 + logvar_clamped - mu.pow(2) - logvar_clamped.exp(), dim=-1)
    kl_loss = torch.mean(kl_loss)

    
    
    # 3. Total Loss
    total_loss = recon_loss + kl_weight * kl_loss
    #print(total_loss, recon_loss, kl_loss)
    return total_loss, recon_loss, kl_loss


# def log_likelihood_inv(x, mu, logvarinv, dims=(1, 2, 3)):
#     """Computes Gaussian log-likelihood using the inverse log-variance:

#     logvarinv = log(1 / sigma^2) = -log(sigma^2)
#     """
#     log_2pi = math.log(2.0 * math.pi)

#     # Component 1: -0.5 * sum(log(2*pi) - logvarinv)
#     ll_part1 = -0.5 * torch.sum(log_2pi - logvarinv, dim=dims)

#     # Component 2: -0.5 * sum((x - mu)^2 * exp(logvarinv))
#     ll_part2 = -0.5 * torch.sum(
#         torch.square(x - mu) * torch.exp(logvarinv), dim=dims
#     )

#     return ll_part1, ll_part2


# def loss_function(
#     recon_mean, recon_logvarinv, x, z_mean, z_logvar, kl_weight=1.0
# ):
#     """Replicates the loss() function from the TensorFlow implementation.

#     Args:
#         recon_mean: Predicted image mean, shape (B, C, H, W) recon_logvarinv:
#         Predicted inverse log-variance, shape (B, C, H, W) x: Ground truth
#         image, shape (B, C, H, W) z_mean: Latent mean, shape (B, C_latent, H',
#         W') or (B, D) z_logvar: Latent log-variance, same shape as z_mean
#         kl_weight: Scalar multiplier for the KL divergence term (beta)
#     """
#     # Determine reduction dimensions (all except batch dimension 0)
#     spatial_dims = tuple(range(1, recon_mean.ndim))
#     latent_dims = tuple(range(1, z_mean.ndim))

#     # 1. Latent KL divergence: 0.5 * sum(mu^2 + exp(logvar) - logvar - 1)
#     z_loss = 0.5 * torch.sum(
#         torch.square(z_mean) + torch.exp(z_logvar) - z_logvar - 1.0,
#         dim=latent_dims,
#     )
#     mean_z_loss = torch.mean(z_loss)

#     # 2. Negative Log-Likelihood Reconstruction Loss: -(ll1 + ll2)
#     ll1, ll2 = log_likelihood_inv(x, recon_mean, recon_logvarinv, dims=spatial_dims)
#     l2_loss = -1.0 * ll1 - 1.0 * ll2
#     mean_l2_loss = torch.mean(l2_loss)

#     # 3. Total VAE Loss
#     total_loss = mean_l2_loss + (kl_weight * mean_z_loss)

#     return total_loss, mean_l2_loss, mean_z_loss





def train(cfg):
    mlflow.end_run()
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.mlflow_experiment_name)

    with mlflow.start_run(run_name=cfg.mlflow_run_name):
        mlflow.set_tag("model_name", cfg.backbone)
        mlflow.log_params(vars(cfg))

    model = VarAutoEncoder(
        spatial_dims=2,  # 2 for 2D slice input (Conv2d / BatchNorm2d)
        in_shape=(1, 256, 256),  # (C, H, W) - Grayscale 256x256 slice
        out_channels=1,  # Reconstructed output channels
        latent_size=2048,  # Size of the flattened latent bottleneck (2 * 2 * 512 = 2048)
        channels=(16, 32, 64, 128, 256, 512),  # 6 residual stages
        strides=(2, 2, 2, 2, 2, 2),  # Stride of 2 at each stage
        kernel_size=3,
        up_kernel_size=3,
        num_res_units=2,  # Residual units per downsampling block
        act="LEAKYRELU",
        norm="BATCH",
        dropout=0.1,
    ).to(cfg.device)

    # Create optimiser
    optimizer = torch.optim.Adam(model.parameters(), cfg.learning_rate)

    avg_train_losses = []
    test_losses = []
    best_loss = 2000
    min_beta = 1e-7

    
    t = trange(cfg.num_epochs, leave=True, desc="epoch 0, average train loss: ?, test loss: ?")
    train_loader, val_loader = get_dataloader(cfg)
    for epoch in t:
        model.train()
        
        current_beta = min_beta + (cfg.target_beta - min_beta) * min(1.0, epoch / cfg.warmup_epochs)
        epoch_loss = 0
        epoch_recons_loss = 0
        epoch_kl_loss = 0

        
      
        for batch_data in train_loader:
             
            inputs = batch_data["im"].to(cfg.device)
            optimizer.zero_grad()

            recon_batch, mu, log_var, _ = model(inputs)
            loss, recons_loss, kl_loss = loss_function(recon_batch, inputs, mu, log_var, current_beta)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_recons_loss += recons_loss.item()
            epoch_kl_loss += kl_loss.item()



        
        mlflow.log_metric("Train average loss", epoch_loss / len(train_loader.dataset), step=epoch)
        mlflow.log_metric("Train reconstruction loss", epoch_recons_loss / len(train_loader.dataset), step=epoch)
        mlflow.log_metric("Train KL loss", epoch_kl_loss / len(train_loader.dataset), step=epoch)
        avg_train_losses.append(epoch_loss / len(train_loader.dataset))

        # Test
        model.eval()
        test_loss = 0
        test_recons_loss = 0 
        test_kl_loss = 0
        with torch.no_grad():
            for batch_data in val_loader:
                inputs = batch_data["im"].to(cfg.device)
                recon, mu, log_var, _ = model(inputs)
                # sum up batch loss
                test_epoch_loss, test_recons_lossep, test_kl_lossep = loss_function(recon, inputs, mu, log_var, current_beta)
                test_loss += test_epoch_loss.item()
                test_recons_loss += test_recons_lossep.item()
                test_kl_loss += test_kl_lossep.item()



            if epoch % 10 == 0:
                
                print('Saving reconst')
                
                local_filename = os.path.join(cfg.save_model_dir, f"reconstructions/recons_{epoch}.png")
                print(local_filename)
                # Salva o tensor diretamente no disco (garante formato 4D [B, C, H, W] ou 3D [C, H, W])
                torchvision.utils.save_image(recon[0:1], local_filename)
                
                # Faz o log no MLflow
                mlflow.log_artifact(local_filename, artifact_path="reconstructions")
                
                # Limpeza
                os.remove(local_filename)

        val_loss = test_loss / len(val_loader.dataset)
        val_recons_loss = test_recons_loss / len(val_loader.dataset)
        val_kl_loss = test_kl_loss / len(val_loader.dataset)
        
        
        best_loss = save_model(cfg, model, epoch, val_loss,best_loss)

        
        mlflow.log_metric("Val average loss", val_loss , step=epoch)
        mlflow.log_metric("Val average recons loss", val_recons_loss , step=epoch)
        mlflow.log_metric("Val average KL loss", val_kl_loss , step=epoch)
        test_losses.append(val_loss)

        t.set_description(  # noqa: B038
            f"epoch {epoch + 1}, average train loss: " f"{avg_train_losses[-1]:.4f}, test loss: {test_losses[-1]:.4f}"
        )
    return model, avg_train_losses, test_losses