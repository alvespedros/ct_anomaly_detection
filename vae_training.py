
import torch
from monai.networks.nets import VarAutoEncoder
from tqdm import trange
from dataset import get_dataloader
from utils import save_model
import mlflow
import torch.nn as nn



BCELoss = torch.nn.BCELoss(reduction="sum")



def loss_function(recon_x, x):
  
    bce = nn.functional.binary_cross_entropy(recon_x, x, reduction="mean")

    return bce

# def loss_function(recon_x, x, mu, log_var, beta):
#     bce = BCELoss(recon_x, x)
#     kld = -0.5 * beta * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
#     return bce + kld


def train(cfg,in_shape, max_epochs, latent_size, learning_rate, beta):

    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.mlflow_experiment_name)

    with mlflow.start_run(run_name=cfg.mlflow_run_name):
        mlflow.set_tag("model_name", cfg.backbone)
        mlflow.log_params(vars(cfg))

    model  = VarAutoEncoder(
    spatial_dims=3,                     # 3 for volumetric 3D (Conv3d / BatchNorm3d)
    in_shape=(1, 128, 128, 128),        # Input spatial shape: (C, D, H, W)
    out_channels=1,                     # Output channels (1 for grayscale MRI/CT)
    latent_size=512,                    # Dimension of the 1D bottleneck latent vector
    channels=(16, 32, 64, 128, 256),    # Feature map channels across downsampling stages
    strides=(2, 2, 2, 2, 2),            # Stride of 2 at each stage for spatial downsampling
    kernel_size=3,
    up_kernel_size=3,
    num_res_units=2,                    # Number of residual units per block
    act="LEAKYRELU",
    norm="BATCH",
    dropout=0.1
).to(cfg.device)

    model = AutoEncoder(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(4,),
        strides=(2,),
        inter_channels=(8, 8, 8),
        inter_dilations=(1, 2, 4),
        num_inter_units=2
    )

    # Create optimiser
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)

    avg_train_losses = []
    test_losses = []
    best_loss = 2000
    t = trange(max_epochs, leave=True, desc="epoch 0, average train loss: ?, test loss: ?")
    train_loader, val_loader = get_dataloader(cfg)
    for epoch in t:
        model.train()
        epoch_loss = 0

        
        
        for batch_data in train_loader:
            inputs = batch_data["im"].to(cfg.device)
            optimizer.zero_grad()

            recon_batch, mu, log_var, _ = model(inputs)
            loss = loss_function(recon_batch, inputs, mu, log_var, beta)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        mlflow.log_metric("Train average loss", epoch_loss / len(train_loader.dataset), step=epoch)
        avg_train_losses.append(epoch_loss / len(train_loader.dataset))

        # Test
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch_data in val_loader:
                inputs = batch_data["im"].to(cfg.device)
                recon, mu, log_var, _ = model(inputs)
                # sum up batch loss
                test_loss += loss_function(recon, inputs, mu, log_var, beta).item()



        val_loss = test_loss / len(val_loader.dataset)
        
        
        best_loss = save_model(cfg, model, epoch, val_loss,best_loss)

        
        mlflow.log_metric("Val average loss", val_loss , step=epoch)
        test_losses.append(val_loss)

        t.set_description(  # noqa: B038
            f"epoch {epoch + 1}, average train loss: " f"{avg_train_losses[-1]:.4f}, test loss: {test_losses[-1]:.4f}"
        )
    return model, avg_train_losses, test_losses

