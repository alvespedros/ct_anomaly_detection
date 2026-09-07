"""
map_restoration.py
===================

Reimplementação do algoritmo de detecção não-supervisionada de lesões via
restauração de imagem com prior normativo, proposto em:

    Chen, X., You, S., Tezcan, K.C., Konukoglu, E. (2020).
    "Unsupervised lesion detection via image restoration with a normative
    prior." Medical Image Analysis, 64, 101713.

O prior normativo P(X) é aproximado pelo ELBO de um VAE (aqui, um
`monai.networks.nets.VarAutoEncoder`), e a restauração é feita via MAP
(Eq. 3 e 6 do artigo):

    X_hat = argmax_X [ -lambda * ||X - Y||_TV + ELBO(X) ]

resolvido por ascensão de gradiente diretamente sobre a imagem X
(inicializada em X^0 = Y), com o modelo (encoder/decoder) mantido FIXO.

Este módulo assume:
- Imagens 2D em tensores shape (B, C, H, W), já normalizadas do mesmo jeito
  que os dados usados para treinar o VAE (o artigo usa z-score + background
  fixo em -3.5, mas qualquer normalização consistente serve).
- `monai.networks.nets.VarAutoEncoder` já treinado (pesos carregados, em
  eval() e com requires_grad_(False) nos parâmetros).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

# ---------------------------------------------------------------------------
# 1. ELBO do VAE (prior normativo), Eq. (1)-(2) da Seção 2.1
# ---------------------------------------------------------------------------

def _vae_forward(model: torch.nn.Module, x: Tensor):
    """
    Roda o encoder/decoder do VarAutoEncoder do MONAI e retorna
    (x_recon, mu, logvar).

    IMPORTANTE: não usamos `model(x)` (forward completo) porque o método
    interno `VarAutoEncoder.reparameterize` do MONAI faz:

        std = torch.exp(0.5 * logvar)
        if self.training:
            std = torch.randn_like(std).mul(std)
        return std.add_(mu)          # <- in-place!

    Em `model.eval()` (self.training=False), `std` é literalmente a saída
    de `torch.exp(...)`, e `std.add_(mu)` a modifica IN-PLACE. Isso corrompe
    o valor que o autograd guardou para o backward do `exp()`, causando:
    "one of the variables needed for gradient computation has been
    modified by an inplace operation ... ExpBackward0 ...".

    Por isso, chamamos `encode_forward`/`decode_forward` separadamente e
    reimplementamos a reparametrização nós mesmos, inteiramente
    out-of-place (sem nenhum método `_` do PyTorch).
    """
    mu, logvar = model.encode_forward(x)

    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    z = eps * std + mu  # totalmente out-of-place, seguro para autograd

    use_sigmoid = getattr(model, "use_sigmoid", True)
    x_recon = model.decode_forward(z, use_sigmoid)

    return x_recon, mu, logvar


def compute_elbo(model, x, sigma_x2=0.5, beta=1e-7, reduction="sum"):
    
    x_recon, mu, logvar = _vae_forward(model, x)

    logvar_clamped = torch.clamp(logvar, min=-25.0, max=5.0)
    mu_clamped = torch.clamp(mu, min=-10.0, max=10.0)
    
    sq_err = (x - x_recon) ** 2
    log_px_given_z = -(sq_err / (2 * sigma_x2))   # <- inalterado

    kl = -0.5 * (1 + logvar_clamped - mu_clamped.pow(2) - logvar_clamped.exp())

    if reduction == "sum":
        log_px_given_z = log_px_given_z.flatten(1).sum(dim=1)
        kl = kl.flatten(1).sum(dim=1)
    elif reduction == "mean":
        log_px_given_z = log_px_given_z.flatten(1).mean(dim=1)
        kl = kl.flatten(1).mean(dim=1)

    elbo = log_px_given_z + beta * kl   # <- só isso muda
    return elbo.sum()


# ---------------------------------------------------------------------------
# 2. Termo de consistência de dados: Total Variation (Eq. 6, Seção 2.2.1)
# ---------------------------------------------------------------------------

def total_variation(diff: Tensor) -> Tensor:
    """
    TV (anisotrópica) de um mapa de diferença D = X - Y, shape (B, C, H, W).
    Soma dos gradientes absolutos horizontais e verticais.
    """
    dh = diff[:, :, 1:, :] - diff[:, :, :-1, :]
    dw = diff[:, :, :, 1:] - diff[:, :, :, :-1]
    tv = dh.abs().flatten(1).sum(dim=1) + dw.abs().flatten(1).sum(dim=1)
    return tv.sum()  # escalar


# ---------------------------------------------------------------------------
# 3. Restauração MAP via ascensão de gradiente (Eq. 3-5)
# ---------------------------------------------------------------------------

@dataclass
class MAPRestorationConfig:
    n_iters: int = 1
    lr_phase1: float = 5e-3   # primeiras `phase1_iters` iterações
    lr_phase2: float = 3e-3   # iterações restantes
    phase1_iters: int = 100
    sigma_x2: float = 0.5
    clamp_range: Optional[tuple] = None  # ex.: (-3.5, valor_max) se quiser
    track_history_every: Optional[int] = None  # ex.: 50 -> guarda X a cada 50 passos


@torch.no_grad()
def _maybe_clamp(x: Tensor, clamp_range):
    if clamp_range is not None:
        x.clamp_(clamp_range[0], clamp_range[1])
    return x


def map_restore(
    model: torch.nn.Module,
    y: Tensor,
    lam: float,
    cfg: MAPRestorationConfig = MAPRestorationConfig(),
) -> dict:
    """
    Restaura Y para X_hat maximizando:

        log P(X|Y) ~= -lambda * TV(X - Y) + ELBO(X)

    via ascensão de gradiente, com X^0 = Y e o modelo (VAE) TOTALMENTE FIXO
    (só X é otimizado).

    Parameters
    ----------
    model : VarAutoEncoder treinado, em eval(), parâmetros congelados
    y     : imagem observada (com possível lesão), shape (B, C, H, W)
    lam   : peso lambda do termo de TV (ver `find_lambda` abaixo)
    cfg   : hiperparâmetros de otimização

    Returns
    -------
    dict com:
      "x_hat"   : imagem restaurada (mesma shape de y), tensor destacado do grafo
      "d_hat"   : mapa de diferença D_hat = Y - X_hat (lesão estimada, com sinal)
      "d_hat_abs": |D_hat| (usado no artigo para considerar hipo/hiperintensidade)
      "history" : lista de snapshots de X ao longo da otimização (se configurado)
    """
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    y = y.detach()
    # x precisa ser uma folha (leaf) do grafo de autograd, mas é RECRIADA a
    # cada iteração (em vez de atualizada in-place) — ver nota abaixo.
    x = y.clone().detach().requires_grad_(True)

    history = []

    for i in range(cfg.n_iters):
        lr = cfg.lr_phase1 if i < cfg.phase1_iters else cfg.lr_phase2

        elbo = compute_elbo(model, x, sigma_x2=cfg.sigma_x2, reduction="sum")
        print('ELBO: ', elbo)
        data_consistency = -lam * total_variation(x - y)
        #print('DC', data_consistency)
        objective = data_consistency + elbo  # Eq. (3)/(6): maximizar

        # Usamos torch.autograd.grad (em vez de objective.backward() +
        # x += lr * x.grad) por dois motivos:
        #   1. Evita acumular/zerar x.grad manualmente.
        #   2. Evita modificar x IN-PLACE. Modificar a folha `x` in-place
        #      (via `x += ...`) corrompe o contador de versão de buffers
        #      internos do VAE que dependem de x (ex.: `logvar.exp()` do
        #      reparameterization trick), causando o erro
        #      "modified by an inplace operation ... ExpBackward0".
        (grad_x,) = torch.autograd.grad(objective, x)

        with torch.no_grad():
            x_next = x + lr * grad_x  # ASCENSÃO de gradiente (Eq. 5)
            _maybe_clamp(x_next, cfg.clamp_range)

        # Nova folha a cada passo: desanexa do grafo anterior e volta a
        # exigir gradiente, sem reaproveitar (in-place) o armazenamento de x.
        x = x_next.detach().clone().requires_grad_(True)

        if cfg.track_history_every and (i % cfg.track_history_every == 0):
            history.append(x.detach().clone())

    x_hat = x.detach()
    #min_val, max_val = torch.aminmax(x_hat, dim=(-2, -1), keepdim=True)
    #x_hat = (x_hat - min_val) / (max_val - min_val).clamp_min(1e-8)
    d_hat = y - x_hat          # Eq.: D_hat = Y - X_hat
    d_hat_abs = d_hat.abs()

    return {
        "x_hat": x_hat,
        "d_hat": d_hat,
        "d_hat_abs": d_hat_abs,
        "history": history,
    }


# ---------------------------------------------------------------------------
# 4. Escolha automática de lambda (Seção 2.2.2, Eq. 7)
# ---------------------------------------------------------------------------

def _iter_batches(images_4d: Tensor, batch_size: Optional[int]):
    """
    Recebe um único tensor 4D (N, C, H, W) com todas as imagens saudáveis de
    validação e produz batches (para não estourar memória durante os 500
    passos de ascensão de gradiente). Se `batch_size` for None, processa
    tudo em um único batch.
    """
    if images_4d.dim() != 4:
        raise ValueError(
            f"healthy_val_images deve ser um tensor 4D (N, C, H, W); "
            f"recebido shape {tuple(images_4d.shape)}"
        )
    n = images_4d.shape[0]
    bs = batch_size or n
    for start in range(0, n, bs):
        yield images_4d[start:start + bs]


def find_lambda(
    model: torch.nn.Module,
    healthy_val_images: Tensor,
    lambda_grid: Iterable[float],
    cfg: MAPRestorationConfig = MAPRestorationConfig(),
    device: Optional[torch.device] = None,
    batch_size: Optional[int] = None,
) -> dict:
    """
    Implementa a heurística da Eq. (7):

        eps(lambda) = (1/S) * sum_s || Y_s - X_hat_{lambda,s} ||_1

    calculada sobre um pequeno conjunto de validação de imagens SAUDÁVEIS
    (sem lesão). O lambda escolhido é o menor que minimiza eps(lambda)
    (o artigo busca o "menor lambda que gera o menor eps").

    Parameters
    ----------
    model : VAE treinado e fixo
    healthy_val_images : tensor 4D (N, C, H, W) com N imagens saudáveis
    lambda_grid : valores de lambda a testar, ex.: torch.arange(1.0, 9.5, 0.5)
    cfg : configuração de otimização (mesma usada na restauração final)
    batch_size : tamanho do batch para processar `healthy_val_images`
        internamente (None = processa tudo de uma vez)

    Returns
    -------
    dict com "best_lambda", "eps_by_lambda" (lista paralela a lambda_grid)
    """
    if device is not None:
        healthy_val_images = healthy_val_images.to(device)

    eps_by_lambda = []

    for lam in lambda_grid:
        total_l1 = 0.0
        n_samples = 0
        for y in tqdm(_iter_batches(healthy_val_images, batch_size)):
            out = map_restore(model, y, lam=lam, cfg=cfg)
            l1 = (y - out["x_hat"]).abs().flatten(1).sum(dim=1)  # por amostra
            total_l1 += l1.sum().item()
            n_samples += y.shape[0]
        eps_by_lambda.append(total_l1 / max(n_samples, 1))

    best_idx = int(torch.tensor(eps_by_lambda).argmin())
    lambda_grid = list(lambda_grid)
    return {
        "best_lambda": lambda_grid[best_idx],
        "eps_by_lambda": eps_by_lambda,
        "lambda_grid": lambda_grid,
    }


# ---------------------------------------------------------------------------
# 5. Threshold binário limitando a Taxa de Falsos Positivos (Seção 2.2.3)
# ---------------------------------------------------------------------------

def find_threshold_for_fpr(
    model: torch.nn.Module,
    healthy_val_images: Tensor,
    lam: float,
    fpr_target: float,
    cfg: MAPRestorationConfig = MAPRestorationConfig(),
    n_thresholds: int = 200,
    device: Optional[torch.device] = None,
    batch_size: Optional[int] = None,
) -> float:
    """
    Determina o menor threshold T tal que a taxa de detecção (pixels com
    |D_hat| > T) nas imagens saudáveis de validação não ultrapasse
    `fpr_target` (ex.: 0.01, 0.05, 0.10), conforme Konukoglu et al. (2018)
    e Seção 2.2.3 do artigo.

    Parameters
    ----------
    healthy_val_images : tensor 4D (N, C, H, W) com imagens saudáveis
    batch_size : tamanho do batch para processar internamente
        (None = processa tudo de uma vez)

    Retorna o threshold T a ser aplicado em |D_hat| nas imagens de teste.
    """
    if device is not None:
        healthy_val_images = healthy_val_images.to(device)

    all_abs_diffs = []
    for y in _iter_batches(healthy_val_images, batch_size):
        out = map_restore(model, y, lam=lam, cfg=cfg)
        all_abs_diffs.append(out["d_hat_abs"].flatten())

    all_abs_diffs = torch.cat(all_abs_diffs)
    max_val = all_abs_diffs.max().item()
    thresholds = torch.linspace(0.0, max_val, n_thresholds)

    total_pixels = all_abs_diffs.numel()
    chosen_t = max_val  # fallback conservador
    for t in thresholds:
        fpr = (all_abs_diffs > t).float().sum().item() / total_pixels
        if fpr <= fpr_target:
            chosen_t = t.item()
            break  # thresholds crescentes -> primeiro que satisfaz é o menor

    return chosen_t


def binarize_detection(d_hat_abs: Tensor, threshold: float) -> Tensor:
    """Converte o mapa contínuo |D_hat| em máscara binária de lesão."""
    return (d_hat_abs > threshold).float()


# ---------------------------------------------------------------------------
# 6. Exemplo de uso ponta-a-ponta
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from monai.networks.nets import VarAutoEncoder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 1. Modelo já treinado (ajuste os hiperparâmetros conforme seu treino) ---
    model = VarAutoEncoder(
        spatial_dims=2,  
        in_shape=(1, 256, 256),  
        out_channels=1,  
        latent_size=2048,  
        channels=(16, 32, 64, 128, 256, 512), 
        strides=(2, 2, 2, 2, 2, 2),  
        kernel_size=3,
        up_kernel_size=3,
        num_res_units=2,  
        act="LEAKYRELU",
        norm="BATCH",
        dropout=0.1,
    ).to(device)
    
    model.load_state_dict(torch.load("reports/vae_training/model_20.pth")['model'])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # --- 2. Imagens de validação saudáveis: um único tensor 4D (N, C, H, W) ---
    # (fictício: substitua pelo seu conjunto real, ex. stack de um DataLoader)
    healthy_val_images = torch.randn(12, 1, 256, 256).to(device)

    cfg = MAPRestorationConfig(n_iters=500, lr_phase1=5e-3, lr_phase2=3e-3, phase1_iters=100)

    # --- 3. Escolher lambda automaticamente (Eq. 7) ---
    # `batch_size` controla quantas imagens são restauradas por vez dentro do
    # tensor 4D (útil para não estourar a memória da GPU); None = tudo junto.
    
    print('Procurando lambda para TV: ')
    lambda_search = find_lambda(
        model,
        healthy_val_images,
        lambda_grid= [1.0, 2.0], #[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        cfg=cfg,
        device=device,
        batch_size=4,
    )
    best_lambda = lambda_search["best_lambda"]
    print("Melhor lambda:", best_lambda)

    # --- 4. Determinar threshold para FPR alvo (ex.: 5%) ---
    print('Definindo threshold para FPR: ')
    # threshold_5 = find_threshold_for_fpr(
    #     model,
    #     healthy_val_images,
    #     lam=best_lambda,
    #     fpr_target=0.05,
    #     cfg=cfg,
    #     device=device,
    #     batch_size=4,
    # )
    # print("Threshold @5% FPR:", threshold_5)

    # --- 5. Restaurar uma imagem de teste (com possível lesão) e detectar ---
    y_test = torch.randn(1, 1, 256, 256).to(device)  # substitua pela imagem real
    result = map_restore(model, y_test, lam=best_lambda, cfg=cfg)
    #mask = binarize_detection(result["d_hat_abs"], threshold_5)

    print("x_hat shape:", result["x_hat"].shape)
    #print("máscara de lesão - pixels positivos:", mask.sum().item())