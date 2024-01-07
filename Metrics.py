import torch
import torch.nn.functional as F

def PSNR(img1, img2):
    mse = F.mse_loss(img1, img2)  # Calcul de l'erreur quadratique moyenne (MSE)
    max_val = torch.max(img1)     # Valeur maximale possible d'un pixel dans l'image
    psnr_val = 10 * torch.log10((max_val ** 2) / mse)
    return psnr_val.item()

