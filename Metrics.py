import torch
import torch.nn.functional as F

def PSNR(img1, img2):
    mse = F.mse_loss(img1, img2)  # Calcul de l'erreur quadratique moyenne (MSE)
    max_val = torch.max(img1)     # Valeur maximale possible d'un pixel dans l'image
    psnr_val = 10 * torch.log10((max_val ** 2) / mse)
    return psnr_val.item()

# L'index de similarité structurelle (SSIM) est une mesure utilisée pour évaluer la similarité entre deux images.
def SSIM(img1, img2, C1=0.01, C2=0.03):
    # Mean of img1 and img2
    mu1 = img1.mean()
    mu2 = img2.mean()

    # Variance of img1 and img2
    sigma1_sq = ((img1 - mu1) * (img1 - mu1)).mean()
    sigma2_sq = ((img2 - mu2) * (img2 - mu2)).mean()

    # Covariance of img1 and img2
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()

    # Constants
    L = 1  # Dynamic range of pixel values
    C1 = (C1 * L) ** 2
    C2 = (C2 * L) ** 2

    # SSIM formula
    ssim_val = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_val.item()