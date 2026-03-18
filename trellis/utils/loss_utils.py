import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from lpips import LPIPS


def smooth_l1_loss(pred, target, beta=1.0):
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
    return loss.mean()

def l1_loss(network_output, gt, weight=None):
    if weight is None:
        return torch.abs((network_output - gt)).mean()
    else:
        return (torch.abs((network_output - gt)) * weight).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def psnr_loss(network_output, gt, max_val=1.0):
    mse = F.mse_loss(network_output, gt)
    return 20 * torch.log10(max_val / torch.sqrt(mse + 1e-8))

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def psnr(img1, img2, max_val=1.0):
    mse = F.mse_loss(img1, img2)
    return 20 * torch.log10(max_val / torch.sqrt(mse))


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


loss_fn_vgg = None
def lpips(img1, img2, value_range=(0, 1)):
    global loss_fn_vgg
    if loss_fn_vgg is None:
        loss_fn_vgg = LPIPS(net='vgg').cuda().eval()
    # normalize to [-1, 1]
    img1 = (img1 - value_range[0]) / (value_range[1] - value_range[0]) * 2 - 1
    img2 = (img2 - value_range[0]) / (value_range[1] - value_range[0]) * 2 - 1
    return loss_fn_vgg(img1, img2).mean()


def normal_angle(pred, gt):
    pred = pred * 2.0 - 1.0
    gt = gt * 2.0 - 1.0
    norms = pred.norm(dim=-1) * gt.norm(dim=-1)
    cos_sim = (pred * gt).sum(-1) / (norms + 1e-9)
    cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
    ang = torch.rad2deg(torch.acos(cos_sim[norms > 1e-9])).mean()
    if ang.isnan():
        return -1
    return ang

def cosine_loss_per_pixel(a, b):
    return 1 - (a * b).sum(dim=-3)

# -- masked reduce mean, 保证只考虑有效区域 --
def masked_mean(x, weight):
    tot = weight.sum()
    if tot < 1e-8:
        return x.sum() * 0  # mask全0不报错，返回0
    return (x * weight).sum() / tot

def gamma_correction(image, gamma=2.2):
    """
    Apply gamma correction to the image.
    
    :param image: Input image tensor.
    :param gamma: Gamma value for correction.
    :return: Gamma corrected image tensor.
    """
    return torch.pow(torch.clamp(image, 0, 1), 1 / gamma)

def get_reflectance_mask(roughness, metallic, method='physical', alpha=2.0, beta=1.0):
    """
    获取反射率mask，用于提高强反射区域在loss中的权重
    Args:
        roughness: 粗糙度 B 1 H W，范围 [0,1]，0表示完全光滑，1表示完全粗糙
        metallic: 金属度 B 1 H W，范围 [0,1]，0表示电介质，1表示金属
        method: 计算方法 ['simple', 'physical', 'adaptive']
        alpha: 粗糙度权重调节参数
        beta: 金属度权重调节参数
    Returns:
        torch.Tensor: 反射率mask，形状与输入相同
    """
    # 确保输入在合理范围内
    roughness = torch.clamp(roughness, 0.001, 0.999)  # 避免极值
    metallic = torch.clamp(metallic, 0.0, 1.0)
    
    if method == 'simple':
        # 简单线性组合：金属度高且粗糙度低的区域权重大
        mask = metallic * (1.0 - roughness)
        
    elif method == 'physical':
        # 基于物理的反射率计算
        # 对于金属：反射率主要由金属度决定
        # 对于电介质：反射率较低，但在掠射角会增加（这里简化处理）
        
        # 金属部分的反射率
        metallic_reflectance = metallic * (1.0 - roughness) ** alpha
        
        # 电介质部分的反射率（菲涅尔反射，这里用简化模型）
        dielectric_reflectance = (1.0 - metallic) * (1.0 - roughness) * 0.04  # 0.04是电介质的基础反射率
        
        mask = metallic_reflectance + dielectric_reflectance
        
    elif method == 'adaptive':
        # 自适应方法：根据场景动态调整权重
        # 使用非线性映射增强对比度
        smoothness = 1.0 - roughness
        
        # 对金属度和光滑度进行非线性变换
        enhanced_metallic = torch.pow(metallic, 1.0/beta)
        enhanced_smoothness = torch.pow(smoothness, alpha)
        
        # 组合两个因素
        mask = enhanced_metallic * enhanced_smoothness
        
        # 使用sigmoid增强对比度
        mask = torch.sigmoid((mask - 0.5) * 6.0)  # 6.0控制sigmoid的陡峭程度
        
    else:
        raise ValueError(f"Unsupported method: {method}")
    
    # 归一化到 [0, 1] 范围
    mask = torch.clamp(mask, 0.0, 1.0)
    
    # 可选：对mask进行轻微平滑以避免过于尖锐的边界
    # mask = F.gaussian_blur(mask, kernel_size=3, sigma=0.5)
    
    return mask

