
"""Generation of new images with a trained VAE."""

import torch


def sample(model, num_samples, device):
    """Generate new images from a trained VAE.

    Args:
        model: the VAE, already on device.
        num_samples: number of images to generate.
        device: torch.device to compute on.

    Returns:
        generated images, shape (num_samples, 1, 32, 32).
    """

    # 1. 모델을 평가 모드로 변경
    model.eval()

    # 2. 불필요한 기울기 계산 방지
    with torch.no_grad():

        # 3. 표준정규분포에서 난수 생성
        z = torch.randn(
            num_samples,
            model.latent_dim,
            device=device
        )

        # 4. 잠재변수에서 이미지 생성
        generated_images = model.decode(z)

    return generated_images
