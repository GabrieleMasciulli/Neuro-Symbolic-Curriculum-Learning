# Generative module

import torch
import torch.nn.functional as F


def conditional_gen(model, pC=None):
    """Conditional generation

    Args:
        model: network
        pc (default=None): probability of concepts

    Returns:
        out: decoder output
    """
    num_chunks = model.n_images * len(model.c_split)
    z_dim_chunk = model.decoder.latent_dim // num_chunks

    zs = torch.randn((8, num_chunks, z_dim_chunk), device=model.device)

    # select whether generate at random or not
    if pC is None:
        # Generate random concept probabilities
        # We assume model.encoder.c_dim is the dimension per chunk
        pC = 5 * torch.randn(
            (8, num_chunks, model.encoder.c_dim), device=model.device
        )
        # pC = torch.softmax(pC, dim=-1)

    latents = []
    chunk_idx = 0
    for _ in range(model.n_images):
        for i in range(len(model.c_split)):
            latents.append(zs[:, chunk_idx, :])
            latents.append(F.gumbel_softmax(pC[:, chunk_idx, :], tau=1, hard=True, dim=-1))
            chunk_idx += 1

    # generated images
    decode = model.decoder(torch.cat(latents, dim=1)).detach()

    return decode


def recon_visualization(out_dict):
    """Recon visualization method

    Args:
        out_dict: output dictionary

    Returns:
        out: images and recons concatenated
    """
    images = out_dict["INPUTS"].detach()[:8]
    recons = out_dict["RECS"].detach()[:8]
    return torch.cat([images, recons], dim=0)
