import torch
import math

from diffusers_helper.k_diffusion.uni_pc_fm import sample_unipc
from diffusers_helper.k_diffusion.wrapper import fm_wrapper
from diffusers_helper.utils import repeat_to_batch_size

def flux_time_shift(t, mu=1.15, sigma=1.0):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

def calculate_flux_mu(context_length, x1=256, y1=0.5, x2=4096, y2=1.15, exp_max=7.0):
    k = (y2 - y1) / (x2 - x1)
    b = y1 - k * x1
    mu = k * context_length + b
    return min(mu, math.log(exp_max))

def get_flux_sigmas_from_mu(n, mu):
    sigmas = torch.linspace(1, 0, steps=n + 1)
    return flux_time_shift(sigmas, mu=mu)

@torch.inference_mode()
def sample_hunyuan(
        transformer,
        sampler='unipc',
        prompt=None,
        initial_latent=None,
        concat_latent=None,
        strength=1.0,
        width=512,
        height=512,
        frames=33,
        real_guidance_scale=3.0,
        distilled_guidance_scale=6.0,
        guidance_rescale=0.7,
        shift=None,
        num_inference_steps=25,
        batch_size=None,
        generator=None,
        prompt_embeds=None,
        prompt_embeds_mask=None,
        prompt_poolers=None,
        negative_prompt_embeds=None,
        negative_prompt_embeds_mask=None,
        negative_prompt_poolers=None,
        dtype=torch.float32,
        device=None,
        negative_kwargs=None,
        callback=None,
        **kwargs,
):
    
    device = device or transformer.device
    if generator is None or not hasattr(generator, 'device') or generator.device != device:
        generator = torch.Generator(device=device)

    if prompt is not None:
        print(f"Prompt: {prompt}")
        
    batch_size = batch_size or prompt_embeds.shape[0]

    latents = torch.randn(
        (batch_size, 16, (frames + 3) // 4, height // 8, width // 8),
        generator=generator,
        device=device,
        dtype=torch.float32
    )
    latents = torch.nan_to_num(latents, nan=0.0, posinf=0.0, neginf=0.0)

    mu = math.log(shift) if shift is not None else calculate_flux_mu(frames * height * width // 4, exp_max=7.0)
    sigmas = get_flux_sigmas_from_mu(num_inference_steps, mu).to(device=device)

    if initial_latent is not None:
        sigmas = sigmas * strength
        first_sigma = sigmas[0].to(dtype=torch.float32, device=device)
        initial_latent = initial_latent.to(dtype=torch.float32, device=device)
        latents = initial_latent * (1.0 - first_sigma) + latents * first_sigma
        latents = torch.nan_to_num(latents, nan=0.0, posinf=0.0, neginf=0.0)

    if concat_latent is not None:
        concat_latent = concat_latent.to(latents)

    # Guidance
    distilled_guidance = torch.tensor([distilled_guidance_scale * 1.0] * batch_size).to(device=device, dtype=dtype)

    # Repeat inputs to match batch size
    def repeat(x): return repeat_to_batch_size(x, batch_size)
    prompt_embeds = repeat(prompt_embeds)
    prompt_embeds_mask = repeat(prompt_embeds_mask)
    prompt_poolers = repeat(prompt_poolers)
    negative_prompt_embeds = repeat(negative_prompt_embeds)
    negative_prompt_embeds_mask = repeat(negative_prompt_embeds_mask)
    negative_prompt_poolers = repeat(negative_prompt_poolers)
    concat_latent = repeat(concat_latent)

    log_tensor_stats("prompt_embeds", prompt_embeds)
    log_tensor_stats("prompt_poolers", prompt_poolers)
    log_tensor_stats("negative_prompt_embeds", negative_prompt_embeds)
    log_tensor_stats("negative_prompt_poolers", negative_prompt_poolers)
    print(f"Prompt Embeds sum: {prompt_embeds.sum().item()}")

    # 🔐 Sanitize prompt data
    prompt_embeds = torch.nan_to_num(prompt_embeds, nan=0.0, posinf=0.0, neginf=0.0)
    prompt_poolers = torch.nan_to_num(prompt_poolers, nan=0.0, posinf=0.0, neginf=0.0)
    negative_prompt_embeds = torch.nan_to_num(negative_prompt_embeds, nan=0.0, posinf=0.0, neginf=0.0)
    negative_prompt_poolers = torch.nan_to_num(negative_prompt_poolers, nan=0.0, posinf=0.0, neginf=0.0)

    extra_args = dict(
        dtype=dtype,
        cfg_scale=real_guidance_scale,
        cfg_rescale=guidance_rescale,
        concat_latent=concat_latent,
        positive=dict(
            pooled_projections=prompt_poolers,
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=prompt_embeds_mask,
            guidance=distilled_guidance,
            **kwargs,
        ),
        negative=dict(
            pooled_projections=negative_prompt_poolers,
            encoder_hidden_states=negative_prompt_embeds,
            encoder_attention_mask=negative_prompt_embeds_mask,
            guidance=distilled_guidance,
            **(kwargs if negative_kwargs is None else {**kwargs, **negative_kwargs}),
        )
    )

    try:
        if torch.isnan(latents).any() or torch.isinf(latents).any():
            print("🔥 NaN/Inf detected in input latents. Applying fallback normalization.")
            latents = torch.nan_to_num(latents, nan=0.0, posinf=1.0, neginf=-1.0)
            latents = torch.clamp(latents, -3.0, 3.0)

        model = fm_wrapper(transformer)
        print("🧪 prompt_embeds stats:", prompt_embeds.shape, prompt_embeds.min().item(), prompt_embeds.max().item())
        results = sample_unipc(
            model, latents, sigmas,
            extra_args=extra_args,
            disable=False,
            callback=lambda *args, **cb_kwargs: safe_callback(callback, *args, **cb_kwargs)
        )

        if isinstance(results, torch.Tensor) and (torch.isnan(results).any() or torch.isinf(results).any()):
            print("🔥 Output latents are invalid. Applying final cleanup.")
            results = torch.nan_to_num(results, nan=0.0, posinf=1.0, neginf=-1.0)

    except Exception as e:
        print(f"⚠️ Sampling failed: {e}")
        results = latents  # Fallback to input

    generated_latents = torch.nan_to_num(results, nan=0.0, posinf=5.0, neginf=-5.0).clamp(-10.0, 10.0)
    print("Expected number of frames:", frames)
    print("Generated latent shape:", generated_latents.shape)
    return generated_latents

def safe_callback(callback_fn, *args, **kwargs):
    if callback_fn is None:
        return
    try:
        return callback_fn(*args, **kwargs)
    except Exception as e:
        print(f"[Preview Fallback] Callback failed: {e}")

def log_tensor_stats(name, tensor):
    if tensor is None:
        print(f"[{name}] is None")
    else:
        print(f"[{name}] shape={tensor.shape}, min={tensor.min().item()}, max={tensor.max().item()}, NaNs={torch.isnan(tensor).sum().item()}")