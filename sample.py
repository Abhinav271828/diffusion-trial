"""
Sample images from a trained DiT model checkpoint.
"""
import torch
import argparse
import os
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL

from model import DiT_models
from utils import create_diffusion


def main(args):
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load checkpoint:
    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    train_args = checkpoint["args"]

    # Reconstruct model from saved training args:
    latent_size = train_args.image_size // 8
    model = DiT_models[train_args.model](
        input_size=latent_size,
        num_classes=train_args.num_classes,
    ).to(device)

    # Load EMA weights (preferred for generation) or raw model weights:
    if args.use_ema:
        model.load_state_dict(checkpoint["ema"])
    else:
        model.load_state_dict(checkpoint["model"])
    model.eval()

    # Create diffusion with the requested number of sampling steps:
    diffusion = create_diffusion(timestep_respacing=str(args.num_sampling_steps))

    # Load VAE decoder:
    vae = AutoencoderKL.from_pretrained(
        f"stabilityai/sd-vae-ft-{train_args.vae}"
    ).to(device)

    # Prepare class labels for sampling:
    if args.class_labels is not None:
        class_labels = args.class_labels
    else:
        class_labels = list(range(args.num_samples))

    # Expand or truncate labels to match num_samples:
    n = args.num_samples
    if len(class_labels) < n:
        class_labels = (class_labels * ((n // len(class_labels)) + 1))[:n]
    else:
        class_labels = class_labels[:n]

    # Sample in batches:
    all_samples = []
    for i in range(0, n, args.per_proc_batch_size):
        batch_labels = class_labels[i : i + args.per_proc_batch_size]
        batch_size = len(batch_labels)
        z = torch.randn(
            batch_size, 4, latent_size, latent_size, device=device
        )
        y = torch.tensor(batch_labels, device=device)

        if args.cfg_scale > 1.0:
            # Classifier-free guidance: double the batch for cond/uncond
            z = torch.cat([z, z], dim=0)
            y_null = (
                torch.tensor([train_args.num_classes] * batch_size, device=device)
            )
            y = torch.cat([y, y_null], dim=0)
            model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
            sample_fn = model.forward_with_cfg
        else:
            model_kwargs = dict(y=y)
            sample_fn = model.forward

        # Run the reverse diffusion process:
        samples = diffusion.p_sample_loop(
            sample_fn,
            z.shape,
            z,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=True,
            device=device,
        )

        if args.cfg_scale > 1.0:
            samples, _ = samples.chunk(2, dim=0)  # Keep only the guided half

        # Decode latents to pixel space:
        samples = vae.decode(samples / 0.18215).sample
        all_samples.append(samples)

    all_samples = torch.cat(all_samples, dim=0)

    # Save images:
    os.makedirs(args.output_dir, exist_ok=True)
    if args.save_grid:
        nrow = int(n ** 0.5)
        save_image(
            all_samples,
            os.path.join(args.output_dir, "grid.png"),
            nrow=nrow,
            normalize=True,
            value_range=(-1, 1),
        )
        print(f"Saved image grid to {args.output_dir}/grid.png")

    if args.save_individual:
        for i, img in enumerate(all_samples):
            save_image(
                img,
                os.path.join(args.output_dir, f"{i:04d}.png"),
                normalize=True,
                value_range=(-1, 1),
            )
        print(f"Saved {n} individual images to {args.output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample images from a trained DiT model.")
    parser.add_argument(
        "--ckpt", type=str,
        default="results/000-DiT-XL-2/checkpoints/0050000.pt",
        help="Path to the model checkpoint.",
    )
    parser.add_argument(
        "--num-samples", type=int, default=16,
        help="Number of images to generate.",
    )
    parser.add_argument(
        "--per-proc-batch-size", type=int, default=16,
        help="Batch size for sampling (reduce if OOM).",
    )
    parser.add_argument(
        "--class-labels", type=int, nargs="+", default=None,
        help="Class labels to condition on (e.g. --class-labels 0 1 2). "
             "Cycles to fill num-samples if fewer labels given. "
             "Defaults to 0..num_samples-1.",
    )
    parser.add_argument(
        "--cfg-scale", type=float, default=4.0,
        help="Classifier-free guidance scale. Set to 1.0 to disable.",
    )
    parser.add_argument(
        "--num-sampling-steps", type=int, default=250,
        help="Number of diffusion sampling steps.",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--use-ema", action="store_true", default=True,
        help="Use EMA model weights (default: True).",
    )
    parser.add_argument(
        "--no-ema", action="store_false", dest="use_ema",
        help="Use raw model weights instead of EMA.",
    )
    parser.add_argument(
        "--output-dir", type=str, default="samples",
        help="Directory to save generated images.",
    )
    parser.add_argument(
        "--save-grid", action="store_true", default=True,
        help="Save a single grid image (default: True).",
    )
    parser.add_argument(
        "--no-grid", action="store_false", dest="save_grid",
        help="Do not save a grid image.",
    )
    parser.add_argument(
        "--save-individual", action="store_true", default=False,
        help="Save each image individually.",
    )
    args = parser.parse_args()
    main(args)
