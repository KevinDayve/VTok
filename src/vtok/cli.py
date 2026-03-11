import argparse
import logging
import yaml
from .config import VTokConfig
from .train import train
from .utils import init_rng

def main() -> None:
    parser = argparse.ArgumentParser(description="VTok — Unified Video Tokeniser Training")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file.")
    parser.add_argument("--data_root", type=str, required=True, help="Path to dataset root.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--max_norm", type=float, default=1.0)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--checkpoint_interval", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    # Config specific args
    parser.add_argument("--backbone", type=str, default=None, choices=["vgg19", "clip"])
    parser.add_argument("--spatial_grid_size", type=int, default=None)
    parser.add_argument("--token_dim", type=int, default=None)
    parser.add_argument("--temporal_stride", type=int, default=None)
    parser.add_argument("--key_frame_index", type=int, default=None)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    if args.config is not None:
        with open(args.config, "r") as f:
            raw = yaml.safe_load(f) or {}
        cfg = VTokConfig(**raw)
    else:
        cfg = VTokConfig()

    if args.backbone is not None:
        cfg.backbone = args.backbone
    if args.spatial_grid_size is not None:
        cfg.spatial_grid_size = args.spatial_grid_size
    if args.token_dim is not None:
        cfg.token_dim = args.token_dim
    if args.temporal_stride is not None:
        cfg.temporal_stride = args.temporal_stride
    if args.key_frame_index is not None:
        cfg.key_frame_index = args.key_frame_index

    init_rng(seed=args.seed)

    train(
        config=cfg,
        root=args.data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        ema_decay=args.ema_decay,
        max_norm=args.max_norm,
        checkpoint_dir=args.checkpoint_dir,
        logging_interval=args.log_interval,
        checkpoint_interval=args.checkpoint_interval,
    )

if __name__ == "__main__":
    main()