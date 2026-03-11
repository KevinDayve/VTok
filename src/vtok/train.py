import logging
from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from .config import VTokConfig
from .framework import UnifiedFramework
from .data.dataset import VideoCaptionDataset

logger = logging.getLogger(__name__)

class EMA:
    """
    Exponential moving average of model parameters.
    """
    def __init__(self, model: torch.nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = {}
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                self.shadow[name] = parameter.data.clone()

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for name, parameter in model.named_parameters():
            if parameter.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(parameter.data, alpha=1 - self.decay)

    def apply(self, model: torch.nn.Module) -> None:
        """
        Swap model parameters with EMA parameters.
        """
        self.backup = {}
        for name, parameter in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = parameter.data.clone()
                parameter.data.copy_(self.shadow[name])

    def restore(self, model: torch.nn.Module) -> None:
        """
        Restore original model parameters after inference.
        """
        for name, parameter in model.named_parameters():
            if name in self.backup:
                parameter.data.copy_(self.backup[name])
        self.backup = {}

def train(
        config: VTokConfig,
        root: str,
        epochs: int = 10,
        batch_size: int = 16,
        lr: float = 1e-5,
        ema_decay: float = 0.999,
        max_norm: float = 1.0,
        checkpoint_dir: str = "./checkpoints",
        logging_interval: int = 100,
        checkpoint_interval: int = 1,
) -> None:
    device = torch.device("cuda")
    model = UnifiedFramework(cfg=config).to(device=device)

    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_parameters, lr=lr)
    
    # count of frozen and trainable parameters.
    trainable_parameters_count: int = sum(p.numel() for p in trainable_parameters)
    frozen_parameters_count: int = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    logger.info(f'Trainable: {trainable_parameters_count}, Frozen parameters: {frozen_parameters_count}')

    ema = EMA(model=model, decay=ema_decay)
    dataset = VideoCaptionDataset(root=root)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True, # it will drop the last icomplete batch.
    )

    ckpt_path = Path(checkpoint_dir)
    ckpt_path.mkdir(parents=True, exist_ok=True)
    start_epoch = 0
    global_step = 0

    latest_ckpt = ckpt_path / "latest.pt"
    if latest_ckpt.exists():
        logger.info(f"Resuming from {latest_ckpt}")
        ckpt = torch.load(latest_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        ema.shadow = ckpt["ema_shadow"]
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["global_step"]
        del ckpt
        torch.cuda.empty_cache()
    

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0

        for batch in loader:
            video = batch['video'].to(device)
            caption = batch['caption']
            
            outputs = model(
                video=video,
                caption=caption,
            )
            loss = outputs['loss']
            if not loss.isfinite():
                logger.warning(f"Step {global_step}: non-finite loss. Skipping gradient flow")
                optimizer.zero_grad()
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=max_norm)
            optimizer.step()

            ema.update(model=model)
            epoch_loss += loss.item()
            epoch_steps += 1

            if global_step % logging_interval == 0:
                logger.info(
                    f"Epoch: {epoch} | Step: {global_step} "
                    f"Loss: {loss.item():.4f} "
                    f"Loss understanding: {outputs['loss_understanding'].item():.4f} "
                    f"Loss decoder: {outputs['loss_decoder'].item():.4f} "
                    f"Loss visual: {outputs['loss_visual'].item():.4f}"
                )
            global_step += 1
        
        average_loss = epoch_loss / max(epoch_steps, 1)
        logger.info(f"Epoch {epoch} complete | Avg loss: {average_loss:.4f}")

        if (epoch + 1) % checkpoint_interval == 0:
            ckpt_data = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "ema_shadow": ema.shadow,
                "epoch": epoch,
                "global_step": global_step,
            }
            torch.save(ckpt_data, ckpt_path / f"checkpoint_epoch_{epoch}.pt")
            torch.save(ckpt_data, latest_ckpt)
            logger.info(f"Checkpoint saved: epoch {epoch}")

    ema.apply(model)
    model.save_pretrained = None  # no HF save, just state dict
    torch.save(model.state_dict(), ckpt_path / "final_ema.pt")
    ema.restore(model)
    torch.save(model.state_dict(), ckpt_path / "final.pt")
    logger.info("Training complete. Final checkpoints saved.")