import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Callable, List
from PIL import Image


class VideoCaptionDataset(Dataset):
    """
    Dataset which will result in (video_tensor, caption) pairs.
    Expected format:
        root/
            sample00/
                frames/
                    frame_000.jpg
                    frame_001.jpg
                caption.txt
            sample01/
                ...
    """
    def __init__(self, root: str, transforms: Optional[Callable] = None, max_frames: int = 120, temporal_stride: int = 1) -> None:
        self.root = Path(root)
        self.transforms = transforms
        self.max_frames = max_frames
        self.temporal_stride = temporal_stride

        self.samples: List[Path] = sorted([
            d for d in self.root.iterdir()
            if d.is_dir() and (d / "frames").is_dir() and (d / "caption.txt").is_file()
        ])
    def __len__(self) -> int:
        return len(self.samples)
    def _load_frames(self, frame_dir: Path) -> torch.Tensor:
        frame_files = sorted([
            f for f in frame_dir.iterdir()
            if f.suffix.lower() in (".jpg", ".jpeg", ".png")
        ])
        # apply temporal stride and select max frames
        frame_files = frame_files[::self.temporal_stride]
        frame_files = frame_files[:self.max_frames]

        frames: List[torch.Tensor] = []
        for f in frame_files:
            image = Image.open(f).convert("RGB")
            if self.transforms is not None:
                image = self.transforms(image)
            else:
                image = torch.tensor(
                    list(image.getdata()),
                    dtype=torch.float32,
                ).reshape(image.size[1], image.size[0], 3).permute(2, 0, 1) # typical C, H, W reshape.
            frames.append(image)
        
        return torch.stack(frames, dim = 0)
    
    def __getitem__(self, index):
        sample_dir = self.samples[index]
        video = self._load_frames(sample_dir / "frames")
        caption = (sample_dir / "caption.txt").read_text().strip()
        return {
            "video": video,
            "caption": caption,
        }