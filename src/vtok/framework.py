import torch
import torch.nn as nn
import torch.nn.functional as Functional
from typing import Tuple, Dict, List

from transformers import LlavaNextForConditionalGeneration, AutoTokenizer
from diffusers import HunyuanVideoTransformer3DModel, AutoencoderKL, DDIMScheduler
from .config import VTokConfig
from .tokeniser import VTokeniser
from .projection import VisualProjection

class UnifiedFramework(nn.Module):
    """
    Unified framework for video understanding and generation.
    Here, the trainable components would be the video tokeniser, the visual projector and the MLLP params.

    Frozen components would be the feature extractor, decoder and VAE.
    """
    def __init__(self, cfg: VTokConfig, model_id: str = "llava-hf/llama3-llava-next-8b-hf", video_decoder_id: str = "hunyuanvideo-community/HunyuanVideo", lambda_visual_lm: float = 1.0, lambda_decoder: float = 1.0) -> None:
        super().__init__()
        self.cfg = cfg
        self.lambda_visual_lm = lambda_visual_lm
        self.lambda_decoder = lambda_decoder
        self.video_tokeniser = VTokeniser(config=cfg)
        self.visualProj = VisualProjection(token_dimension=cfg.token_dim, model_dim=4096)

        # load the mllm,
        self.mllm = LlavaNextForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        self.mllm.config.output_hidden_states = True
        self.mllm_tokeniser = AutoTokenizer.from_pretrained(model_id)
        if self.mllm_tokeniser.pad_token is None:
            self.mllm_tokeniser.pad_token = self.mllm_tokeniser.eos_token

        # load the video-decoder
        self.decoder_transformer = HunyuanVideoTransformer3DModel.from_pretrained(
            video_decoder_id, subfolder="transformer", torch_dtype=torch.bfloat16,
        )
        self.decoder_vae = AutoencoderKL.from_pretrained(
            video_decoder_id, subfolder="vae", torch_dtype=torch.bfloat16,
        )
        self.scheduler = DDIMScheduler.from_pretrained(
            video_decoder_id, subfolder="scheduler",
        )
        self.decoder_transformer.eval()
        self.decoder_vae.eval()
        for p in self.decoder_transformer.parameters():
            p.requires_grad = False
        for p in self.decoder_vae.parameters():
            p.requires_grad = False

        # The paper mentions that LLava is compatible with Hunyuan, but in case it isn't we might
        # need to project MLLM's hidden states into decoder's space.
        self.mllm_to_dit_proj = None
        dit_dim = self.decoder_transformer.config.cross_attention_dim
        if dit_dim != 4096:
            self.mllm_to_dit_proj = nn.Linear(4096, dit_dim).to(torch.bfloat16)

    def _get_text_embeddings(self, text: str | List[str], device: torch.device | str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenise the text and return the embeddings along with token IDs.
        """
        if isinstance(text, list):
            tokens = self.mllm_tokeniser(text, return_tensors="pt", padding=True, truncation=True)
        else:
            tokens = self.mllm_tokeniser(text, return_tensors="pt", padding=True)
        tokens = {k: v.to(device) for k, v in tokens.items()}
        ids = tokens["input_ids"]
        embeddings = self.mllm.language_model.get_input_embeddings()(ids)
        return embeddings, ids
    
    def forward_understanding(self, video: torch.Tensor, text_prompt: str | List[str], target_text: str | List[str]) -> torch.Tensor:
        """
        The understanding part of the video, utilising the MLLM language backbone.
        A typical sequence looks like:
            [visual embeddings | prompt embeddings | target embeddings]
        """
        # tokenise the video
        video_tokens = self.video_tokeniser(video=video, key_frame_index=self.cfg.key_frame_index)
        video_embeddings = self.visualProj(video_tokens).to(torch.bfloat16)
        # get the embeddings for the prompt, and the target.
        prompt_embeddings, prompt_ids = self._get_text_embeddings(text=text_prompt, device=self.cfg.device)
        target_embeddings, target_ids = self._get_text_embeddings(text=target_text, device=self.cfg.device)

        # concatenate along the sequence dimension.
        embeddings = torch.cat([video_embeddings, prompt_embeddings, target_embeddings], dim=1)

        # construct labels tensor.
        video_labels = torch.full((video_embeddings.shape[0], video_embeddings.shape[1]), fill_value=-100, dtype=torch.long, device=self.cfg.device)
        prompt_labels = torch.full((prompt_embeddings.shape[0], prompt_embeddings.shape[1]), fill_value=-100, dtype=torch.long, device=self.cfg.device)

        target_ids = target_ids.long()
        labels = torch.cat([video_labels, prompt_labels, target_ids], dim = 1)
        assert labels.shape == embeddings.shape[:2], "Labels tensor is not of the same sequence length (or has different batch size than) as the embeddings."

        outputs = self.mllm.language_model(inputs_embeds=embeddings, labels=labels)
        # huggingface will compute cross-entropy internally.
        return outputs.loss
    
    def forward_generation(self, video: torch.Tensor, text_prompt: str | List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The generation part of the framework, which utilises the decoder.
        Here, the sequence looks like:
            [prompt embeddings | visual embeddings]
        """
        prompt_embeddings, prompt_ids = self._get_text_embeddings(text=text_prompt, device=self.cfg.device)
        _, promptlen, _ = prompt_embeddings.shape
        with torch.no_grad():
            video_tokens = self.video_tokeniser(video=video, key_frame_index=self.cfg.key_frame_index)
            video_embeddings = self.visualProj(video_tokens)
        _, vidlen, _ = video_embeddings.shape

        embeddings = torch.cat([prompt_embeddings, video_embeddings], dim=1)
        outputs = self.mllm.language_model(inputs_embeds=embeddings)
        hidden_states = outputs.hidden_states[-1][:, promptlen:promptlen+vidlen, :]
        prediction = hidden_states[:, :-1, :]
        target = video_embeddings[:, 1:, :]
        loss_visual_lm = Functional.mse_loss(prediction, target)

        # now the diffuser loss.
        video_vae = video.permute(0, 2, 1, 3, 4).to(torch.bfloat16)
        with torch.no_grad():
            # get the latent distrubtion q(z)
            posterior = self.decoder_vae.encode(video_vae).latent_dist
            latents = posterior.sample()
            latents = latents * self.decoder_vae.config.scaling_factor
        
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (bsz, ),
            device=latents.device,
        ).long()

        # add gaussian noise at indexed timesteps.
        noisedLatents = self.scheduler.add_noise(latents, noise, timesteps)

        # conditioned DiT
        conditioned_inputs = prediction
        if self.mllm_to_dit_proj is not None:
            conditioned_inputs = self.mllm_to_dit_proj(conditioned_inputs)

        noise_pred = self.decoder_transformer(
            hidden_states=noisedLatents,
            timesteps=timesteps,
            encoder_hidden_states=conditioned_inputs,
        ).sample
        loss_decoder = Functional.mse_loss(noise_pred, noise)
        return loss_visual_lm, loss_decoder
    
    def forward(self, video: torch.Tensor, caption: str | List[str]) -> Dict[str, torch.Tensor]:
        """
        Args:
            video: The video tensor
            caption: The string which serves as Ground truth for understanding, and as input for the generation phase.
        Returns:
            A dictionary containing losses.
        """
        if isinstance(caption, list):
            prompt: str | List[str] = ["Describe this video."] * len(caption)
        else:
            prompt = "Describe this video."

        loss_understanding = self.forward_understanding(
            video=video,
            text_prompt=prompt,
            target_text=caption,
        )
        loss_visual, loss_decoder = self.forward_generation(
            video=video,
            text_prompt=caption,
        )

        total_loss = loss_understanding + self.lambda_visual_lm * loss_visual + self.lambda_decoder * loss_decoder
        return {
            "loss": total_loss,
            "loss_understanding": loss_understanding,
            "loss_visual": loss_visual,
            "loss_decoder": loss_decoder
        }

class VTokFramework(UnifiedFramework):
    pass