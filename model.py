"""
Architecture (Sign2GPT-style — chosen over full Uni-Sign because Uni-Sign's
pose-tokenization pretraining needs multi-lingual pose corpora you don't have;
this version needs only How2Sign clips + sentences):

    video clip (16 frames)
        -> VideoMAE encoder (pretrained, "MCG-NJU/videomae-base")
        -> patch/tubelet embeddings (B, N_patches, 768)
        -> learned-query cross-attention pooler ("visual mapper")
        -> K soft visual tokens (B, K, gpt2_hidden)
        -> prepended as a soft prompt to GPT-2's input embeddings
        -> GPT-2 causal LM generates the English sentence

This mirrors Sign2GPT's idea of mapping frozen visual features into the
LLM's embedding space instead of decoding through cross-attention layers,
which keeps the decoder a stock, easily-swappable HF causal LM.
"""

import torch
import torch.nn as nn
from transformers import VideoMAEModel, VideoMAEImageProcessor, GPT2LMHeadModel, GPT2TokenizerFast


class VisualMapper(nn.Module):
    """Perceiver-style pooler: a small set of learned query vectors
    cross-attend over the (many) VideoMAE patch tokens and compress them
    into K soft tokens in the decoder's embedding space."""

    def __init__(self, encoder_dim: int, decoder_dim: int, num_queries: int = 16, num_layers: int = 2, num_heads: int = 8):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_queries, decoder_dim) * 0.02)
        self.in_proj = nn.Linear(encoder_dim, decoder_dim)
        layer = nn.TransformerDecoderLayer(
            d_model=decoder_dim, nhead=num_heads, dim_feedforward=decoder_dim * 4,
            batch_first=True, activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.out_norm = nn.LayerNorm(decoder_dim)

    def forward(self, patch_embeds: torch.Tensor) -> torch.Tensor:
        # patch_embeds: (B, N, encoder_dim)
        B = patch_embeds.size(0)
        memory = self.in_proj(patch_embeds)                 # (B, N, decoder_dim)
        queries = self.queries.unsqueeze(0).expand(B, -1, -1)  # (B, K, decoder_dim)
        pooled = self.decoder(tgt=queries, memory=memory)   # (B, K, decoder_dim)
        return self.out_norm(pooled)


class SignTranslationModel(nn.Module):
    def __init__(
        self,
        videomae_name: str = "MCG-NJU/videomae-base",
        gpt2_name: str = "gpt2",
        num_visual_tokens: int = 16,
        freeze_encoder: bool = True,
    ):
        super().__init__()
        self.encoder = VideoMAEModel.from_pretrained(videomae_name)
        self.decoder = GPT2LMHeadModel.from_pretrained(gpt2_name)
        self.tokenizer = GPT2TokenizerFast.from_pretrained(gpt2_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        encoder_dim = self.encoder.config.hidden_size
        decoder_dim = self.decoder.config.n_embd
        self.mapper = VisualMapper(encoder_dim, decoder_dim, num_queries=num_visual_tokens)
        self.num_visual_tokens = num_visual_tokens

    def encode_video(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """pixel_values: (B, T, C, H, W) -> visual soft-tokens (B, K, decoder_dim)"""
        enc_out = self.encoder(pixel_values=pixel_values).last_hidden_state  # (B, N, encoder_dim)
        return self.mapper(enc_out)

    def forward(self, pixel_values, input_ids, attention_mask):
        visual_tokens = self.encode_video(pixel_values)               # (B, K, D)
        text_embeds = self.decoder.transformer.wte(input_ids)         # (B, L, D)
        inputs_embeds = torch.cat([visual_tokens, text_embeds], dim=1)  # (B, K+L, D)

        vis_mask = torch.ones(visual_tokens.shape[:2], dtype=attention_mask.dtype, device=attention_mask.device)
        full_attention_mask = torch.cat([vis_mask, attention_mask], dim=1)

        # Labels: ignore loss on the visual-prefix positions and on padding
        ignore = torch.full(visual_tokens.shape[:2], -100, dtype=input_ids.dtype, device=input_ids.device)
        labels = torch.cat([ignore, input_ids], dim=1)
        labels = labels.masked_fill(full_attention_mask == 0, -100)

        out = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            labels=labels,
        )
        return out  # out.loss, out.logits

    @torch.no_grad()
    def generate(self, pixel_values, max_new_tokens: int = 40, num_beams: int = 4):
        self.eval()
        visual_tokens = self.encode_video(pixel_values)  # (B, K, D)
        vis_mask = torch.ones(visual_tokens.shape[:2], dtype=torch.long, device=visual_tokens.device)

        # GPT2LMHeadModel.generate supports inputs_embeds directly.
        gen_ids = self.decoder.generate(
            inputs_embeds=visual_tokens,
            attention_mask=vis_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            no_repeat_ngram_size=3,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        return [self.tokenizer.decode(g, skip_special_tokens=True) for g in gen_ids]


def build_image_processor(videomae_name: str = "MCG-NJU/videomae-base") -> VideoMAEImageProcessor:
    return VideoMAEImageProcessor.from_pretrained(videomae_name)
