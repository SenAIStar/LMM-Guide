from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class JanusConfig:
    model_id: str = "deepseek-ai/Janus-Pro-7B"
    revision: str = "PIN_A_REVIEWED_HF_COMMIT"
    dtype: str = "bfloat16"


class JanusBaseline:
    """Thin wrapper around the official Janus inference API.

    Heavy dependencies are imported lazily so policy, reward and evaluation tests
    remain runnable on CPU-only machines.
    """

    def __init__(self, config: JanusConfig):
        if config.revision.startswith("PIN_"):
            raise ValueError("pin a reviewed model revision before loading weights")
        import torch
        from janus.models import MultiModalityCausalLM, VLChatProcessor
        from transformers import AutoModelForCausalLM

        dtype = getattr(torch, config.dtype)
        self.torch = torch
        self.processor = VLChatProcessor.from_pretrained(config.model_id, revision=config.revision)
        self.tokenizer = self.processor.tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            revision=config.revision,
            trust_remote_code=True,
        ).to(dtype=dtype, device="cuda").eval()
        if not isinstance(self.model, MultiModalityCausalLM):
            raise TypeError("loaded model is not a Janus multimodality model")

    def understand(self, image_paths: Sequence[str | Path], instruction: str, max_new_tokens: int = 768) -> str:
        from janus.utils.io import load_pil_images

        conversation = [
            {
                "role": "User",
                "content": f"<image_placeholder>\n{instruction}",
                "images": [str(path) for path in image_paths],
            },
            {"role": "Assistant", "content": ""},
        ]
        images = load_pil_images(conversation)
        prepared = self.processor(conversations=conversation, images=images, force_batchify=True).to(self.model.device)
        inputs_embeds = self.model.prepare_inputs_embeds(**prepared)
        outputs = self.model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepared.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
        return self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)

    def generate(
        self,
        prompt: str,
        output_path: str | Path,
        temperature: float = 1.0,
        parallel_size: int = 4,
        cfg_weight: float = 5.0,
        image_size: int = 384,
        patch_size: int = 16,
    ) -> Path:
        import numpy as np
        from PIL import Image

        torch = self.torch
        conversation = [{"role": "User", "content": prompt}, {"role": "Assistant", "content": ""}]
        text = self.processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=self.processor.sft_format,
            system_prompt="",
        ) + self.processor.image_start_tag
        input_ids = self.tokenizer.encode(text)
        tokens = torch.zeros((parallel_size * 2, len(input_ids)), dtype=torch.int, device="cuda")
        for index in range(parallel_size * 2):
            tokens[index] = torch.tensor(input_ids, device="cuda")
            if index % 2 != 0:
                tokens[index, 1:-1] = self.processor.pad_id
        inputs_embeds = self.model.language_model.get_input_embeddings()(tokens)
        generated = torch.zeros(
            (parallel_size, image_size // patch_size * image_size // patch_size),
            dtype=torch.int,
            device="cuda",
        )
        past_key_values: Any = None
        for step in range(generated.shape[1]):
            outputs = self.model.language_model.model(
                inputs_embeds=inputs_embeds,
                use_cache=True,
                past_key_values=past_key_values,
            )
            past_key_values = outputs.past_key_values
            logits = self.model.gen_head(outputs.last_hidden_state[:, -1, :])
            conditional, unconditional = logits[0::2], logits[1::2]
            logits = unconditional + cfg_weight * (conditional - unconditional)
            probabilities = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            generated[:, step] = next_token.squeeze(-1)
            paired = torch.cat([next_token.unsqueeze(1), next_token.unsqueeze(1)], dim=1).reshape(-1)
            image_embeddings = self.model.prepare_gen_img_embeds(paired)
            inputs_embeds = image_embeddings.unsqueeze(1)
        decoded = self.model.gen_vision_model.decode_code(
            generated.to(dtype=torch.int),
            shape=[parallel_size, 8, image_size // patch_size, image_size // patch_size],
        )
        pixels = decoded.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
        pixels = np.clip((pixels + 1) / 2 * 255, 0, 255).astype(np.uint8)
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pixels[0]).save(destination)
        return destination

