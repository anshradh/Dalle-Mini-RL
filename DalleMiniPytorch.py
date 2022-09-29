# %%
import os
from PIL import Image
import numpy
from torch import LongTensor, FloatTensor
import torch
import torch.nn as nn
import torch as t
import torch.backends.cudnn, torch.backends.cuda
import json
import requests
from typing import Iterator, Optional, Union
from min_dalle.min_dalle import TextTokenizer
from min_dalle.min_dalle import DalleBartEncoder, DalleBartDecoder, VQGanDetokenizer

MIN_DALLE_REPO = "https://huggingface.co/kuprel/min-dalle/resolve/main/"
IMAGE_TOKEN_COUNT = 256

# %%
class DalleMiniWithValueHead(nn.Module):
    def __init__(
        self,
        models_root: str = "pretrained",
        dtype: torch.dtype = torch.float32,
        device: Optional[str] = None,
        is_mega: bool = True,
        is_reusable: bool = True,
        is_verbose=True,
    ):
        super().__init__()
        if device == None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if is_verbose:
            print("using device", device)
        self.device = device
        self.is_mega = is_mega
        self.is_reusable = is_reusable
        self.dtype = dtype
        self.is_verbose = is_verbose
        self.text_token_count = 64
        self.layer_count = 24 if is_mega else 12
        self.attention_head_count = 32 if is_mega else 16
        self.embed_count = 2048 if is_mega else 1024
        self.glu_embed_count = 4096 if is_mega else 2730
        self.text_vocab_count = 50272 if is_mega else 50264
        self.image_vocab_count = 16415 if is_mega else 16384

        self.image_token_count = IMAGE_TOKEN_COUNT

        model_name = "dalle_bart_{}".format("mega" if is_mega else "mini")
        dalle_path = os.path.join(models_root, model_name)
        vqgan_path = os.path.join(models_root, "vqgan")
        if not os.path.exists(dalle_path):
            os.makedirs(dalle_path)
        if not os.path.exists(vqgan_path):
            os.makedirs(vqgan_path)
        self.vocab_path = os.path.join(dalle_path, "vocab.json")
        self.merges_path = os.path.join(dalle_path, "merges.txt")
        self.encoder_params_path = os.path.join(dalle_path, "encoder.pt")
        self.decoder_params_path = os.path.join(dalle_path, "decoder.pt")
        self.detoker_params_path = os.path.join(vqgan_path, "detoker.pt")

        self.init_tokenizer()
        if is_reusable:
            self.init_encoder()
            self.init_decoder()
            self.init_detokenizer()
        self.v_head = nn.Sequential(
            nn.Linear(self.embed_count, 1), nn.Tanh(), nn.Linear(1, 1)
        ).to(self.device)

    def download_tokenizer(self):
        if self.is_verbose:
            print("downloading tokenizer params")
        suffix = "" if self.is_mega else "_mini"
        _ = requests.get(MIN_DALLE_REPO + "config.json")  # trigger HF download
        vocab = requests.get(MIN_DALLE_REPO + "vocab{}.json".format(suffix))
        merges = requests.get(MIN_DALLE_REPO + "merges{}.txt".format(suffix))
        with open(self.vocab_path, "wb") as f:
            f.write(vocab.content)
        with open(self.merges_path, "wb") as f:
            f.write(merges.content)

    def download_encoder(self):
        if self.is_verbose:
            print("downloading encoder params")
        suffix = "" if self.is_mega else "_mini"
        params = requests.get(MIN_DALLE_REPO + "encoder{}.pt".format(suffix))
        with open(self.encoder_params_path, "wb") as f:
            f.write(params.content)

    def download_decoder(self):
        if self.is_verbose:
            print("downloading decoder params")
        suffix = "" if self.is_mega else "_mini"
        params = requests.get(MIN_DALLE_REPO + "decoder{}.pt".format(suffix))
        with open(self.decoder_params_path, "wb") as f:
            f.write(params.content)

    def download_detokenizer(self):
        if self.is_verbose:
            print("downloading detokenizer params")
        params = requests.get(MIN_DALLE_REPO + "detoker.pt")
        with open(self.detoker_params_path, "wb") as f:
            f.write(params.content)

    def init_tokenizer(self):
        _ = requests.get(MIN_DALLE_REPO + "config.json")  # trigger HF download
        is_downloaded = os.path.exists(self.vocab_path)
        is_downloaded &= os.path.exists(self.merges_path)
        if not is_downloaded:
            self.download_tokenizer()
        if self.is_verbose:
            print("intializing TextTokenizer")
        with open(self.vocab_path, "r", encoding="utf8") as f:
            vocab = json.load(f)
        with open(self.merges_path, "r", encoding="utf8") as f:
            merges = f.read().split("\n")[1:-1]
        self.tokenizer = TextTokenizer(vocab, merges)

    def init_encoder(self):
        is_downloaded = os.path.exists(self.encoder_params_path)
        if not is_downloaded:
            self.download_encoder()
        if self.is_verbose:
            print("initializing DalleBartEncoder")
        self.encoder = (
            DalleBartEncoder(
                attention_head_count=self.attention_head_count,
                embed_count=self.embed_count,
                glu_embed_count=self.glu_embed_count,
                text_token_count=self.text_token_count,
                text_vocab_count=self.text_vocab_count,
                layer_count=self.layer_count,
                device=self.device,
            )
            .to(self.dtype)
            .eval()
        )
        params = torch.load(self.encoder_params_path)
        self.encoder.load_state_dict(params, strict=False)
        del params
        self.encoder = self.encoder.to(device=self.device)  # type: ignore

    def init_decoder(self):
        is_downloaded = os.path.exists(self.decoder_params_path)
        if not is_downloaded:
            self.download_decoder()
        if self.is_verbose:
            print("initializing DalleBartDecoder")
        self.decoder = (
            DalleBartDecoder(
                image_vocab_count=self.image_vocab_count,
                attention_head_count=self.attention_head_count,
                embed_count=self.embed_count,
                glu_embed_count=self.glu_embed_count,
                layer_count=self.layer_count,
                device=self.device,
            )
            .to(self.dtype)
            .eval()
        )
        params = torch.load(self.decoder_params_path)
        self.decoder.load_state_dict(params, strict=False)
        del params
        self.decoder = self.decoder.to(device=self.device)  # type: ignore

    def init_detokenizer(self):
        is_downloaded = os.path.exists(self.detoker_params_path)
        if not is_downloaded:
            self.download_detokenizer()
        if self.is_verbose:
            print("initializing VQGanDetokenizer")
        self.detokenizer = VQGanDetokenizer().eval()
        params = torch.load(self.detoker_params_path)
        self.detokenizer.load_state_dict(params)
        del params
        self.detokenizer = self.detokenizer.to(device=self.device)  # type: ignore

    def image_grid_from_tokens(
        self, image_tokens: LongTensor, is_seamless: bool, is_verbose: bool = False
    ) -> FloatTensor:
        if not self.is_reusable:
            del self.decoder
        torch.cuda.empty_cache()
        if not self.is_reusable:
            self.init_detokenizer()
        if is_verbose:
            print("detokenizing image")
        images = self.detokenizer.forward(is_seamless, image_tokens)
        if not self.is_reusable:
            del self.detokenizer
        return images

    def generate_raw_image_stream(
        self,
        text: str,
        seed: int,
        grid_size: int,
        progressive_outputs: bool = False,
        is_seamless: bool = False,
        temperature: float = 1,
        top_k: int = 256,
        supercondition_factor: int = 16,
        is_verbose: bool = False,
    ) -> Iterator[FloatTensor]:
        image_count = grid_size ** 2
        if is_verbose:
            print("tokenizing text")
        tokens = self.tokenizer.tokenize(text, is_verbose=is_verbose)
        if len(tokens) > self.text_token_count:
            tokens = tokens[: self.text_token_count]
        if is_verbose:
            print("{} text tokens".format(len(tokens)), tokens)
        text_tokens = numpy.ones((2, 64), dtype=numpy.int32)
        text_tokens[0, :2] = [tokens[0], tokens[-1]]
        text_tokens[1, : len(tokens)] = tokens
        text_tokens = torch.tensor(text_tokens, dtype=torch.long, device=self.device)

        if not self.is_reusable:
            self.init_encoder()
        if is_verbose:
            print("encoding text tokens")
        with torch.cuda.amp.autocast(dtype=self.dtype):
            encoder_state = self.encoder.forward(text_tokens)  # type: ignore
        if not self.is_reusable:
            del self.encoder
        torch.cuda.empty_cache()

        if not self.is_reusable:
            self.init_decoder()

        with torch.cuda.amp.autocast(dtype=self.dtype):
            expanded_indices = [0] * image_count + [1] * image_count
            text_tokens = text_tokens[expanded_indices]
            encoder_state = encoder_state[expanded_indices]
            attention_mask = text_tokens.not_equal(1)[:, None, None, :]
            attention_state = torch.zeros(
                size=(
                    self.layer_count,
                    image_count * 4,
                    IMAGE_TOKEN_COUNT,
                    self.embed_count,
                ),
                device=self.device,
            )
            image_tokens = torch.full(
                (image_count, IMAGE_TOKEN_COUNT + 1),
                2 ** 14 - 1,
                dtype=torch.long,
                device=self.device,
            )

            if seed > 0:
                torch.manual_seed(seed)

        token_indices = torch.arange(IMAGE_TOKEN_COUNT, device=self.device)
        settings = torch.tensor(
            [temperature, top_k, supercondition_factor],
            dtype=torch.float32,
            device=self.device,
        )
        for i in range(IMAGE_TOKEN_COUNT):
            torch.cuda.empty_cache()
            with torch.cuda.amp.autocast(dtype=self.dtype):
                image_tokens[:, i + 1], attention_state = self.decoder.sample_tokens(
                    settings=settings,
                    attention_mask=attention_mask,
                    encoder_state=encoder_state,
                    attention_state=attention_state,
                    prev_tokens=image_tokens[:, [i]],
                    token_index=token_indices[[i]],
                )

            with torch.cuda.amp.autocast(dtype=torch.float32):
                if ((i + 1) % 32 == 0 and progressive_outputs) or i + 1 == 256:
                    yield self.image_grid_from_tokens(
                        image_tokens=image_tokens[:, 1:], is_seamless=is_seamless, is_verbose=is_verbose  # type: ignore
                    )

    def generate_image_stream(self, *args, **kwargs) -> Iterator[Image.Image]:
        image_stream = self.generate_raw_image_stream(*args, **kwargs)
        for image in image_stream:
            image = image.to(torch.uint8).to("cpu").numpy()
            yield Image.fromarray(image)

    def generate_image(self, *args, **kwargs) -> Image.Image:
        image_stream = self.generate_image_stream(
            *args, **kwargs, progressive_outputs=False
        )
        return next(image_stream)

    def tokenize_for_forward(self, prompt: Union[str, list[str]]):
        if isinstance(prompt, str):
            prompt = [prompt]
        batched_text_tokens = []
        for p in prompt:
            assert isinstance(p, str)
            tokens = self.tokenizer.tokenize(p)
            if len(tokens) > 64:
                tokens = tokens[:64]
            text_tokens = numpy.ones((2, 64), dtype=numpy.int32)
            text_tokens[0, :2] = [tokens[0], tokens[-1]]
            text_tokens[1, : len(tokens)] = tokens
            text_tokens = torch.tensor(
                text_tokens, dtype=torch.long, device=self.device
            )
            batched_text_tokens.append(text_tokens)
        return t.stack(batched_text_tokens, dim=0)

    def decoder_forward_with_state(
        self, attention_mask, encoder_state, attention_state, prev_tokens, token_index
    ):
        image_count = encoder_state.shape[0] // 2
        token_index = token_index.unsqueeze(0).repeat(image_count * 2, 1)
        prev_tokens = prev_tokens.repeat(2, 1)
        decoder_state = self.decoder.embed_tokens.forward(prev_tokens)
        decoder_state += self.decoder.embed_positions.forward(token_index)
        decoder_state = self.decoder.layernorm_embedding.forward(decoder_state)
        for i in range(self.layer_count):
            decoder_state, attention_state[i] = self.decoder.layers[i].forward(
                decoder_state, encoder_state, attention_state[i], attention_mask, token_index  # type: ignore
            )
        decoder_state = self.decoder.final_ln(decoder_state)
        logits = self.decoder.lm_head(decoder_state)
        return logits, attention_state, decoder_state

    def decoder_sample_tokens_with_state(self, settings, **kwargs):
        orig_logits, attention_state, decoder_state = self.decoder_forward_with_state(
            **kwargs
        )
        logits = orig_logits
        image_count = logits.shape[0] // 2
        temperature = settings[[0]]
        top_k = settings[[1]].to(torch.long)
        supercondition_factor = settings[[2]]
        logits = logits[:, -1, : 2 ** 14]  # type: ignore
        logits: FloatTensor = (
            logits[:image_count] * (1 - supercondition_factor)
            + logits[image_count:] * supercondition_factor
        )
        logits_sorted, _ = logits.sort(descending=True)
        is_kept = logits >= logits_sorted[:, top_k - 1]
        logits -= logits_sorted[:, [0]]  # type: ignore
        logits /= temperature
        logits.exp_()
        logits *= is_kept.to(torch.float32)  # type: ignore
        image_tokens = torch.multinomial(logits, 1)[:, 0]
        return image_tokens, orig_logits, attention_state, decoder_state

    def forward(
        self,
        text_tokens_batched,
        seed=1,
        temperature: float = 1,
        top_k: int = 256,
        supercondition_factor: int = 16,
        use_grad: bool = True,
    ):
        with torch.set_grad_enabled(use_grad):
            prompt_count = text_tokens_batched.shape[0]
            assert (
                prompt_count <= 4
            )  # not sure if needed, going to be conservative with memory
            batched_logits = []
            batched_values = []
            batched_image_tokens = []
            for p in range(0, prompt_count):
                text_tokens = text_tokens_batched[p]
                encoder_state = self.encoder(text_tokens)
                expanded_indices = [0] * 1 + [1] * 1
                text_tokens = text_tokens[expanded_indices]
                encoder_state = encoder_state[expanded_indices]
                attention_mask = text_tokens.not_equal(1)[:, None, None, :]

                attention_state = torch.zeros(
                    size=(self.layer_count, 1 * 4, IMAGE_TOKEN_COUNT, self.embed_count),
                    device=self.device,
                    dtype=self.dtype,
                )
                image_tokens = torch.full(
                    (1, IMAGE_TOKEN_COUNT + 1),
                    2 ** 14 - 1,
                    dtype=torch.long,
                    device=self.device,
                )

                if seed > 0:
                    torch.manual_seed(seed)

                token_indices = torch.arange(IMAGE_TOKEN_COUNT, device=self.device)
                settings = torch.tensor(
                    [temperature, top_k, supercondition_factor],
                    dtype=self.dtype,
                    device=self.device,
                    requires_grad=True,
                )
                image_logits = []
                image_values = []
                for i in range(IMAGE_TOKEN_COUNT):
                    with torch.cuda.amp.autocast(dtype=self.dtype):
                        (
                            image_tokens[:, i + 1],
                            logits,
                            attention_state,
                            decoder_state,
                        ) = self.decoder_sample_tokens_with_state(
                            settings=settings,
                            attention_mask=attention_mask,
                            encoder_state=encoder_state,
                            attention_state=attention_state,
                            prev_tokens=image_tokens[:, [i]],
                            token_index=token_indices[[i]],
                        )
                        image_logits.append(logits[-1])
                        values = self.v_head(decoder_state.squeeze(-2))
                        image_values.append(values[-1])
                batched_image_tokens.append(image_tokens[0, 1:])
                batched_logits.append(t.cat(image_logits, 0))
                batched_values.append(t.cat(image_values, 0))

            stacked_batched_logits = t.stack(batched_logits, 0).to(torch.float32)
            stacked_batched_values = t.stack(batched_values, 0).to(torch.float32)
            stacked_batched_image_tokens = t.stack(batched_image_tokens, 0)

        return (
            stacked_batched_logits,
            stacked_batched_values,
            stacked_batched_image_tokens,
        )


# %%
