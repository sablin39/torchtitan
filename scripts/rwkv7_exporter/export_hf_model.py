import argparse
from contextlib import nullcontext
import json
import os
import re
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    Qwen3VLVisionConfig,
    Qwen3VLVisionModel,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from .configuration_rwkv7 import RWKV7Config
    from .modeling_rwkv7 import RWKV7ForCausalLM, RWKV7Model
    from .tokenizer import CHAT_TEMPLATE_FAKE_THINKING, RwkvTokenizer
except ImportError:
    from configuration_rwkv7 import RWKV7Config
    from modeling_rwkv7 import RWKV7ForCausalLM, RWKV7Model
    from tokenizer import CHAT_TEMPLATE_FAKE_THINKING, RwkvTokenizer


IM_START_TOKEN = "\x16"
IM_END_TOKEN = "\x17"
IM_START_TOKEN_ID = 23
IM_END_TOKEN_ID = 24
DEFAULT_MAX_SHARD_SIZE = "1000GB"
VISION_PREFIXES = (
    "model.visual.",
    "visual.",
    "model.vision_model.",
    "vision_model.",
)
VISION_KEY_HINTS = (
    "patch_embed.",
    "pos_embed.",
    "blocks.",
    "merger.",
    "deepstack_merger_list.",
)


def resolve_dtype(precision: str, sample_dtype: torch.dtype) -> tuple[str, torch.dtype]:
    normalized = precision.lower()
    if normalized in {"auto", "same", "source"}:
        if sample_dtype in {torch.bfloat16, torch.float16, torch.float32, torch.float64}:
            return str(sample_dtype).split(".")[-1], sample_dtype
        return "float32", torch.float32
    if normalized in {"bf16", "bfloat16"}:
        return "bfloat16", torch.bfloat16
    if normalized in {"fp16", "float16"}:
        return "float16", torch.float16
    if normalized in {"fp32", "float32"}:
        return "float32", torch.float32
    if normalized in {"fp64", "double", "float64"}:
        return "float64", torch.float64
    raise ValueError(f"Unsupported precision '{precision}'.")


def infer_max_position_embeddings(rwkv7: str, override: int | None = None) -> int:
    if override is not None:
        return override
    match = re.search(r"ctx(\d+)", Path(rwkv7).stem)
    if match:
        return int(match.group(1))
    return RWKV7Config().max_position_embeddings


def resolve_vocab_file(name: str) -> Path:
    candidates = [
        Path(__file__).with_name(name),
    ]
    if name == "rwkv_vocab_v20230424.txt":
        candidates.append(Path(__file__).with_name("wr_vocab_v20230424.txt"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    formatted = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(f"Could not find RWKV vocab file. Checked:\n{formatted}")


def save_tokenizer_core(output: str) -> None:
    import torchtitan.models.rwkv7.tokenizer_core as tokenizer_core

    source = Path(tokenizer_core.__file__)
    if not source.is_file():
        raise FileNotFoundError(f"Could not find tokenizer_core.py at {source}")
    shutil.copyfile(source, Path(output) / "tokenizer_core.py")


def save_processor_core(output: str) -> None:
    candidates = [
        Path(__file__).parents[2]
        / "torchtitan"
        / "hf_datasets"
        / "multimodal"
        / "processor_core.py",
    ]
    try:
        import torchtitan.hf_datasets.multimodal.processor_core as processor_core
    except ImportError:
        pass
    else:
        candidates.append(Path(processor_core.__file__))
    source = next((candidate for candidate in candidates if candidate.is_file()), None)
    if source is None:
        formatted = "\n".join(f"  - {candidate}" for candidate in candidates)
        raise FileNotFoundError(f"Could not find processor_core.py. Checked:\n{formatted}")
    shutil.copyfile(source, Path(output) / "processor_core.py")


def torch_load_weights(path: str) -> dict[str, torch.Tensor]:
    try:
        weights = torch.load(path, weights_only=True, map_location="cpu", mmap=True)
    except TypeError:
        weights = torch.load(path, weights_only=True, map_location="cpu")
    if isinstance(weights, dict) and "state_dict" in weights:
        state_dict = weights["state_dict"]
        if isinstance(state_dict, dict):
            weights = state_dict
    if not isinstance(weights, dict):
        raise TypeError(f"Expected checkpoint at {path!r} to load as a state dict.")
    return weights


def extract_text_weights(weights: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    text_weights = {}
    for name, tensor in weights.items():
        if name.startswith(("proj.", "encoder.", "vision.", "visual.")):
            continue
        if name.startswith("llm."):
            name = name.replace("llm.", "", 1)
        text_weights[name] = tensor

    if "emb.weight" not in text_weights:
        raise KeyError("Expected a text-only RWKV checkpoint with 'emb.weight'.")
    return text_weights


def build_config(
    weights: dict[str, torch.Tensor],
    *,
    precision: str,
    max_position_embeddings: int,
) -> RWKV7Config:
    hidden_size = weights["blocks.0.ffn.key.weight"].shape[1]
    intermediate_size = weights["blocks.0.ffn.key.weight"].shape[0]
    num_hidden_layers = 0
    while f"blocks.{num_hidden_layers}.ffn.key.weight" in weights:
        num_hidden_layers += 1

    num_heads, head_dim = weights["blocks.0.att.r_k"].shape
    decay_low_rank_dim = weights["blocks.0.att.w1"].shape[1]
    gate_low_rank_dim = weights["blocks.0.att.g1"].shape[1]
    a_low_rank_dim = weights["blocks.0.att.a1"].shape[1]
    v_low_rank_dim = (
        weights["blocks.1.att.v1"].shape[1]
        if "blocks.1.att.v1" in weights
        else weights.get("blocks.0.att.v1", torch.empty(hidden_size, 32)).shape[1]
    )

    config = RWKV7Config(
        vocab_size=weights["emb.weight"].shape[0],
        hidden_size=hidden_size,
        hidden_ratio=intermediate_size // hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        head_dim=head_dim,
        num_heads=num_heads,
        num_attention_heads=num_heads,
        value_dim=[hidden_size] * num_hidden_layers,
        decay_low_rank_dim=decay_low_rank_dim,
        gate_low_rank_dim=gate_low_rank_dim,
        a_low_rank_dim=a_low_rank_dim,
        v_low_rank_dim=v_low_rank_dim,
        max_position_embeddings=max_position_embeddings,
        bos_token_id=IM_START_TOKEN_ID,
        eos_token_id=IM_END_TOKEN_ID,
        pad_token_id=IM_END_TOKEN_ID,
        fuse_cross_entropy=False,
        fuse_linear_cross_entropy=False,
    )
    config.dtype = precision
    config.auto_map = {
        "AutoConfig": "configuration_rwkv7.RWKV7Config",
        "AutoModel": "modeling_rwkv7.RWKV7Model",
        "AutoModelForCausalLM": "modeling_rwkv7.RWKV7ForCausalLM",
    }
    return config


def translate_into_hf(
    name: str,
    *,
    num_hidden_layers: int,
) -> tuple[str, bool]:
    unused_names = {"blocks.0.att.v0", "blocks.0.att.v1", "blocks.0.att.v2"}
    if name in unused_names:
        return "", False

    emb_head = {
        "emb.weight": "model.embeddings.weight",
        "ln_out.weight": "model.norm.weight",
        "ln_out.bias": "model.norm.bias",
        "head.weight": "lm_head.weight",
    }
    proj = {
        "receptance": "r_proj",
        "key": "k_proj",
        "value": "v_proj",
        "ln_x": "g_norm",
        "output": "o_proj",
    }

    if name in emb_head:
        return emb_head[name], False

    name_parts = name.split(".")
    if name_parts[0] != "blocks":
        raise KeyError(f"Unexpected checkpoint key '{name}'.")

    layer_idx = int(name_parts[1])
    if layer_idx not in range(num_hidden_layers):
        raise KeyError(f"Unexpected layer index in '{name}'.")

    name_parts[0] = "model.layers"
    name_parts[2] = {
        "att": "attn",
        "ffn": "ffn",
        "ln0": "pre_norm",
        "ln1": "attn_norm",
        "ln2": "ffn_norm",
    }[name_parts[2]]

    transposed = False
    if re.fullmatch(r"[wvag][012]", name_parts[3]):
        typ, num = name_parts[3]
        name_parts[3] = f"{typ}_lora.lora." + {
            "0": "2.bias",
            "1": "0.weight",
            "2": "2.weight",
        }[num]
        transposed = num in {"1", "2"}
    elif name_parts[2] == "attn" and name_parts[3] in proj:
        name_parts[3] = proj[name_parts[3]]

    return ".".join(name_parts), transposed


def build_converted_state_dict(
    weights: dict[str, torch.Tensor],
    model: RWKV7ForCausalLM,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    model_dict = model.state_dict()
    pending_names = set(model_dict)
    converted: dict[str, torch.Tensor] = {}
    possible_absent_weights = {
        "model.layers.0.pre_norm.weight",
        "model.layers.0.pre_norm.bias",
    }

    for source_name, source_weight in weights.items():
        hf_name, transposed = translate_into_hf(
            source_name,
            num_hidden_layers=model.config.num_hidden_layers,
        )
        if not hf_name:
            continue

        weight = source_weight.detach()
        if transposed:
            weight = weight.t()

        target_tensor = model_dict[hf_name]
        if (
            weight.ndim == 3
            and weight.shape[:2] == (1, 1)
            and target_tensor.ndim == 1
            and weight.shape[-1] == target_tensor.shape[0]
        ):
            weight = weight.squeeze(0).squeeze(0)
        expected_shape = target_tensor.shape
        if weight.shape != expected_shape:
            raise ValueError(
                f"Shape mismatch for {source_name} -> {hf_name}: "
                f"checkpoint={tuple(weight.shape)} expected={tuple(expected_shape)}"
            )

        converted[hf_name] = weight.to(dtype=dtype).contiguous()
        pending_names.discard(hf_name)

    missing_required = sorted(pending_names - possible_absent_weights)
    if missing_required:
        raise KeyError(f"Missing required parameters after conversion: {missing_required}")

    return converted


def build_model(config: RWKV7Config) -> RWKV7ForCausalLM:
    with torch.device("meta"):
        model = RWKV7ForCausalLM(config)
    return model


def prefix_text_state_for_vl(
    converted_state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    vl_state = {}
    for key, value in converted_state.items():
        if key.startswith("model."):
            vl_state["model.llm." + key.removeprefix("model.")] = value
        elif key == "lm_head.weight":
            vl_state[key] = value
        else:
            raise KeyError(f"Unexpected converted RWKV key for VL export: {key}")
    return vl_state


def _normalize_vision_key(key: str) -> str | None:
    for prefix in VISION_PREFIXES:
        if key.startswith(prefix):
            return key.removeprefix(prefix)
    if key.startswith(VISION_KEY_HINTS):
        return key
    return None


def _safetensor_files(path: Path) -> list[Path]:
    if path.is_file() and path.suffix == ".safetensors":
        return [path]
    if not path.is_dir():
        return []

    index_path = path / "model.safetensors.index.json"
    if index_path.is_file():
        with index_path.open("r", encoding="utf-8") as handle:
            index = json.load(handle)
        files = sorted(set(index.get("weight_map", {}).values()))
        return [path / file for file in files]
    return sorted(path.glob("*.safetensors"))


def _torch_weight_files(path: Path) -> list[Path]:
    if path.is_file() and path.suffix in {".bin", ".pt", ".pth"}:
        return [path]
    if not path.is_dir():
        return []
    index_path = path / "pytorch_model.bin.index.json"
    if index_path.is_file():
        with index_path.open("r", encoding="utf-8") as handle:
            index = json.load(handle)
        files = sorted(set(index.get("weight_map", {}).values()))
        return [path / file for file in files]
    return sorted(path.glob("pytorch_model*.bin"))


def _ensure_qwen3_vl_vision_config(vision_config) -> Qwen3VLVisionConfig:
    if isinstance(vision_config, Qwen3VLVisionConfig):
        return vision_config
    if not isinstance(vision_config, dict) and hasattr(vision_config, "to_dict"):
        vision_config = vision_config.to_dict()
    if isinstance(vision_config, dict):
        model_type = vision_config.get("model_type")
        if model_type not in {None, Qwen3VLVisionConfig.model_type}:
            raise TypeError(
                "Expected a Qwen3-VL vision config; "
                f"got model_type={model_type!r}."
            )
        return Qwen3VLVisionConfig(**vision_config)
    raise TypeError(f"Unsupported vision config type: {type(vision_config)!r}.")


def _to_qwen3_vl_vision_config(vision_config) -> Qwen3VLVisionConfig:
    if isinstance(vision_config, Qwen3VLVisionConfig):
        return vision_config
    if not isinstance(vision_config, dict) and hasattr(vision_config, "to_dict"):
        vision_config = vision_config.to_dict()
    if not isinstance(vision_config, dict):
        raise TypeError(f"Unsupported vision config type: {type(vision_config)!r}.")

    return Qwen3VLVisionConfig(
        depth=vision_config["depth"],
        hidden_size=vision_config["hidden_size"],
        hidden_act=vision_config.get("hidden_act", "gelu_pytorch_tanh"),
        intermediate_size=vision_config["intermediate_size"],
        num_heads=vision_config["num_heads"],
        in_channels=vision_config.get("in_channels", 3),
        patch_size=vision_config.get("patch_size", 16),
        spatial_merge_size=vision_config.get("spatial_merge_size", 2),
        temporal_patch_size=vision_config.get("temporal_patch_size", 2),
        out_hidden_size=vision_config["out_hidden_size"],
        num_position_embeddings=vision_config.get("num_position_embeddings", 2304),
        deepstack_visual_indexes=list(
            vision_config.get("deepstack_visual_indexes")
            or vision_config.get("deepstack_visual_indices")
            or []
        ),
        initializer_range=vision_config.get("initializer_range", 0.02),
    )


def load_qwen3_vl_vision_config(vision_model: str) -> Qwen3VLVisionConfig:
    config = AutoConfig.from_pretrained(vision_model, trust_remote_code=True)
    vision_config = getattr(config, "vision_config", config)
    try:
        return _to_qwen3_vl_vision_config(vision_config)
    except TypeError as exc:
        raise TypeError(
            f"Could not derive a Qwen3-VL-compatible vision config from {vision_model!r}; "
            f"got {type(vision_config)!r}."
        ) from exc


def _extract_visual_module(model: torch.nn.Module) -> torch.nn.Module:
    for path in (
        "model.visual",
        "visual",
        "model.vision_model",
        "vision_model",
    ):
        module: object = model
        for part in path.split("."):
            module = getattr(module, part, None)
            if module is None:
                break
        if isinstance(module, torch.nn.Module):
            return module
    raise AttributeError(
        "Could not find a visual module on the loaded HF model. Checked "
        "model.visual, visual, model.vision_model, and vision_model."
    )


def _load_hf_vision_source_model(vision_model: str, dtype: torch.dtype) -> torch.nn.Module:
    kwargs = {
        "trust_remote_code": True,
        "dtype": dtype,
        "low_cpu_mem_usage": True,
    }
    try:
        return AutoModelForImageTextToText.from_pretrained(vision_model, **kwargs)
    except (ValueError, TypeError):
        return AutoModel.from_pretrained(vision_model, **kwargs)


def load_qwen_vision_package(
    vision_model: str,
    *,
    dtype: torch.dtype,
) -> tuple[Qwen3VLVisionConfig, dict[str, torch.Tensor]]:
    config = AutoConfig.from_pretrained(vision_model, trust_remote_code=True)
    vision_config = _to_qwen3_vl_vision_config(getattr(config, "vision_config", config))
    model = _load_hf_vision_source_model(vision_model, dtype=dtype)
    model.eval()
    visual = _extract_visual_module(model)
    vision_state = {
        key: value.detach().cpu().to(dtype=dtype).contiguous()
        for key, value in visual.state_dict().items()
    }
    validate_vision_state(vision_state, vision_config)
    del visual, model
    return vision_config, vision_state


def load_qwen3_vl_vision_state_dict(
    vision_model: str,
    *,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    path = Path(vision_model).expanduser()
    state_dict: dict[str, torch.Tensor] = {}

    safetensor_files = _safetensor_files(path)
    for file in safetensor_files:
        with safe_open(file, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                normalized = _normalize_vision_key(key)
                if normalized is None:
                    continue
                if normalized in state_dict:
                    continue
                state_dict[normalized] = handle.get_tensor(key).to(dtype=dtype).contiguous()

    if state_dict:
        return state_dict

    for file in _torch_weight_files(path):
        weights = torch_load_weights(str(file))
        for key, value in weights.items():
            normalized = _normalize_vision_key(key)
            if normalized is None or normalized in state_dict:
                continue
            state_dict[normalized] = value.detach().to(dtype=dtype).contiguous()

    if not state_dict:
        raise KeyError(
            f"Could not find Qwen3-VL-compatible vision weights in {vision_model!r}. "
            "Expected keys such as 'model.visual.patch_embed.proj.weight' "
            "or 'patch_embed.proj.weight'."
        )
    return state_dict


def validate_component_state(
    state_dict: dict[str, torch.Tensor],
    expected_state: dict[str, torch.Tensor],
    *,
    component: str,
) -> None:
    missing = sorted(set(expected_state) - set(state_dict))
    unexpected = sorted(set(state_dict) - set(expected_state))
    if missing:
        raise KeyError(f"Missing {component} parameters: {missing[:20]}")
    if unexpected:
        raise KeyError(f"Unexpected {component} parameters: {unexpected[:20]}")

    for key, value in state_dict.items():
        expected = expected_state[key]
        if tuple(value.shape) != tuple(expected.shape):
            raise ValueError(
                f"Shape mismatch for {component} key {key}: "
                f"checkpoint={tuple(value.shape)} expected={tuple(expected.shape)}"
            )


def validate_vision_state(
    vision_state: dict[str, torch.Tensor],
    vision_config: Qwen3VLVisionConfig,
) -> None:
    with torch.device("meta"):
        vision = Qwen3VLVisionModel(vision_config)
    validate_component_state(
        vision_state,
        vision.state_dict(),
        component="Qwen3-VL vision encoder",
    )


def build_projector_state_dict(
    *,
    encoder_dim: int,
    project_dim: int,
    hidden_dim: int | None,
    num_deepstack: int,
    dtype: torch.dtype,
    seed: int | None,
) -> dict[str, torch.Tensor]:
    try:
        from .modeling_modrwkv import VisualAdapter
    except ImportError:
        from modeling_modrwkv import VisualAdapter

    rng_context = torch.random.fork_rng(devices=[]) if seed is not None else nullcontext()
    with rng_context:
        if seed is not None:
            torch.manual_seed(seed)
        projector = VisualAdapter(
            encoder_dim=encoder_dim,
            project_dim=project_dim,
            hidden_dim=hidden_dim,
            num_deepstack=num_deepstack,
            use_conv=False,
        )
    return {
        "model.proj." + key: value.detach().to(dtype=dtype).contiguous()
        for key, value in projector.state_dict().items()
    }


def build_multimodal_config(
    *,
    text_config: RWKV7Config,
    vision_config: Qwen3VLVisionConfig,
    projector_hidden_dim: int | None,
):
    try:
        from .modeling_modrwkv import ModRWKVConfig, ModRWKVProjectorConfig
    except ImportError:
        from modeling_modrwkv import ModRWKVConfig, ModRWKVProjectorConfig

    projector_config = ModRWKVProjectorConfig(
        encoder_dim=vision_config.out_hidden_size,
        project_dim=text_config.hidden_size,
        hidden_dim=projector_hidden_dim,
        num_deepstack=len(getattr(vision_config, "deepstack_visual_indexes", [])),
    )
    config = ModRWKVConfig.from_text_vision_configs(
        text_config=text_config,
        vision_config=vision_config,
        projector_config=projector_config,
        image_token_id=65532,
        vision_start_token_id=65530,
        vision_end_token_id=65531,
        tie_word_embeddings=False,
    )
    config.architectures = ["RWKV7VLForConditionalGeneration"]
    return config


def build_multimodal_model(config):
    try:
        from .modeling_modrwkv import RWKV7VLForConditionalGeneration
    except ImportError:
        from modeling_modrwkv import RWKV7VLForConditionalGeneration

    with torch.device("meta"):
        model = RWKV7VLForConditionalGeneration(config)
    return model


def save_multimodal_processor(
    *,
    output: str,
    image_processor_source: str,
    max_pixels: int | None,
    fake_thinking: bool,
) -> None:
    try:
        from .processor import (
            CHAT_TEMPLATE_FAKE_THINKING as PROCESSOR_CHAT_TEMPLATE_FAKE_THINKING,
        )
        from .processor import ModRWKVProcessor
        from .tokenizer import RwkvTokenizer as VLRwkvTokenizer
    except ImportError:
        from processor import (  # type: ignore[no-redef]
            CHAT_TEMPLATE_FAKE_THINKING as PROCESSOR_CHAT_TEMPLATE_FAKE_THINKING,
        )
        from processor import ModRWKVProcessor
        from tokenizer import RwkvTokenizer as VLRwkvTokenizer

    VLRwkvTokenizer.register_for_auto_class("AutoTokenizer")
    ModRWKVProcessor.register_for_auto_class("AutoProcessor")

    tokenizer = VLRwkvTokenizer(
        vocab_file=str(resolve_vocab_file("wr_vocab_v20230424.txt")),
        bos_token="\x16",
        eos_token="\x17",
        pad_token="\x17",
        unk_token="\x16",
    )
    image_processor = AutoImageProcessor.from_pretrained(
        image_processor_source,
        trust_remote_code=True,
    )
    if max_pixels is not None:
        if max_pixels <= 0:
            raise ValueError("--max-pixels must be positive when provided.")
        image_processor.size["longest_edge"] = int(max_pixels)

    processor = ModRWKVProcessor(
        tokenizer=tokenizer,
        image_processor=image_processor,
        chat_template=PROCESSOR_CHAT_TEMPLATE_FAKE_THINKING if fake_thinking else None,
    )
    processor.save_pretrained(output)
    save_tokenizer_core(output)
    save_processor_core(output)


def verify_text_export(output: str, *, dtype: torch.dtype, verify_model_load: bool) -> None:
    print("Verifying AutoConfig / AutoTokenizer loading...")
    loaded_config = AutoConfig.from_pretrained(output, trust_remote_code=True)
    loaded_tokenizer = AutoTokenizer.from_pretrained(output, trust_remote_code=True)
    loaded_model = None
    if verify_model_load:
        print("Verifying AutoModelForCausalLM loading...")
        loaded_model = AutoModelForCausalLM.from_pretrained(
            output,
            trust_remote_code=True,
            dtype=dtype,
            low_cpu_mem_usage=True,
        )

    sample_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "hello"},
    ]
    rendered_prompt = loaded_tokenizer.apply_chat_template(
        sample_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    sample_batch = loaded_tokenizer.apply_chat_template(
        sample_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if hasattr(sample_batch, "keys"):
        prompt_shape = tuple(sample_batch["input_ids"].shape)
    else:
        prompt_shape = tuple(sample_batch.shape)
    del loaded_config, loaded_model
    print("Rendered chat_template preview:")
    print(rendered_prompt)
    print(f"Tokenizer verification prompt shape: {prompt_shape}")


def verify_multimodal_export(
    output: str,
    *,
    dtype: torch.dtype,
    verify_model_load: bool,
) -> None:
    print("Verifying AutoConfig / AutoProcessor loading...")
    loaded_config = AutoConfig.from_pretrained(output, trust_remote_code=True)
    loaded_processor = AutoProcessor.from_pretrained(output, trust_remote_code=True)
    if verify_model_load:
        print("Verifying AutoModelForImageTextToText loading...")
        loaded_model = AutoModelForImageTextToText.from_pretrained(
            output,
            trust_remote_code=True,
            dtype=dtype,
            low_cpu_mem_usage=True,
        )
        del loaded_model
    print(f"Loaded config type: {type(loaded_config).__name__}")
    print(f"Loaded processor type: {type(loaded_processor).__name__}")
    del loaded_config, loaded_processor


def convert(
    rwkv7: str,
    output: str,
    precision: str = "auto",
    max_position_embeddings: int | None = None,
    max_shard_size: str = DEFAULT_MAX_SHARD_SIZE,
    verify_model_load: bool = True,
    fake_thinking: bool = False,
):
    output = os.path.realpath(output)
    text_weights = extract_text_weights(torch_load_weights(rwkv7))

    precision_name, dtype = resolve_dtype(precision, next(iter(text_weights.values())).dtype)
    config = build_config(
        text_weights,
        precision=precision_name,
        max_position_embeddings=infer_max_position_embeddings(rwkv7, max_position_embeddings),
    )
    print(f"Creating text-only RWKV7 HF model with config:\n{config}")

    RWKV7Config.register_for_auto_class()
    RWKV7ForCausalLM.register_for_auto_class("AutoModelForCausalLM")

    model = build_model(config)
    converted_state = build_converted_state_dict(text_weights, model, dtype)
    missing, unexpected = model.load_state_dict(converted_state, strict=True, assign=True)
    if missing or unexpected:
        raise RuntimeError(f"Unexpected load_state_dict result: missing={missing}, unexpected={unexpected}")

    os.makedirs(output, exist_ok=True)
    model.save_pretrained(
        output,
        safe_serialization=True,
        max_shard_size=max_shard_size,
    )

    tokenizer = RwkvTokenizer(
        vocab_file=str(resolve_vocab_file("wr_vocab_v20230424.txt")),
        bos_token=IM_START_TOKEN,
        eos_token=IM_END_TOKEN,
        pad_token=IM_END_TOKEN,
        unk_token=IM_START_TOKEN,
        chat_template=CHAT_TEMPLATE_FAKE_THINKING if fake_thinking else None,
    )
    tokenizer.register_for_auto_class()
    tokenizer.save_pretrained(output)
    save_tokenizer_core(output)

    print(f"Saved text-only HF checkpoint to {output}")
    verify_text_export(output, dtype=dtype, verify_model_load=verify_model_load)
    print(f"Export completed successfully: {output}")


def convert_multimodal(
    rwkv7: str,
    vision_model: str,
    output: str,
    precision: str = "auto",
    max_position_embeddings: int | None = None,
    max_shard_size: str = DEFAULT_MAX_SHARD_SIZE,
    projector_hidden_dim: int | None = None,
    projector_seed: int | None = None,
    image_processor: str | None = None,
    max_pixels: int | None = None,
    verify_model_load: bool = False,
    fake_thinking: bool = False,
) -> None:
    output = os.path.realpath(output)
    text_weights = extract_text_weights(torch_load_weights(rwkv7))
    precision_name, dtype = resolve_dtype(precision, next(iter(text_weights.values())).dtype)

    text_config = build_config(
        text_weights,
        precision=precision_name,
        max_position_embeddings=infer_max_position_embeddings(rwkv7, max_position_embeddings),
    )
    vision_config, vision_state = load_qwen_vision_package(vision_model, dtype=dtype)
    config = build_multimodal_config(
        text_config=text_config,
        vision_config=vision_config,
        projector_hidden_dim=projector_hidden_dim,
    )
    print(f"Creating RWKV-VL HF model with config:\n{config}")

    text_model = build_model(text_config)
    text_state = prefix_text_state_for_vl(
        build_converted_state_dict(text_weights, text_model, dtype)
    )

    vision_state = {"model.encoder." + key: value for key, value in vision_state.items()}

    projector_state = build_projector_state_dict(
        encoder_dim=vision_config.out_hidden_size,
        project_dim=text_config.hidden_size,
        hidden_dim=projector_hidden_dim,
        num_deepstack=len(getattr(vision_config, "deepstack_visual_indexes", [])),
        dtype=dtype,
        seed=projector_seed,
    )

    state_dict = {}
    state_dict.update(text_state)
    state_dict.update(vision_state)
    state_dict.update(projector_state)

    model = build_multimodal_model(config)
    validate_component_state(
        state_dict,
        model.state_dict(),
        component="RWKV-VL checkpoint",
    )

    os.makedirs(output, exist_ok=True)
    model.save_pretrained(
        output,
        state_dict=state_dict,
        safe_serialization=True,
        max_shard_size=max_shard_size,
    )
    save_multimodal_processor(
        output=output,
        image_processor_source=image_processor or vision_model,
        max_pixels=max_pixels,
        fake_thinking=fake_thinking,
    )
    print(f"Saved RWKV-VL HF checkpoint to {output}")
    verify_multimodal_export(
        output,
        dtype=dtype,
        verify_model_load=verify_model_load,
    )
    print(f"Export completed successfully: {output}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert RWKV7 .pth checkpoints to HF format.')
    parser.add_argument('--rwkv7', type=str, required=True, help='Path to the input RWKV .pth checkpoint.')
    parser.add_argument('--output', type=str, required=True, help='Directory to save the exported model.')
    parser.add_argument('--precision', type=str, default='bfloat16')
    parser.add_argument('--max-position-embeddings', type=int, default=None)
    parser.add_argument('--max-shard-size', type=str, default=DEFAULT_MAX_SHARD_SIZE)
    parser.add_argument(
        '--multimodal',
        action='store_true',
        help=(
            'Export an RWKV-VL checkpoint by combining RWKV text weights with '
            'Qwen3-VL-compatible vision weights.'
        ),
    )
    parser.add_argument(
        '--vision-model',
        type=str,
        default=None,
        help='HF-compatible Qwen3-VL model or vision-encoder path used when --multimodal is set.',
    )
    parser.add_argument(
        '--image-processor',
        type=str,
        default=None,
        help='Optional image processor source. Defaults to --vision-model for multimodal exports.',
    )
    parser.add_argument(
        '--projector-hidden-dim',
        type=int,
        default=None,
        help='Optional hidden dimension for the freshly initialized visual adapter.',
    )
    parser.add_argument(
        '--projector-seed',
        type=int,
        default=None,
        help='Optional RNG seed for reproducible fresh visual adapter initialization.',
    )
    parser.add_argument(
        '--max-pixels',
        type=int,
        default=None,
        help='Optional max spatial pixel budget used to cap the saved image processor longest_edge.',
    )
    parser.add_argument(
        '--verify-model-load',
        action='store_true',
        help='After saving, load the full exported model through AutoModel. This can require substantial RAM.',
    )
    parser.add_argument(
        '--fake-thinking',
        action='store_true',
        help='Save a chat template that prefixes every assistant message with an empty <think> block.',
    )
    args = parser.parse_args()

    if args.multimodal:
        if args.vision_model is None:
            raise ValueError("--multimodal requires --vision-model.")
        convert_multimodal(
            args.rwkv7,
            args.vision_model,
            args.output,
            precision=args.precision,
            max_position_embeddings=args.max_position_embeddings,
            max_shard_size=args.max_shard_size,
            projector_hidden_dim=args.projector_hidden_dim,
            projector_seed=args.projector_seed,
            image_processor=args.image_processor,
            max_pixels=args.max_pixels,
            verify_model_load=args.verify_model_load,
            fake_thinking=args.fake_thinking,
        )
    else:
        convert(
            args.rwkv7,
            args.output,
            precision=args.precision,
            max_position_embeddings=args.max_position_embeddings,
            max_shard_size=args.max_shard_size,
            verify_model_load=args.verify_model_load,
            fake_thinking=args.fake_thinking,
        )
