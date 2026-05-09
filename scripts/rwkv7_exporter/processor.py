import copy

import torch
from transformers import BaseImageProcessor, PreTrainedTokenizer
from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import MultiModalData, ProcessingKwargs, ProcessorMixin, Unpack

try:
    from .processor_core import (
        CHAT_TEMPLATE,
        CHAT_TEMPLATE_FAKE_THINKING,
        make_image_config_from_processor,
        process_images,
    )
except ImportError:
    try:
        from processor_core import (  # type: ignore[no-redef]
            CHAT_TEMPLATE,
            CHAT_TEMPLATE_FAKE_THINKING,
            make_image_config_from_processor,
            process_images,
        )
    except ImportError:
        from torchtitan.hf_datasets.multimodal.processor_core import (  # type: ignore[no-redef]
            CHAT_TEMPLATE,
            CHAT_TEMPLATE_FAKE_THINKING,
            make_image_config_from_processor,
            process_images,
        )


class ModRWKVProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "return_token_type_ids": False,
        },
        "images_kwargs": {},
    }


class ModRWKVProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    tokenizer_class = "RwkvTokenizer"
    user_image_tag = "<image>"

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer = None,
        image_processor: BaseImageProcessor = None,
        chat_template=None,
        auto_insert_image_tags: bool = True,
        total_pixels_budget: bool = True,
    ):
        chat_template = CHAT_TEMPLATE if chat_template is None else chat_template
        super().__init__(
            tokenizer=tokenizer,
            image_processor=image_processor,
            chat_template=chat_template,
        )
        self.auto_insert_image_tags = auto_insert_image_tags
        self.total_pixels_budget = total_pixels_budget
        self.image_token = getattr(tokenizer, "image_token", "<|image_pad|>")
        self.vision_start_token = getattr(tokenizer, "vision_start_token", "<|vision_start|>")
        self.vision_end_token = getattr(tokenizer, "vision_end_token", "<|vision_end|>")
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)
        self.vision_start_token_id = self.tokenizer.convert_tokens_to_ids(self.vision_start_token)
        self.vision_end_token_id = self.tokenizer.convert_tokens_to_ids(self.vision_end_token)
        self.vision_image_token = (
            f"{self.vision_start_token}{self.image_token}{self.vision_end_token}"
        )

    def to_dict(self):
        output = {}
        if self.image_processor is not None:
            output["image_processor"] = self.image_processor.to_dict()
        if getattr(self, "auto_map", None) is not None:
            output["auto_map"] = copy.deepcopy(self.auto_map)
        output["processor_class"] = self.__class__.__name__
        if not self.auto_insert_image_tags:
            output["auto_insert_image_tags"] = False
        output["total_pixels_budget"] = self.total_pixels_budget
        return output

    def _flatten_images(self, images):
        if images is None:
            return []
        if not isinstance(images, (list, tuple)):
            return [images]

        flat_images = []
        for item in images:
            if isinstance(item, (list, tuple)):
                flat_images.extend(self._flatten_images(item))
            else:
                flat_images.append(item)
        return flat_images

    def _get_num_images_per_text_sample(self, images, batch_size):
        if images is None:
            return [0] * batch_size
        if batch_size == 1:
            return [len(self._flatten_images(images))]
        if isinstance(images, (list, tuple)) and len(images) == batch_size:
            return [len(self._flatten_images(sample_images)) for sample_images in images]
        return None

    def _get_images_per_text_sample(self, images, batch_size):
        if images is None:
            return [[] for _ in range(batch_size)]
        if batch_size == 1:
            return [self._flatten_images(images)]
        if isinstance(images, (list, tuple)) and len(images) == batch_size:
            return [self._flatten_images(sample_images) for sample_images in images]
        return None

    def _process_images(self, images, batch_size, images_kwargs):
        image_groups = self._get_images_per_text_sample(images, batch_size)
        if image_groups is None:
            image_groups = [self._flatten_images(images)]
            num_images_per_sample = None
        else:
            num_images_per_sample = [len(group) for group in image_groups]

        image_config = make_image_config_from_processor(
            self.image_processor,
            **images_kwargs,
        )
        processed_groups = [process_images(group, image_config) for group in image_groups]
        num_image_tokens = [
            count
            for processed in processed_groups
            for count in processed.image_token_counts
        ]
        if not num_image_tokens:
            return {}, None, None, num_images_per_sample

        pixel_values = torch.cat(
            [
                processed.flat_patches
                for processed in processed_groups
                if processed.flat_patches.numel() > 0
            ],
            dim=0,
        )
        image_grid_thw = torch.cat(
            [
                processed.grid_thw
                for processed in processed_groups
                if processed.grid_thw.numel() > 0
            ],
            dim=0,
        )
        return (
            {
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw,
            },
            image_grid_thw,
            num_image_tokens,
            num_images_per_sample,
        )

    def _normalize_image_tags(self, text):
        return text.replace(self.user_image_tag, self.vision_image_token)

    def _strip_excess_image_tags(self, text, num_allowed):
        tag = self.user_image_tag
        count = text.count(tag)
        if count <= num_allowed:
            return text
        parts = text.split(tag)
        kept = tag.join(parts[: num_allowed + 1])
        rest = "".join(parts[num_allowed + 1 :])
        return kept + rest

    def _append_missing_image_tags(self, text, num_missing_images):
        if num_missing_images <= 0:
            return text
        return text + self.vision_image_token * num_missing_images

    def _get_num_multimodal_tokens(self, image_grid_thw=None, **kwargs):
        vision_data = {}
        if image_grid_thw is not None:
            processor_defaults = getattr(self.image_processor, "_defaults", {})
            images_kwargs = dict(processor_defaults.get("images_kwargs", {}))
            images_kwargs.update(kwargs)
            merge_size = images_kwargs.get("merge_size", None) or self.image_processor.merge_size

            num_image_patches = [int(grid[0] * grid[1] * grid[2]) for grid in image_grid_thw]
            num_image_tokens = [num_patches // merge_size**2 for num_patches in num_image_patches]
            vision_data.update(
                {
                    "num_image_tokens": num_image_tokens,
                    "num_image_patches": num_image_patches,
                }
            )
        return MultiModalData(**vision_data)

    def _count_token_occurrences(self, input_ids, token_id):
        return [sum(1 for token in sample_ids if token == token_id) for sample_ids in input_ids]

    def _validate_image_token_alignment(self, text_inputs, expected_image_tokens, expected_num_images):
        input_ids = text_inputs["input_ids"]
        actual_image_tokens = self._count_token_occurrences(input_ids, self.image_token_id)
        actual_vision_starts = self._count_token_occurrences(input_ids, self.vision_start_token_id)
        actual_vision_ends = self._count_token_occurrences(input_ids, self.vision_end_token_id)

        if actual_image_tokens != expected_image_tokens:
            raise ValueError(
                "Image token count does not match image_grid_thw-derived token count: "
                f"expected {expected_image_tokens}, got {actual_image_tokens}."
            )
        if actual_vision_starts != expected_num_images or actual_vision_ends != expected_num_images:
            raise ValueError(
                "Vision boundary token count does not match the number of image placeholders: "
                f"expected {expected_num_images}, got starts={actual_vision_starts}, ends={actual_vision_ends}."
            )

    def __call__(self, images=None, text=None, **kwargs: Unpack[ModRWKVProcessorKwargs]):
        output_kwargs = self._merge_kwargs(
            ModRWKVProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if not isinstance(text, list):
            text = [text] if text is not None else None

        batch_size = len(text) if text is not None else 1
        if images is not None:
            (
                image_inputs,
                image_grid_thw,
                num_image_tokens,
                num_images_per_sample,
            ) = self._process_images(
                images,
                batch_size,
                output_kwargs["images_kwargs"],
            )
        else:
            image_inputs = {}
            image_grid_thw = None
            num_image_tokens = None
            num_images_per_sample = None

        if text is None:
            return BatchFeature(data=image_inputs)

        text = text.copy()
        expected_image_tokens = [0 for _ in text]
        expected_num_images = [0 for _ in text]
        if image_grid_thw is not None:
            index = 0
            for i in range(len(text)):
                if not self.auto_insert_image_tags:
                    text[i] = text[i].replace(self.user_image_tag, " ")
                else:
                    if num_images_per_sample is not None:
                        text[i] = self._strip_excess_image_tags(text[i], num_images_per_sample[i])
                    text[i] = self._normalize_image_tags(text[i])

                if self.auto_insert_image_tags and num_images_per_sample is not None:
                    missing = num_images_per_sample[i] - text[i].count(self.image_token)
                    text[i] = self._append_missing_image_tags(text[i], missing)

                if self.auto_insert_image_tags:
                    placeholder_count = text[i].count(self.vision_image_token)
                    if index + placeholder_count > len(num_image_tokens):
                        raise ValueError(
                            "Number of image placeholders in text exceeds provided images: "
                            f"consumed {index + placeholder_count}, available {len(num_image_tokens)}."
                        )
                    sample_counts = num_image_tokens[index : index + placeholder_count]
                    text[i] = self.tokenizer.expand_image_placeholders(
                        text[i],
                        sample_counts,
                    )
                    expected_image_tokens[i] += sum(sample_counts)
                    expected_num_images[i] += len(sample_counts)
                    index += placeholder_count
                else:
                    while self.image_token in text[i]:
                        if index >= len(num_image_tokens):
                            raise ValueError(
                                "Number of image placeholders in text exceeds provided images: "
                                f"consumed {index + 1}, available {len(num_image_tokens)}."
                            )
                        text[i] = text[i].replace(
                            self.image_token,
                            "<|placeholder|>" * num_image_tokens[index],
                            1,
                        )
                        expected_image_tokens[i] += num_image_tokens[index]
                        expected_num_images[i] += 1
                        index += 1
                    text[i] = text[i].replace("<|placeholder|>", self.image_token)

            if self.auto_insert_image_tags and index != len(num_image_tokens):
                raise ValueError(
                    "Number of image placeholders in text does not match provided images: "
                    f"consumed {index}, available {len(num_image_tokens)}."
                )
        else:
            for i in range(len(text)):
                text[i] = text[i].replace(self.user_image_tag, "")

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
        if image_grid_thw is not None:
            self._validate_image_token_alignment(
                text_inputs,
                expected_image_tokens,
                expected_num_images,
            )
        self._check_special_mm_tokens(text, text_inputs, modalities=["image"])
        return BatchFeature(data={**text_inputs, **image_inputs}, tensor_type=return_tensors)

    def apply_chat_template(self, conversation, chat_template=None, **kwargs):
        kwargs.setdefault("return_dict", True)
        return super().apply_chat_template(
            conversation,
            chat_template=chat_template,
            **kwargs,
        )


ModRWKVProcessor.register_for_auto_class("AutoProcessor")
