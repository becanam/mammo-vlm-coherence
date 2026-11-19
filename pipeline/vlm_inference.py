"""
VLM inference backend.

Loads one of the four evaluated vision-language models and returns its text reply, for single-image,
multi-image (EMBED exams), and text-only (no-image) inputs. Imported by ``run_eval.py``; not run
directly.

Supported model types:
  gemma    -- MedGemma-4B           (AutoModelForImageTextToText)
  qwen2vl  -- Qwen2-VL-7B, Lingshu-7B (Qwen2-VL / Qwen2.5-VL conditional-generation)
  internvl -- InternVL3-8B          (AutoModel chat() API)
A generic fallback handles any other Vision2Seq model.
"""

import numpy as np
from PIL import Image
import torch


# ---------------------------------------------------------------------------
# Image helper
# ---------------------------------------------------------------------------

def internvl_pixel_values(image: Image.Image, input_size: int = 448):
    """ImageNet-normalized pixel tensor for InternVL's chat() API."""
    img = image.convert("RGB").resize((input_size, input_size), Image.BICUBIC)
    arr = np.asarray(img).astype(np.float32) / 255.0                 # H,W,C
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)       # 1,3,H,W


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_name: str):
    print(f"Loading {model_name} ...")
    dtype = torch.bfloat16

    if "medgemma" in model_name.lower() or "gemma" in model_name.lower():
        from transformers import AutoProcessor, AutoModelForImageTextToText
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForImageTextToText.from_pretrained(
            model_name, torch_dtype=dtype, device_map="auto"
        )
        model_type = "gemma"

    elif "qwen2-vl" in model_name.lower() or "qwen2_vl" in model_name.lower():
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        processor = AutoProcessor.from_pretrained(model_name)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=dtype, device_map="auto"
        )
        model_type = "qwen2vl"

    elif "lingshu" in model_name.lower() or "qwen2.5-vl" in model_name.lower() \
            or "qwen2_5_vl" in model_name.lower():
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        processor = AutoProcessor.from_pretrained(model_name)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=dtype, device_map="auto"
        )
        model_type = "qwen2vl"

    elif "internvl" in model_name.lower():
        from transformers import AutoModel, AutoTokenizer, PreTrainedModel
        processor = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
        if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
            PreTrainedModel.all_tied_weights_keys = {}
        model = AutoModel.from_pretrained(
            model_name, torch_dtype=dtype, trust_remote_code=True, low_cpu_mem_usage=False
        ).eval().to("cuda")
        model_type = "internvl"

    else:
        from transformers import AutoProcessor, AutoModelForVision2Seq
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForVision2Seq.from_pretrained(
            model_name, torch_dtype=dtype, device_map="auto"
        )
        model_type = "generic"

    if hasattr(model, "eval"):
        model.eval()
    print(f"Loaded. Model type: {model_type}")
    return model, processor, model_type


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(model, processor, model_type: str, image: Image.Image, prompt: str,
                  max_new_tokens: int = 32) -> str:
    """Single-image inference."""
    with torch.no_grad():
        if model_type == "gemma":
            messages = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ]}]
            inputs = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt"
            ).to(model.device, dtype=torch.bfloat16)
            input_len = inputs["input_ids"].shape[1]
            output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            decoded = processor.decode(output[0][input_len:], skip_special_tokens=True)

        elif model_type == "qwen2vl":
            messages = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            try:
                from qwen_vl_utils import process_vision_info
                image_inputs, video_inputs = process_vision_info(messages)
            except ImportError:
                image_inputs, video_inputs = [image], None   # single-image fallback
            inputs = processor(
                text=[text], images=image_inputs, videos=video_inputs,
                return_tensors="pt"
            ).to(model.device)
            output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            input_len = inputs["input_ids"].shape[1]
            decoded = processor.decode(output[0][input_len:], skip_special_tokens=True)

        elif model_type == "internvl":
            # processor is the tokenizer; use the repo's chat() API with a leading <image>.
            pixel_values = internvl_pixel_values(image).to(model.device, dtype=torch.bfloat16)
            gen_cfg = dict(max_new_tokens=max_new_tokens, do_sample=False)
            decoded = model.chat(processor, pixel_values, "<image>\n" + prompt, gen_cfg)

        else:
            inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
            output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            decoded = processor.decode(output[0], skip_special_tokens=True)

    return decoded.strip().upper()


def run_inference_multi(model, processor, model_type: str, images, prompt: str,
                        max_new_tokens: int = 32) -> str:
    """Multi-image inference (EMBED exams). ``images`` is a list of PIL images. Falls back to
    single-image ``run_inference`` when only one image is given."""
    images = list(images)
    if len(images) == 1:
        return run_inference(model, processor, model_type, images[0], prompt, max_new_tokens)
    with torch.no_grad():
        if model_type == "gemma":
            content = [{"type": "image", "image": im} for im in images]
            content.append({"type": "text", "text": prompt})
            messages = [{"role": "user", "content": content}]
            inputs = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt").to(model.device, dtype=torch.bfloat16)
            input_len = inputs["input_ids"].shape[1]
            output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            decoded = processor.decode(output[0][input_len:], skip_special_tokens=True)

        elif model_type == "qwen2vl":
            content = [{"type": "image", "image": im} for im in images]
            content.append({"type": "text", "text": prompt})
            messages = [{"role": "user", "content": content}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            try:
                from qwen_vl_utils import process_vision_info
                image_inputs, video_inputs = process_vision_info(messages)
            except ImportError:
                image_inputs, video_inputs = images, None
            inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                               return_tensors="pt").to(model.device)
            input_len = inputs["input_ids"].shape[1]
            output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            decoded = processor.decode(output[0][input_len:], skip_special_tokens=True)

        elif model_type == "internvl":
            pv_list, patches = [], []
            for im in images:
                pv = internvl_pixel_values(im).to(model.device, dtype=torch.bfloat16)
                pv_list.append(pv); patches.append(pv.shape[0])
            pixel_values = torch.cat(pv_list, dim=0)
            prefix = "".join(f"Image-{i+1}: <image>\n" for i in range(len(images)))
            gen_cfg = dict(max_new_tokens=max_new_tokens, do_sample=False)
            decoded = model.chat(processor, pixel_values, prefix + prompt, gen_cfg,
                                 num_patches_list=patches)

        else:  # generic: fall back to the first image
            return run_inference(model, processor, model_type, images[0], prompt, max_new_tokens)

    return decoded.strip().upper()


def run_text_inference(model, processor, model_type: str, prompt: str,
                       max_new_tokens: int = 32) -> str:
    """Text-only inference: NO image tensor at all. Runs the model's language side."""
    with torch.no_grad():
        if model_type == "gemma":
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            inputs = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt").to(model.device)
            input_len = inputs["input_ids"].shape[1]
            output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            decoded = processor.decode(output[0][input_len:], skip_special_tokens=True)

        elif model_type == "qwen2vl":
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], return_tensors="pt").to(model.device)
            output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            input_len = inputs["input_ids"].shape[1]
            decoded = processor.decode(output[0][input_len:], skip_special_tokens=True)

        elif model_type == "internvl":
            decoded = model.chat(processor, None, prompt,
                                 dict(max_new_tokens=max_new_tokens, do_sample=False))

        else:
            inputs = processor(text=prompt, return_tensors="pt").to(model.device)
            output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            decoded = processor.decode(output[0], skip_special_tokens=True)

    return decoded.strip().upper()
