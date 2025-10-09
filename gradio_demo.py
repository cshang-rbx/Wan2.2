import copy
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import gradio as gr
import torch
from PIL import Image

import wan
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.utils.utils import save_video


PIPELINE_CACHE: Dict[Tuple[str, str, bool, bool], object] = {}
DEVICE_ID = int(os.environ.get("WAN_DEVICE_ID", 0))


def _require_cuda():
    if not torch.cuda.is_available():
        raise gr.Error("CUDA device is required to run Wan2.2 inference.")


def _normalize_frame_num(value: int) -> int:
    value = int(value)
    if value < 21:
        value = 21
    remainder = (value - 1) % 4
    return value if remainder == 0 else value - remainder


def _load_pipeline(task: str, ckpt_dir: str, convert_model_dtype: bool,
                   t5_cpu: bool):
    key = (task, os.path.abspath(ckpt_dir), convert_model_dtype, t5_cpu)
    pipeline = PIPELINE_CACHE.get(key)
    if pipeline is not None:
        return pipeline

    if not ckpt_dir or not Path(ckpt_dir).exists():
        raise gr.Error(f"Checkpoint directory not found: {ckpt_dir}")

    cfg = copy.deepcopy(WAN_CONFIGS[task])

    if task == "t2v-A14B":
        pipeline = wan.WanT2V(
            config=cfg,
            checkpoint_dir=ckpt_dir,
            device_id=DEVICE_ID,
            rank=0,
            convert_model_dtype=convert_model_dtype,
            t5_cpu=t5_cpu,
        )
    elif task == "i2v-A14B":
        pipeline = wan.WanI2V(
            config=cfg,
            checkpoint_dir=ckpt_dir,
            device_id=DEVICE_ID,
            rank=0,
            convert_model_dtype=convert_model_dtype,
            t5_cpu=t5_cpu,
        )
    elif task == "ti2v-5B":
        pipeline = wan.WanTI2V(
            config=cfg,
            checkpoint_dir=ckpt_dir,
            device_id=DEVICE_ID,
            rank=0,
            convert_model_dtype=convert_model_dtype,
            t5_cpu=t5_cpu,
        )
    else:
        raise gr.Error(f"Unsupported task: {task}")

    PIPELINE_CACHE[key] = pipeline
    return pipeline


def _save_video_tensor(tensor, fps: int, save_path: str = None) -> str:
    if save_path is None:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            save_path = tmp.name
    save_video(
        tensor=tensor[None],
        save_file=save_path,
        fps=fps,
        nrow=1,
        normalize=True,
        value_range=(-1, 1),
    )
    return save_path


def generate_video(task: str,
                   ckpt_dir: str,
                   prompt: str,
                   image: Image.Image,
                   size_key: str,
                   frame_num: int,
                   sampling_steps: int,
                   shift: float,
                   guide_low: float,
                   guide_high: float,
                   solver: str,
                   seed: int,
                   offload_model: bool,
                   convert_model_dtype: bool,
                   t5_cpu: bool):
    _require_cuda()

    prompt = prompt.strip()
    if not prompt:
        raise gr.Error("Prompt cannot be empty.")

    frame_num = _normalize_frame_num(frame_num)
    guide_scale = (float(guide_low), float(guide_high))
    sampling_steps = int(sampling_steps)
    seed = int(seed)

    pipeline = _load_pipeline(task, ckpt_dir, convert_model_dtype, t5_cpu)
    cfg = WAN_CONFIGS[task]

    if task == "i2v-A14B":
        frame_num = cfg.frame_num
        guide_scale = (float(guide_low), float(guide_low))

    # Create output directory
    output_dir = Path("./output")
    output_dir.mkdir(exist_ok=True)
    
    # Generate unique filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_stem = f"{task}_{timestamp}"
    video_path = output_dir / f"{file_stem}.mp4"

    if task == "t2v-A14B":
        video = pipeline.generate(
            prompt,
            size=SIZE_CONFIGS[size_key],
            frame_num=frame_num,
            shift=float(shift),
            sample_solver=solver,
            sampling_steps=sampling_steps,
            guide_scale=guide_scale,
            seed=seed,
            offload_model=offload_model,
        )
        # Save prompt as text file
        prompt_path = output_dir / f"{file_stem}.txt"
        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write(prompt)
    elif task == "ti2v-5B":
        # ti2v-5B supports both t2v and i2v modes
        if image is not None:
            # I2V mode
            video = pipeline.generate(
                prompt,
                img=image.convert("RGB"),
                max_area=MAX_AREA_CONFIGS[size_key],
                frame_num=frame_num,
                shift=float(shift),
                sample_solver=solver,
                sampling_steps=sampling_steps,
                guide_scale=guide_scale,
                seed=seed,
                offload_model=offload_model,
            )
            # Save input image
            image_path = output_dir / f"{file_stem}.png"
            image.save(image_path)
        else:
            # T2V mode
            video = pipeline.generate(
                prompt,
                size=SIZE_CONFIGS[size_key],
                frame_num=frame_num,
                shift=float(shift),
                sample_solver=solver,
                sampling_steps=sampling_steps,
                guide_scale=guide_scale,
                seed=seed,
                offload_model=offload_model,
            )
        # Save prompt as text file
        prompt_path = output_dir / f"{file_stem}.txt"
        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write(prompt)
    else:
        # i2v-A14B
        if image is None:
            raise gr.Error("Please upload an initial image for I2V generation.")
        video = pipeline.generate(
            prompt,
            img=image.convert("RGB"),
            max_area=MAX_AREA_CONFIGS[size_key],
            frame_num=frame_num,
            shift=float(shift),
            sample_solver=solver,
            sampling_steps=sampling_steps,
            guide_scale=guide_scale,
            seed=seed,
            offload_model=offload_model,
        )
        # Save input image
        image_path = output_dir / f"{file_stem}.png"
        image.save(image_path)
        # Save prompt as text file
        prompt_path = output_dir / f"{file_stem}.txt"
        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write(prompt)

    save_path = _save_video_tensor(video, cfg.sample_fps, str(video_path))
    info = (
        f"Task: {task} | Size: {size_key} | Steps: {sampling_steps} | "
        f"Guide: ({guide_scale[0]:.2f}, {guide_scale[1]:.2f}) | Seed: {seed}\n"
        f"Saved to: {video_path}"
    )

    del video
    torch.cuda.empty_cache()

    return save_path, info


DESCRIPTION = """# Wan2.2 Gradio Demo
- Supports text-to-video (T2V) with Wan2.2-T2V-A14B
- Supports image-to-video (I2V) with Wan2.2-I2V-A14B
- Supports text+image-to-video (TI2V) with Wan2.2-TI2V-5B (can work in both T2V and I2V modes)
- Expect long generation times and high VRAM usage (recommended: ≥80GB GPU)
"""


def update_task_ui(task: str):
    sizes = list(SUPPORTED_SIZES[task])
    default_size = "1280*720" if "1280*720" in sizes else sizes[0]
    guide_default = WAN_CONFIGS[task].sample_guide_scale
    if isinstance(guide_default, (int, float)):
        guide_default = (guide_default, guide_default)
    shift_default = WAN_CONFIGS[task].sample_shift
    steps_default = WAN_CONFIGS[task].sample_steps
    frame_default = WAN_CONFIGS[task].frame_num
    is_i2v = task == "i2v-A14B"
    # ti2v-5B supports optional image input
    show_image = task in ["i2v-A14B", "ti2v-5B"]

    return (
        gr.update(choices=sizes, value=default_size),
        gr.update(visible=show_image),
        gr.update(value=guide_default[0]),
        gr.update(
            value=guide_default[0] if is_i2v else guide_default[1],
            interactive=not is_i2v,
        ),
        gr.update(value=shift_default),
        gr.update(value=frame_default, interactive=not is_i2v),
        gr.update(value=steps_default),
    )


with gr.Blocks(title="Wan2.2 Demo") as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        task = gr.Radio(
            label="Task",
            choices=["t2v-A14B", "i2v-A14B", "ti2v-5B"],
            value="t2v-A14B",
        )
        ckpt_dir = gr.Dropdown(
            label="Checkpoint directory",
            choices=["./Wan2.2-T2V-A14B", "./Wan2.2-I2V-A14B", "./Wan2.2-TI2V-5B"],
            value="./Wan2.2-T2V-A14B",
        )
        solver = gr.Dropdown(
            label="Sampler",
            choices=["unipc", "dpm++"],
            value="unipc",
        )

    with gr.Row():
        prompt = gr.Textbox(
            label="Prompt",
            lines=4,
            value="Two anthropomorphic cats in comfy boxing gear fight intensely on a spotlighted stage.",
        )

    image_input = gr.Image(
        label="Reference image (for I2V)",
        type="pil",
        visible=False,
    )

    with gr.Row():
        size = gr.Dropdown(
            label="Output resolution",
            choices=list(SUPPORTED_SIZES["t2v-A14B"]),
            value="832*480",
        )
        frame_num = gr.Slider(
            label="Frame count (4n+1)",
            minimum=21,
            maximum=125,
            step=4,
            value=WAN_CONFIGS["t2v-A14B"].frame_num,
        )
        sampling_steps = gr.Slider(
            label="Sampling steps",
            minimum=20,
            maximum=80,
            step=1,
            value=WAN_CONFIGS["t2v-A14B"].sample_steps,
        )

    with gr.Row():
        guide_low = gr.Slider(
            label="CFG (low noise)",
            minimum=0.0,
            maximum=10.0,
            step=0.1,
            value=WAN_CONFIGS["t2v-A14B"].sample_guide_scale[0],
        )
        guide_high = gr.Slider(
            label="CFG (high noise)",
            minimum=0.0,
            maximum=10.0,
            step=0.1,
            value=WAN_CONFIGS["t2v-A14B"].sample_guide_scale[1],
        )
        shift = gr.Slider(
            label="Shift",
            minimum=0.0,
            maximum=15.0,
            step=0.1,
            value=WAN_CONFIGS["t2v-A14B"].sample_shift,
        )

    with gr.Row():
        seed = gr.Number(
            label="Seed (-1 for random)",
            value=-1,
            precision=0,
        )
        offload_model = gr.Checkbox(
            label="Offload model to CPU between steps",
            value=False,
        )
        convert_model_dtype = gr.Checkbox(
            label="Convert DiT weights to config dtype",
            value=True,
        )
        t5_cpu = gr.Checkbox(
            label="Keep T5 encoder on CPU",
            value=False,
        )

    generate_btn = gr.Button("Generate", variant="primary")

    with gr.Row():
        video_output = gr.Video(label="Generated video")
        info_output = gr.Textbox(label="Generation info", interactive=False)

    task.change(
        update_task_ui,
        inputs=[task],
        outputs=[size, image_input, guide_low, guide_high, shift, frame_num,
                 sampling_steps],
    )

    generate_btn.click(
        fn=generate_video,
        inputs=[
            task,
            ckpt_dir,
            prompt,
            image_input,
            size,
            frame_num,
            sampling_steps,
            shift,
            guide_low,
            guide_high,
            solver,
            seed,
            offload_model,
            convert_model_dtype,
            t5_cpu,
        ],
        outputs=[video_output, info_output],
    )

    demo.queue(max_size=1)


if __name__ == "__main__":
    demo.launch(share=True)
