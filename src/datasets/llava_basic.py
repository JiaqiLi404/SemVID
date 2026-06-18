# llava_basic.py
import os
import copy
import math
from typing import Any, Dict, List, Optional, Tuple, Union

from transformers import AutoProcessor
from decord import VideoReader, cpu
import numpy as np
from PIL import Image

from .base import BaseDataset
from .builder import DATASETS, build_prompt


@DATASETS.register_module()
class BasicDatasetForLlava(BaseDataset):
    """
    transformers.llava_onevision oriented dataset wrapper, with support for:
      - video-only: load frames at `sample_fps` (or uniform) up to `max_num_frames`, optionally resize long side
      - image-text interleaved (+ subtitles): same sampled frames, but represented as <image> placeholders and
        optional subtitle texts inserted in-between (lmms-eval LongVideoBench style).

    Adapts video-FlexReduc-like video loading knobs:
      - sample_fps: target sampling fps (e.g., 2)
      - max_num_frames: cap frames (e.g., 1024)
      - longsize_resolution: resize long side (e.g., 682)
      - force_even_frames: make sampled frame count even (video-FlexReduc does this)
      - longvideo_kwargs: stored into data for downstream model/runner usage (chunking/compression configs, etc.)

    Output keys added:
      - prompt: structured chat messages (list[dict])
      - prompt_text: chat template string (if processor available)
      - processor_inputs: tokenized inputs (if processor available)
      - media_mode: "video" or "interleaved"
      - video_inputs / image_inputs: decoded frames (nested list for one sample)
      - video_kwargs: reserved (currently empty; keep for downstream extensions)
    """

    def __init__(
            self,
            dataset_name,
            train_data_path,
            eval_data_path,
            train_video_folder,
            eval_video_folder,
            subtitle_folder=None,
            video_name_only=False,
            video_name_postfix="",
            is_shuffle=True,
            video_pipeline=None,
            sample_amount=None,
            add_choice_indices=False,
            logger=None,
            is_train=True,
            seed=42,
            # ---- LLaVA OneVision specific ----
            eval_model_path: Optional[str] = None,
            prompt_cls=None,
            prompt_task: str = "temporal_grounding",
            skip_loading_video: bool = False,
            # ---- modes ----
            use_interleaved: bool = False,
            # ---- video-FlexReduc-like loading knobs ----
            sample_fps: Optional[float] = None,  # e.g., 2
            max_num_frames: int = 32,  # e.g., 1024
            longsize_resolution: Optional[int] = None,  # e.g., 682
            force_even_frames: bool = False,
            frame_extraction_fps: float = 30.0,  # for frames-dir fallback
            keep_raw_media: bool = False,  # debug only
            **kwargs,
    ):
        super().__init__(
            dataset_name=dataset_name,
            train_data_path=train_data_path,
            eval_data_path=eval_data_path,
            train_video_folder=train_video_folder,
            eval_video_folder=eval_video_folder,
            subtitle_folder=subtitle_folder,
            video_name_only=video_name_only,
            video_name_postfix=video_name_postfix,
            is_shuffle=is_shuffle,
            video_pipeline=video_pipeline,
            sample_amount=sample_amount,
            add_choice_indices=add_choice_indices,
            logger=logger,
            is_train=is_train,
            seed=seed,
            **kwargs,
        )

        self.prompt_cls = build_prompt(prompt_cls)
        self.prompt_func = getattr(self.prompt_cls, f"for_{prompt_task}")

        self.skip_loading_video = bool(skip_loading_video)
        self.use_interleaved = bool(use_interleaved)

        self.sample_fps = float(sample_fps) if sample_fps is not None else None
        self.max_num_frames = int(max_num_frames)
        self.longsize_resolution = int(longsize_resolution) if longsize_resolution is not None else None
        self.force_even_frames = bool(force_even_frames)
        self.frame_extraction_fps = float(frame_extraction_fps)
        self.keep_raw_media = bool(keep_raw_media)

        self.dataset = self.dataset.with_transform(self._transform)

    def _process_one(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = copy.deepcopy(data)

        clip_start = float(data.get("video_start") or 0.0)
        clip_end = data.get("video_end")
        clip_end = float(clip_end) if clip_end is not None else None
        full_duration = float(data.get("durations")) if data.get("durations") is not None else 0.0
        media_mode = "interleaved" if self.use_interleaved else "video"

        prompts = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "path": data["video_path"]},
                    {"type": "text", "text": self.prompt_func(event=data["problem"], choices=data["choices"])},
                ],
            }
        ]

        video_decode_kwargs = {
            'max_num_frames': self.max_num_frames,
            'sample_fps': self.sample_fps,
            'longsize_resolution': self.longsize_resolution,
            'force_even_frames': self.force_even_frames,
            'frame_extraction_fps': self.frame_extraction_fps,
        }
        if self.skip_loading_video:
            image_inputs = video_inputs = video_kwargs = frame_ts = frames = sampling_fps = None
        else:
            prompts, image_inputs, video_inputs, video_kwargs, frames, frame_ts, sampling_fps \
                = process_vision_info_llava(prompts,
                                            subtitles=data.get("subtitles"),
                                            clip_starts=clip_start,
                                            clip_ends=clip_end,
                                            durations=full_duration,
                                            media_modes=media_mode,
                                            video_decode_kwargs=video_decode_kwargs
                                            )
            prompts = prompts[0]
            frames = frames[0]
            frame_ts = frame_ts[0]

        data.update(
            {
                "media_mode": media_mode,
                "prompt": prompts,
                "image_inputs": image_inputs[0] if image_inputs is not None else None,
                "video_inputs": video_inputs[0] if video_inputs is not None else None,
                "video_kwargs": video_kwargs[0] if video_inputs is not None else None,
                "sampling_fps": sampling_fps[0] if sampling_fps is not None else None,
                "video_decode_kwargs": video_decode_kwargs,
            }
        )

        if self.keep_raw_media:
            data["raw_frames"] = frames
            data["raw_frame_timestamps"] = frame_ts
        else:
            data["raw_frames"] = None
            data["raw_frame_timestamps"] = None

        if self.video_pipeline is not None:
            data = self.video_pipeline(data)

        return data

    def _transform(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        any_key = next(iter(examples.keys()))
        is_batched = isinstance(examples[any_key], list)

        if not is_batched:
            return self._process_one(examples)

        outs = []
        n = len(examples[any_key])
        for i in range(n):
            one = {k: v[i] for k, v in examples.items()}
            outs.append(self._process_one(one))

        merged = {k: [o.get(k) for o in outs] for k in outs[0].keys()}
        return merged

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def process_vision_info_llava(prompts_batch, subtitles=None, clip_starts=None, clip_ends=None, durations=None,
                              media_modes="video", video_decode_kwargs={}):
    frames: Optional[List[Image.Image]] = None
    frame_ts: Optional[List[float]] = None
    sampling_fps: float = 0.0

    if not isinstance(prompts_batch[0], list):
        prompts_batch = [prompts_batch]
        subtitles = [subtitles]
        clip_starts = [clip_starts]
        clip_ends = [clip_ends]
        durations = [durations]
        media_modes = [media_modes]
        video_decode_kwargs = [video_decode_kwargs]

    image_inputs = video_inputs = None
    frames_list = []
    frame_ts_list = []
    sampling_fps_list = []
    video_kwargs=[]

    for prompts, subtitle, clip_st, clip_en, duration, media_mode, video_decode_kwarg in zip(prompts_batch, subtitles,
                                                                                             clip_starts, clip_ends,
                                                                                             durations,
                                                                                             media_modes,
                                                                                             video_decode_kwargs):
        # prompts[0]["content"] 中取 video block
        vb = None
        vb_idx = 0
        for i, c in enumerate(prompts[0]["content"]):
            if isinstance(c, dict) and c.get("type") == "video":
                vb = c
                vb_idx = i
                break
        if vb is None:
            raise ValueError("Cannot find video block in prompts[0]['content'].")
        # print(prompts)

        clip_st = 0.0 if clip_st is None else clip_st
        clip_en = duration if clip_en is None else clip_en
        if clip_en  < clip_st:
            clip_st, clip_en = clip_en, clip_st

        # if not self.skip_loading_video:
        frames, frame_ts, sampling_fps, clip_duration = _decode_video(
            video_path=vb["path"],
            start_sec=clip_st,
            end_sec=clip_en,
            **video_decode_kwarg
        )
        frames_list.append(frames)
        frame_ts_list.append(frame_ts)
        sampling_fps_list.append(sampling_fps)

        if media_mode == 'interleaved':
            prompts[0]["content"].pop(vb_idx)

            if subtitle:
                interleaved_content = _build_interleaved_content_from_subtitles(
                    frame_timestamps=frame_ts,
                    subtitles=subtitle,
                    subtitle_time_offset=clip_st,
                    clip_duration=clip_duration,
                )
            else:
                interleaved_content = [{"type": "image"} for _ in frame_ts]
            interleaved_content.extend(prompts[0]["content"])
            prompts[0]["content"] = interleaved_content

            if image_inputs is None:
                image_inputs = []
            image_inputs.append(frames)
            video_kwargs.append({})
        else:
            if video_inputs is None:
                video_inputs = []
            video_inputs.append(frames)
            video_kwargs.append({})

    return prompts_batch, image_inputs, video_inputs, video_kwargs, frames_list, frame_ts_list, sampling_fps_list


def _decode_video(
        video_path: str,
        start_sec: float,
        end_sec: Optional[float],
        max_num_frames: int = 1024,
        sample_fps: Optional[float] = 2,
        longsize_resolution: Optional[int] = 682,
        force_even_frames: bool = True,
        frame_extraction_fps: float = 30.0,
) -> Tuple[List[Image.Image], List[float], float, float]:
    if isinstance(video_path, str) and video_path.startswith("file://"):
        video_path = video_path[7:]

    if os.path.isdir(video_path):
        return _load_frames_from_dir(
            frames_dir=video_path,
            start_sec=start_sec,
            end_sec=end_sec,
            max_num_frames=max_num_frames,
            sample_fps=sample_fps,
            longsize_resolution=longsize_resolution,
            force_even_frames=force_even_frames,
            frame_extraction_fps=frame_extraction_fps,
        )
    return _load_frames_decord(
        video_path=video_path,
        start_sec=start_sec,
        end_sec=end_sec,
        max_num_frames=max_num_frames,
        sample_fps=sample_fps,
        longsize_resolution=longsize_resolution,
        force_even_frames=force_even_frames,
    )


def timestamp_to_seconds(timestamp: str) -> float:
    # "HH:MM:SS" or "HH:MM:SS.xxx"
    h, m, s = timestamp.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def _parse_subtitle_item(
        subtitle: Dict[str, Any], duration_fallback: float
) -> Optional[Tuple[float, float, str]]:
    """
    Supports two common subtitle json formats:
      1) {"timestamp":[start,end], "text": "..."} (end can be non-float -> treat as duration)
      2) {"start":"HH:MM:SS.xxx", "end":"HH:MM:SS.xxx", "line":"..."} (or "text")
    Returns (start_seconds, end_seconds, text) or None if cannot parse.
    """
    if not isinstance(subtitle, dict):
        return None

    if "timestamp" in subtitle:
        ts = subtitle.get("timestamp")
        if not (isinstance(ts, (list, tuple)) and len(ts) == 2):
            return None
        start, end = ts[0], ts[1]
        try:
            start = float(start)
        except Exception:
            return None

        if isinstance(end, (float, int)):
            end = float(end)
        else:
            end = float(duration_fallback)

        text = subtitle.get("text", "")
        if not isinstance(text, str):
            text = str(text)
        return float(start), float(end), text

    if "start" in subtitle and "end" in subtitle:
        try:
            start = timestamp_to_seconds(subtitle["start"])
            end = timestamp_to_seconds(subtitle["end"])
        except Exception:
            return None
        text = subtitle.get("line", subtitle.get("text", ""))
        if not isinstance(text, str):
            text = str(text)
        return float(start), float(end), text

    return None


def _resize_long_side(img: Image.Image, longsize: Optional[int]) -> Image.Image:
    """
    Follow video-FlexReduc behavior: if long side > longsize, scale down by factor, resample=NEAREST.
    """
    if longsize is None:
        return img
    try:
        longsize = int(longsize)
    except Exception:
        return img
    if longsize <= 0:
        return img

    w, h = img.size
    if max(w, h) <= longsize:
        return img
    factor = longsize / float(max(w, h))
    new_w = max(1, int(round(w * factor)))
    new_h = max(1, int(round(h * factor)))
    return img.resize((new_w, new_h), resample=Image.NEAREST)


def _compute_num_sampled_frames(
        total_frames: int,
        extraction_fps: float,
        max_num_frames: int,
        sample_fps: Optional[float],
        force_even: bool = True,
) -> int:
    """
    Match video-FlexReduc/demo.py logic:
      sample_frames = duration * sample_fps
      sample_frames = min(total_frames, max_num_frames, sample_frames)
      sample_frames = floor(sample_frames)
      sample_frames = int(sample_frames/2)*2  (ensure even)
    But we also guard against 0.
    """
    total_frames = int(max(0, total_frames))
    if total_frames <= 0:
        return 0

    max_num_frames = int(max(1, max_num_frames))
    extraction_fps = float(extraction_fps) if extraction_fps and extraction_fps > 0 else 30.0

    if sample_fps is None:
        sample_frames = min(total_frames, max_num_frames)
    else:
        sample_fps = float(sample_fps)
        duration = total_frames / extraction_fps
        sample_frames = duration * sample_fps
        sample_frames = min(float(total_frames), float(max_num_frames), float(sample_frames))
        sample_frames = math.floor(sample_frames)
        sample_frames = int(sample_frames)

    sample_frames = max(1, int(sample_frames))

    if force_even and sample_frames > 1:
        sample_frames = (sample_frames // 2) * 2
        if sample_frames < 2 and total_frames >= 2:
            sample_frames = 2

    sample_frames = min(sample_frames, total_frames)
    return int(sample_frames)


def _linspace_indices(total_frames: int, num: int) -> np.ndarray:
    if num <= 1:
        return np.array([0], dtype=np.int32)
    idx = np.linspace(0, total_frames - 1, num).astype(np.int32)
    # de-dup while preserving order
    _, first_pos = np.unique(idx, return_index=True)
    idx = idx[np.sort(first_pos)]
    return idx.astype(np.int32)


def _load_frames_decord(
        video_path: str,
        start_sec: float,
        end_sec: Optional[float],
        max_num_frames: int,
        sample_fps: Optional[float],
        longsize_resolution: Optional[int],
        force_even_frames: bool,
) -> Tuple[List[Image.Image], List[float], float, float]:
    """
    Load frames via decord within [start_sec, end_sec] and sample according to video-FlexReduc semantics.

    Returns:
      frames: list[PIL.Image] RGB
      frame_timestamps: list[float] relative to start_sec (seconds)
      sampling_fps: float (effective fps over the clipped segment)
    """
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    extraction_fps = float(vr.get_avg_fps()) if vr.get_avg_fps() else 30.0

    if end_sec is None or not isinstance(end_sec, (float, int)):
        # best-effort: end_sec from last frame timestamp, else from len/fps
        try:
            last = len(vr) - 1
            ts = vr.get_frame_timestamp(last)
            end_sec = float(np.asarray(ts).reshape(-1)[1])
        except Exception:
            end_sec = float(len(vr) / extraction_fps)

    start_sec=0.0 if start_sec is None else float(max(0.0, start_sec))
    end_sec = float(max(start_sec, float(end_sec)))

    start_frame = int(start_sec * extraction_fps)
    end_frame = int(end_sec * extraction_fps)
    start_frame = max(0, min(start_frame, len(vr) - 1))
    end_frame = max(start_frame, min(end_frame, len(vr) - 1))

    total_frames = int(end_frame - start_frame + 1)
    if total_frames <= 0:
        return [], [], 0.0

    num = _compute_num_sampled_frames(
        total_frames=total_frames,
        extraction_fps=extraction_fps,
        max_num_frames=max_num_frames,
        sample_fps=sample_fps,
        force_even=force_even_frames,
    )
    num = max(1, num)

    rel_idx = _linspace_indices(total_frames, num)
    frame_indices = (rel_idx + start_frame).astype(np.int64)

    batch = vr.get_batch(frame_indices.tolist())
    arr = batch.asnumpy() if hasattr(batch, "asnumpy") else np.asarray(batch)
    frames = [Image.fromarray(fr).convert("RGB") for fr in arr]
    if longsize_resolution is not None:
        frames = [_resize_long_side(im, longsize_resolution) for im in frames]

    frame_timestamps = [(int(idx) - start_frame) / extraction_fps for idx in frame_indices]
    duration = total_frames / extraction_fps
    sampling_fps = float(len(frames) / duration) if duration > 0 else 0.0
    return frames, frame_timestamps, sampling_fps, duration


def _load_frames_from_dir(
        frames_dir: str,
        start_sec: float,
        end_sec: Optional[float],
        max_num_frames: int,
        sample_fps: Optional[float],
        longsize_resolution: Optional[int],
        force_even_frames: bool,
        frame_extraction_fps: float = 30.0,
) -> Tuple[List[Image.Image], List[float], float, float]:
    """
    Support video-FlexReduc behavior: `video_path` can be a directory of extracted frames.
    Requires frame_extraction_fps (either provided or default 30).
    """
    files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir)])
    total_all = len(files)
    if total_all <= 0:
        return [], [], 0.0

    extraction_fps = float(frame_extraction_fps) if frame_extraction_fps and frame_extraction_fps > 0 else 30.0
    if end_sec is None or not isinstance(end_sec, (float, int)):
        end_sec = total_all / extraction_fps

    start_sec=0.0 if start_sec is None else float(max(0.0, start_sec))
    end_sec = float(max(start_sec, float(end_sec)))

    start_frame = int(start_sec * extraction_fps)
    end_frame = int(end_sec * extraction_fps)
    start_frame = max(0, min(start_frame, total_all - 1))
    end_frame = max(start_frame, min(end_frame, total_all - 1))

    total_frames = int(end_frame - start_frame + 1)
    num = _compute_num_sampled_frames(
        total_frames=total_frames,
        extraction_fps=extraction_fps,
        max_num_frames=max_num_frames,
        sample_fps=sample_fps,
        force_even=force_even_frames,
    )
    num = max(1, num)

    rel_idx = _linspace_indices(total_frames, num)
    frame_indices = (rel_idx + start_frame).astype(np.int64)

    frames = []
    for idx in frame_indices:
        img = Image.open(files[int(idx)]).convert("RGB")
        if longsize_resolution is not None:
            img = _resize_long_side(img, longsize_resolution)
        frames.append(img)

    frame_timestamps = [(int(idx) - start_frame) / extraction_fps for idx in frame_indices]
    duration = total_frames / extraction_fps
    sampling_fps = float(len(frames) / duration) if duration > 0 else 0.0
    return frames, frame_timestamps, sampling_fps, duration


def _build_interleaved_content_from_subtitles(
        frame_timestamps: List[float],
        subtitles: List[Dict[str, Any]],
        subtitle_time_offset: float,
        clip_duration: float,
) -> List[Dict[str, Any]]:
    """
    lmms-eval LongVideoBench-style interleaving:
      - emit {"type":"image"} placeholders for frames in time order
      - insert subtitle text if it overlaps at least one sampled frame in its span
    frame_timestamps are relative to clip start.
    subtitle_time_offset is (starting_timestamp_for_subtitles + video_start), so we subtract it.
    """
    content: List[Dict[str, Any]] = []
    cur_i = 0

    for sub in subtitles or []:
        parsed = _parse_subtitle_item(sub, duration_fallback=clip_duration)
        if parsed is None:
            continue

        start, end, text = parsed
        start -= float(subtitle_time_offset)
        end -= float(subtitle_time_offset)

        if end < start:
            start, end = end, start

        subtitle_center = (start + end) / 2.0

        for ft in frame_timestamps[cur_i:]:
            if ft <= subtitle_center:
                content.append({"type": "image"})
                cur_i += 1
            else:
                break

        if (end - start) < 1.0:
            end = subtitle_center + 0.5
            start = subtitle_center - 0.5

        covering = any((ft > start and ft < end) for ft in frame_timestamps)
        if covering and text.strip():
            content.append({"type": "text", "text": text.strip()})

    for _ in frame_timestamps[cur_i:]:
        content.append({"type": "image"})

    return content
