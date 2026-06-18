import json
import os

import numpy as np
from mmengine.dataset import Compose
from datasets import Dataset as HFDataset
import random
from tqdm import tqdm
from decord import VideoReader
from decord import cpu
import pandas as pd

from .utils import align_record_names
from ..builder import DATASETS


@DATASETS.register_module()
class BaseDataset:
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
            **kwargs
    ):
        # basic settings
        if is_train:
            self.split = "train"
            self.ann_file = train_data_path
            self.video_path = train_video_folder
        else:
            self.split = "val"
            self.ann_file = eval_data_path
            self.video_path = eval_video_folder if eval_video_folder is not None else train_video_folder

        self.subtitle_folder = subtitle_folder
        self.dataset_name = dataset_name
        self.video_name_only = video_name_only
        self.video_name_postfix = video_name_postfix
        self.is_shuffle = is_shuffle
        self.sample_amount = sample_amount
        self.add_choice_indices = add_choice_indices
        self.seed = seed

        self.logger = logger.info if logger is not None else print
        self.video_pipeline = Compose(video_pipeline) if video_pipeline is not None else None

        # build list then HF Dataset
        data_list = self._build_data_list()

        # sample/shuffle on python list（也可以改成 HF 的 select/shuffle，这里保持你原逻辑）
        if self.sample_amount is not None:
            rng = random.Random(self.seed)
            data_list = rng.sample(data_list, min(self.sample_amount, len(data_list)))
        if self.is_shuffle:
            rng = random.Random(self.seed)
            rng.shuffle(data_list)

        self.dataset = HFDataset.from_list(data_list)

        self.logger(f"{self.dataset_name} {self.split} dataset: {len(self.dataset)} videos.")

    def _build_data_list(self):
        if self.ann_file is None:
            return []

        json_path = self.ann_file.replace('.txt', '.json')
        json_path = json_path.replace('.parquet', '.json')
        if self.ann_file.endswith(".json"):
            with open(self.ann_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        elif self.ann_file.endswith('.txt') and os.path.exists(json_path):
            self.logger(f"Loading json format annotations from {json_path}...")
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        elif self.ann_file.endswith('.txt'):
            data = load_txt2json(self.ann_file)
        elif self.ann_file.endswith('.parquet') and os.path.exists(json_path):
            self.logger(f"Loading parquet format annotations from {json_path}...")
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        elif self.ann_file.endswith('.parquet'):
            data = load_parquet2json(self.ann_file)
        else:
            raise ValueError(f"Invalid annotation file format: {self.ann_file}")

        data_list = []
        for item in tqdm(data, desc=f"Processing {self.split} dataset"):
            if isinstance(item, str):
                # {"vid":{parameters}}
                video_path = item.replace(":", "_") + self.video_name_postfix
                if self.video_name_only:
                    video_path = os.path.basename(video_path)
                video_path = os.path.join(self.video_path, video_path)

                data[item] = align_record_names(data[item])
                sentence = data[item].get("sentence", data[item].get("sentences", None))
                duration = data[item].get("duration", None)
                video_start = data[item].get("video_start", None)
                video_end = data[item].get("video_end", None)
                type = data[item].get("type", None)
                choices = data[item].get("choices", None)
                solution = data[item].get("solution", data[item].get("timestamp", data[item].get("timestamps", None)))
                subtitle_path = data[item].get("subtitle_path", None) if self.subtitle_folder is not None else None
                item_id = item
            else:
                item = align_record_names(item)
                video_path = item.get("video")
                video_path = video_path.replace(":", "_") + self.video_name_postfix
                if self.video_name_only:
                    video_path = os.path.basename(video_path)
                video_path = os.path.join(self.video_path, video_path)

                sentence = item.get("sentence", None)
                duration = item.get("duration", None)
                video_start = item.get("video_start", None)
                video_end = item.get("video_end", None)
                item_id = item.get("video", os.path.basename(video_path))
                type = item.get("type", None)
                choices = item.get("choices", None)
                solution = item.get("solution", item.get("timestamp", None))
                subtitle_path = item.get("subtitle_path", None) if self.subtitle_folder is not None else None

            if choices is not None and isinstance(solution, int):
                solution = chr(ord("A") + solution)
                if self.add_choice_indices:
                    for ind in range(len(choices)):
                        ind_str = chr(ord("A") + ind)
                        choices[ind] = f"{ind_str}. {choices[ind]}"

            subtitles = None
            if subtitle_path is not None:
                if not subtitle_path.endswith(".json"):
                    subtitle_path = subtitle_path + ".json"
                subtitle_path = os.path.join(self.subtitle_folder, subtitle_path)
                if os.path.isfile(subtitle_path):
                    subtitles = load_subtitles(subtitle_path)

            assert video_path is not None and sentence is not None, f"Invalid data format for video {item}"

            if not os.path.isfile(video_path):
                self.logger(f"Video {video_path} not found.")
                continue

            if duration is None:
                # load video by decord and get duration
                duration = duration_decord(video_path)

            if isinstance(sentence, str):
                sentence = [sentence]
                solution = [solution]

            assert len(sentence) == len(solution), f"Invalid sentence and timestamps length for video {item}"

            for query_index, (sen, tim) in enumerate(zip(sentence, solution)):
                sen = sen.strip().lower()
                if sen.endswith("."):
                    sen = sen[:-1]

                if isinstance(tim, list) or isinstance(tim, tuple):
                    tim = [float(tim[0]), float(tim[1])]

                example = {
                    "id": f"{item_id}_{query_index}_{tim}",
                    "task_type": "tg",
                    "type": type,
                    "problem": sen,
                    "choices": choices,
                    "solution": tim,
                    "video_path": video_path,
                    "durations": duration,
                    "video_start": video_start,
                    "video_end": video_end,
                    "preprocessed_path": "",
                    "use_preprocessed": False,
                }
                if subtitles is not None:
                    example["subtitles"] = subtitles
                data_list.append(example)

        if not os.path.exists(json_path):
            self.logger(f"Saving json format annotations to {json_path}...")
            save_data = []
            for item in data_list:
                video_name = os.path.basename(item["video_path"])
                if self.video_name_postfix:
                    video_name = os.path.splitext(video_name)[0]
                save_data.append({
                    "video": video_name,
                    "solution": item["solution"],
                    "sentence": item["problem"],
                    "video_start": item["video_start"],
                    "video_end": item["video_end"],
                    "duration": item["durations"],
                    "type": item["type"],
                    "choices": item["choices"]
                })
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(save_data, f, ensure_ascii=False, indent=4, sort_keys=False)

        return data_list

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # 注意：如果子类对 self.dataset 做了 with_transform，这里会自动触发 transform
        ex = self.dataset[idx]
        # 如果你希望 mmengine pipeline 在这里执行也可以，但更建议放到 transform 里
        return ex

    def __getattr__(self, name):
        # 透传 HF Dataset 的常用方法（shuffle/select/with_transform等）
        if name in ("dataset", "__getstate__", "__setstate__"):
            raise AttributeError
        return getattr(self.dataset, name)


def load_txt2json(txt_path: str):
    data = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            if len(line.strip()) == 0:
                continue
            line = line.strip().split('##')
            attributes = line[0].split(' ')
            data.append({
                "video": attributes[0],
                "timestamp": [float(attributes[1]), float(attributes[2])],
                "sentence": line[1],
                "video_start": None,
                "video_end": None,
                "duration": None
            })
    return data


def duration_decord(path: str) -> float:
    vr = VideoReader(path, ctx=cpu(0))
    last = len(vr) - 1
    ts = vr.get_frame_timestamp(last)
    a = np.asarray(ts).astype(np.float64).reshape(-1)  # 展平为 1D
    if a.size >= 2:
        return round(float(a[1]), 1)  # (start, end) -> 取 end
    if a.size == 1:
        return round(float(a[0]), 1)
    raise RuntimeError(f"Empty timestamp: {ts}")


def load_parquet2json(parquet_path: str):
    df = pd.read_parquet(parquet_path)
    items = df.to_dict(orient="records")
    items = [align_record_names(it, drop_source_keys=True) for it in items]
    return items


def load_subtitles(subtitle_path: str):
    with open(subtitle_path, "r", encoding="utf-8") as f:
        subtitles = json.load(f)
    return subtitles
