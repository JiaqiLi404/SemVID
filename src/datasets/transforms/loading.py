import time
import os
import pickle
import torch
import random
import pandas as pd
import numpy as np

from ..builder import PIPELINES
from torch.nn import functional as F

data_cache = {}


@PIPELINES.register_module()
class LoadFeats:
    def __init__(self, feat_format, prefix="", suffix="", offset=None, random_aug_times=0, dict_data=None, cache=False):
        self.feat_format = feat_format
        self.prefix = prefix
        self.suffix = suffix
        self.offset = offset
        self.random_aug_times = random_aug_times
        self.dict_data = dict_data
        self.cache = cache
        # check feat format
        if isinstance(self.feat_format, str):
            check_feat_format(self.feat_format)
        elif isinstance(self.feat_format, list):
            for feat_format in self.feat_format:
                check_feat_format(feat_format)

    def get_feature_path(self, data_path, video_name, feat_format):
        rand_n = ""
        if self.random_aug_times != 0:
            rand_n = f"_{random.randint(0, self.random_aug_times - 1)}"
        return os.path.join(data_path, f"{self.prefix}{video_name}{self.suffix}{rand_n}.{feat_format}")

    def __call__(self, results):
        global data_cache
        video_name = results["video_name"]
        # if offset_frame is less than 0, then we should truncate the video
        offset_frames = results.get("offset_frames", 0)
        offset_frames = max(-offset_frames, 0) if self.offset is None else self.offset

        if video_name not in data_cache:
            if isinstance(results["data_path"], str):
                file_path = self.get_feature_path(results["data_path"], video_name, self.feat_format)
                feats = load_single_feat(file_path, self.feat_format)
            elif isinstance(results["data_path"], list):
                feats = []

                # check if the feat_format is a list
                if isinstance(self.feat_format, str):
                    self.feat_format = [self.feat_format] * len(results["data_path"])

                for data_path, feat_format in zip(results["data_path"], self.feat_format):
                    file_path = self.get_feature_path(data_path, video_name, feat_format)
                    feats.append(load_single_feat(file_path, feat_format))

                max_len = max([feat.shape[0] for feat in feats])
                for i in range(len(feats)):
                    if feats[i].shape[0] != max_len:
                        # assume the first dimension is T
                        tmp_feat = F.interpolate(
                            torch.Tensor(feats[i]).permute(1, 0).unsqueeze(0),
                            size=max_len,
                            mode="linear",
                            align_corners=False,
                        ).squeeze(0)
                        feats[i] = tmp_feat.permute(1, 0).numpy()
                feats = np.concatenate(feats, axis=1)

            if self.dict_data is not None:
                if isinstance(feats, list):
                    feats = [d[self.dict_data] for d in feats]
                elif isinstance(feats, dict):
                    feats = feats[self.dict_data]
                feats = np.array(feats).astype(np.float32)

            # sample the feature
            sample_stride = results.get("sample_stride", 1)
            if self.dict_data is not None and sample_stride > 1:
                feats = feats[offset_frames::sample_stride]
        else:
            feats = data_cache[video_name]

        if self.cache:
            data_cache[video_name] = feats

        results["feats"] = feats
        return results

    def __repr__(self):
        repr_str = f"{self.__class__.__name__}(" f"feat_format={self.feat_format}"
        return repr_str


def load_single_feat(file_path, feat_format):
    try:
        if feat_format == "npy":
            feats = read_from_npy(file_path)
        elif feat_format == "npz":
            feats = read_from_npz(file_path)
        elif feat_format == "pt" or feat_format == "pth":
            feats = read_from_tensor(file_path)
        elif feat_format == "csv":
            feats = read_from_csv(file_path)
        elif feat_format == "pkl":
            feats = read_from_pkl(file_path)
    except Exception as e:
        print("Missing data:", file_path)
        print(e)
        exit()
    return feats


def check_feat_format(feat_format):
    assert feat_format in ["npy", "npz", "pt", "csv", "pkl", 'pth'], print(f"not support {feat_format}")


def read_from_tensor(file_path):
    feats = torch.load(file_path, weights_only=False, map_location="cpu").float()
    return feats


def read_from_npy(file_path):
    feats = np.load(file_path).astype(np.float32)
    return feats


def read_from_npz(file_path):
    feats = np.load(file_path)["feats"].astype(np.float32)
    return feats


def read_from_csv(file_path):
    feats = pd.read_csv(file_path, dtype="float32").to_numpy()
    feats = feats.astype(np.float32)
    return feats


def read_from_pkl(file_path):
    feats = pickle.load(open(file_path, "rb"))
    if isinstance(feats, np.ndarray):
        feats = feats.astype(np.float32)
    return feats


@PIPELINES.register_module()
class LoadRawFeats(LoadFeats):
    def __init__(self, feat_format, prefix="", suffix="", offset=None, stride=None, load_keys=None):
        super().__init__(feat_format, prefix, suffix, offset)
        self.stride = stride
        self.load_keys = load_keys

    def __call__(self, results):
        video_name = results["video_name"]
        # if offset_frame is less than 0, then we should truncate the video
        offset_frames = results.get("offset_frames", 0)
        offset_frames = max(-offset_frames, 0) if self.offset is None else self.offset

        files = []
        ori_filepath = os.path.join(results["data_path"], f"{self.prefix}{video_name}{self.suffix}.{self.feat_format}")
        if os.path.exists(ori_filepath):
            files.append(ori_filepath)
            n = 1
        else:
            loop = True
            n = 0
            while loop:
                filepath = os.path.join(results["data_path"],
                                        f"{self.prefix}{video_name}{self.suffix}_{str(n)}.{self.feat_format}")
                if os.path.exists(filepath):
                    files.append(filepath)
                    n += 1
                else:
                    loop = False

        features = []
        with torch.no_grad():
            for i in range(len(files)):
                loaded_features = torch.load(files[i], weights_only=True, map_location="cpu")
                features.append(loaded_features)
                del loaded_features
        new_features = {}
        for b_i, sample_feats in enumerate(features):
            for feat_name, feat in sample_feats.items():
                if self.load_keys is not None and feat_name not in self.load_keys:
                    continue
                if feat_name not in new_features:
                    new_features[feat_name] = 1 / n * feat if type(feat) == torch.Tensor else feat
                elif type(feat) == list:
                    new_features[feat_name].extend(feat)
                elif type(feat) == torch.Tensor:
                    new_features[feat_name] = new_features[feat_name] + 1 / n * feat
        assert len(new_features) > 0, f"raw feature for {video_name} is not found"
        # sample the feature
        sample_stride = results.get("sample_stride", 1) if self.stride is None else self.stride
        assert sample_stride > 0, "sample_stride should be greater than 0"
        # stack the features
        for feat_name, feat in new_features.items():
            if type(feat) == list and type(feat[0]) == torch.Tensor:
                new_features[feat_name] = torch.stack(feat[offset_frames::sample_stride], dim=0)
                new_features[feat_name] = new_features[feat_name].detach()
            elif type(feat) == torch.Tensor:
                new_features[feat_name] = new_features[feat_name].detach()

        results["feats"] = new_features
        del features
        return results


@PIPELINES.register_module()
class RawTrunc:
    def __init__(self, window_size=None, trunc_thresh=0.75):
        self.window_size = window_size
        self.trunc_thresh = trunc_thresh

    def __call__(self, results):
        window_size = results.get("window_size", None) if self.window_size is None else self.window_size
        start_idx = results.get("trunc_start", results.get('feature_start_idx', None))
        trunc_len = results.get("trunc_len", None)
        if trunc_len is None and 'feature_end_idx' in results.keys():
            trunc_len = results.get('feature_end_idx') - start_idx
        elif trunc_len is None:
            trunc_len = window_size
        assert start_idx is not None and trunc_len is not None, "trunc_start and trunc_len should be set"
        assert window_size is not None, "window_size should be set"
        end_idx = start_idx + trunc_len
        process_segment = 'feature_start_idx' not in results.keys() and 'feature_end_idx' not in results.keys()
        process_segment = process_segment and 'gt_segments' in results.keys() and 'gt_labels' in results.keys()

        feats = results["feats"]
        valid_len = -1
        masks = None
        for name, feat in feats.items():
            if type(feat) == torch.Tensor and feat.shape[0] > 1:
                window_feats = feat[start_idx:end_idx]
                assert valid_len == -1 or valid_len == window_feats.shape[0], "all features should have the same length"
                valid_len = window_feats.shape[0]

                # if the valid window is smaller than window size, pad with -1
                if valid_len < window_size:
                    pad_shape = list(window_feats.shape)
                    pad_shape[0] = window_size - valid_len
                    pad_data = torch.zeros(*pad_shape)
                    window_feats = torch.cat((window_feats, pad_data), dim=0)

                # if we need padding mask (valid is 1, pad is 0)
                if masks is None:
                    if valid_len < window_size:
                        masks = torch.cat([torch.ones(valid_len), torch.zeros(window_size - valid_len)])
                    else:
                        masks = torch.ones(window_size)
                    results["masks"] = masks.bool()

                feats[name] = window_feats.float()

        results["feats"] = feats

        if process_segment:
            gt_segments = results["gt_segments"]
            gt_labels = results["gt_labels"]
            num_segs = gt_segments.shape[0]
            window = torch.as_tensor([start_idx, end_idx], dtype=torch.float32)

            # compute the intersection between the sampled window and all segments
            window = window[None].repeat(num_segs, 1)
            left = torch.maximum(window[:, 0], gt_segments[:, 0])
            right = torch.minimum(window[:, 1], gt_segments[:, 1])
            inter = (right - left).clamp(min=0)
            area_segs = torch.abs(gt_segments[:, 1] - gt_segments[:, 0])
            inter_ratio = inter / area_segs

            # only select those segments over the thresh
            seg_idx = inter_ratio >= self.trunc_thresh
            gt_segments = torch.stack((left[seg_idx], right[seg_idx]), dim=1)  # [N,2] in feature grids
            gt_segments = gt_segments - start_idx  # shift the time stamps due to truncation
            gt_labels = gt_labels[seg_idx]  # [N]
            results["gt_segments"] = gt_segments
            results["gt_labels"] = gt_labels

        return results


@PIPELINES.register_module()
class SlidingWindowTrunc:
    """This is used for sliding window dataset, which will give a window start and window end in the result dict,
    and we will extract the window features, also pad to fixed length"""

    def __init__(self, with_mask=True):
        self.with_mask = with_mask

    def __call__(self, results):
        assert "window_size" in results.keys(), "should have window_size as a key"
        assert isinstance(results["feats"], torch.Tensor)
        window_size = results["window_size"]

        feats_length = results["feats"].shape[0]
        start_idx = min(results["feature_start_idx"], feats_length)
        end_idx = min(results["feature_end_idx"] + 1, feats_length)

        window_feats = results["feats"][start_idx:end_idx]
        valid_len = window_feats.shape[0]

        # if the valid window is smaller than window size, pad with -1
        if valid_len < window_size:
            pad_data = torch.zeros(window_size - valid_len, window_feats.shape[1])
            window_feats = torch.cat((window_feats, pad_data), dim=0)

        # if we need padding mask (valid is 1, pad is 0)
        if self.with_mask:
            if valid_len < window_size:
                masks = torch.cat([torch.ones(valid_len), torch.zeros(window_size - valid_len)])
            else:
                masks = torch.ones(window_size)
            results["masks"] = masks.bool()

        results["feats"] = window_feats.float()
        return results


def random_trunc(feats, trunc_len, gt_segments, gt_labels, crop_ratio, trunc_thresh, no_trunc, has_action, offset=0,
                 max_num_trials=200):
    feat_len = feats.shape[0]
    num_segs = gt_segments.shape[0]

    trunc_len = trunc_len
    if feat_len <= trunc_len:
        if crop_ratio == None:  # do nothing
            return feats, gt_segments, gt_labels, 0, feat_len
        else:  # randomly crop the seq by setting trunc_len to a value in [l, r]
            trunc_len = random.randint(
                max(round(crop_ratio[0] * feat_len), 1),
                min(round(crop_ratio[1] * feat_len), feat_len),
            )
            # corner case
            if feat_len == trunc_len:
                return feats, gt_segments, gt_labels, 0, feat_len

    # try a few times till a valid truncation with at least one action
    for _ in range(max_num_trials):
        # sample a random truncation of the video feats
        st = random.randint(0, feat_len - trunc_len)
        ed = st + trunc_len
        window = torch.as_tensor([st, ed], dtype=torch.float32)

        # compute the intersection between the sampled window and all segments
        window = window[None].repeat(num_segs, 1)
        left = torch.maximum(window[:, 0] - offset, gt_segments[:, 0])
        right = torch.minimum(window[:, 1] + offset, gt_segments[:, 1])
        inter = (right - left).clamp(min=0)
        area_segs = torch.abs(gt_segments[:, 1] - gt_segments[:, 0])
        inter_ratio = inter / area_segs

        # only select those segments over the thresh
        seg_idx = inter_ratio >= trunc_thresh

        if no_trunc:
            # with at least one action and not truncating any actions
            seg_trunc_idx = (inter_ratio > 0.0) & (inter_ratio < 1.0)
            if (seg_idx.sum().item() > 0) and (seg_trunc_idx.sum().item() == 0):
                break
        elif has_action:
            # with at least one action
            if seg_idx.sum().item() > 0:
                break
        else:
            # without any constraints
            break

    feats = feats[st:ed, :]  # [T,C]
    gt_segments = torch.stack((left[seg_idx], right[seg_idx]), dim=1)  # [N,2] in feature grids
    gt_segments = gt_segments - st  # shift the time stamps due to truncation
    gt_labels = gt_labels[seg_idx]  # [N]
    trunc_start = st
    return feats, gt_segments, gt_labels, trunc_start, trunc_len


def trunc_features(feats, trunc_start, trunc_len, gt_segments, gt_labels, trunc_thresh, offset=0):
    feat_len = feats.shape[0]
    num_segs = gt_segments.shape[0]
    st = trunc_start
    ed = trunc_start + trunc_len
    feats = feats[st:ed, :]  # [T,C]
    window = torch.as_tensor([st, ed], dtype=torch.float32)

    # compute the intersection between the sampled window and all segments
    window = window[None].repeat(num_segs, 1)
    left = torch.maximum(window[:, 0] - offset, gt_segments[:, 0])
    right = torch.minimum(window[:, 1] + offset, gt_segments[:, 1])
    inter = (right - left).clamp(min=0)
    area_segs = torch.abs(gt_segments[:, 1] - gt_segments[:, 0])
    inter_ratio = inter / area_segs

    # only select those segments over the thresh
    seg_idx = inter_ratio >= trunc_thresh
    gt_segments = torch.stack((left[seg_idx], right[seg_idx]), dim=1)  # [N,2] in feature grids
    gt_segments = gt_segments - st  # shift the time stamps due to truncation
    gt_labels = gt_labels[seg_idx]  # [N]
    return feats, gt_segments, gt_labels


@PIPELINES.register_module()
class RandomTrunc:
    """Crops features within a window such that they have a large overlap with ground truth segments.
    Withing the cropping ratio, the length is sampled."""

    def __init__(
            self,
            trunc_len,
            trunc_thresh,
            crop_ratio=None,
            max_num_trials=200,
            has_action=True,
            no_trunc=False,
            pad_value=0,
            channel_first=False,
    ):
        self.trunc_len = trunc_len
        self.trunc_thresh = trunc_thresh
        self.crop_ratio = crop_ratio
        self.max_num_trials = max_num_trials
        self.has_action = has_action
        self.no_trunc = no_trunc
        self.pad_value = pad_value
        self.channel_first = channel_first

    def random_trunc(self, feats, gt_segments, gt_labels, offset):
        return random_trunc(feats, self.trunc_len, gt_segments, gt_labels, self.crop_ratio, self.trunc_thresh,
                            self.no_trunc, self.has_action, offset, self.max_num_trials)

    def pad_features(self, feats):
        feat_len = feats.shape[0]
        if feat_len < self.trunc_len:
            feats_pad = torch.ones((self.trunc_len - feat_len,) + feats.shape[1:]) * self.pad_value
            feats = torch.cat([feats, feats_pad], dim=0)
            masks = torch.cat([torch.ones(feat_len), torch.zeros(self.trunc_len - feat_len)])
            return feats, masks
        else:
            return feats, torch.ones(feat_len)

    def __call__(self, results):
        assert isinstance(results["feats"], torch.Tensor)
        offset = 0

        if self.channel_first:
            results["feats"] = results["feats"].transpose(0, 1)  # [C,T] -> [T,C]

        trunc_start = results.get("trunc_start", None)
        trunc_len = results.get("trunc_len", None)
        if trunc_start is None or trunc_len is None:
            feats, gt_segments, gt_labels, trunc_start, trunc_len = self.random_trunc(
                results["feats"],
                results["gt_segments"],
                results["gt_labels"],
                offset,
            )
        else:
            gt_segments = results["gt_segments"]
            gt_labels = results["gt_labels"]
            feats = results["feats"]
            feats, gt_segments, gt_labels = trunc_features(feats, trunc_start, trunc_len, gt_segments, gt_labels,
                                                           self.trunc_thresh, offset)

        # pad the features to the fixed length
        feats, masks = self.pad_features(feats)

        results["feats"] = feats.float()
        results["masks"] = masks.bool()
        results["gt_segments"] = gt_segments
        results["gt_labels"] = gt_labels
        results["trunc_start"] = trunc_start
        results["trunc_len"] = trunc_len

        if self.channel_first:
            results["feats"] = results["feats"].transpose(0, 1)  # [T,C] -> [C,T]
        return results
