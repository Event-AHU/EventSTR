"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root或https://opensource.org/licenses/BSD-3-Clause
"""
import os
import logging
import warnings
import yaml

from bliva.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from bliva.common.registry import registry
from bliva.datasets.datasets.aok_vqa_datasets import AOKVQADataset, AOKVQAEvalDataset, VQGAOKVQADataset
from bliva.datasets.datasets.coco_vqa_datasets import COCOVQADataset, COCOVQAEvalDataset, VQGCOCOVQADataset
from bliva.datasets.datasets.ocr_vqa_dataset import OCRVQADataset, STVQADataset, DocVQADataset
from bliva.datasets.datasets.llava_dataset import LLAVADataset
# from bliva.datasets.datasets.event_ocr import MJ_ST_Dataset  # 引入你的自定义数据集类
from bliva.datasets.datasets.event_ocr import EVENTOCRDataset

@registry.register_builder("eventocr")
class EventOCRBuilder(BaseDatasetBuilder):
    train_dataset_cls = EVENTOCRDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/eventocr/defaults.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        self.build_processors()

        build_info = self.config.build_info
        storage_path = build_info.storage
        vis_root = build_info.vis_root

        datasets = dict()

        if not os.path.exists(storage_path):
            warnings.warn("storage path {} does not exist.".format(storage_path))

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            ann_paths=[os.path.join(storage_path, 'train.json')], 
            vis_root=vis_root
        )

        return datasets

