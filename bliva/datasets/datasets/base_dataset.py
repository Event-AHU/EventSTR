"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json
from typing import Iterable

from torch.utils.data import Dataset, ConcatDataset
from torch.utils.data.dataloader import default_collate

class LmdbDataset(Dataset):
    def __init__(
        self, vis_processor=None, text_processor=None, vis_root=None, ann_paths=[], lmdb_paths=[]
    ):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_paths (list): List of paths to annotation files
        lmdb_paths (list): List of paths to LMDB databases
        """
        self.vis_root = vis_root
        self.annotation = []
        
        # Load annotation from JSON files
        for ann_path in ann_paths:
            with open(ann_path, "r") as f:
                self.annotation.extend(json.load(f))

        # Initialize LMDB environments if provided
        self.lmdb_envs = [lmdb.open(path, readonly=True, lock=False) for path in lmdb_paths] if lmdb_paths else []

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self._add_instance_ids()

    def __len__(self):
        return len(self.annotation)

    def collater(self, samples):
        return default_collate(samples)

    def set_processors(self, vis_processor, text_processor):
        self.vis_processor = vis_processor
        self.text_processor = text_processor

    def _add_instance_ids(self, key="instance_id"):
        for idx, ann in enumerate(self.annotation):
            ann[key] = str(idx)

    def get_lmdb_data(self, index):
        """Optional: Method to retrieve data from LMDB using index."""
        if not self.lmdb_envs:
            raise ValueError("LMDB environments are not initialized.")
        
        # Example: Retrieving data from LMDB environments
        for env in self.lmdb_envs:
            with env.begin(write=False) as txn:
                data = txn.get(str(index).encode('utf-8'))
                if data:
                    return data  # Return the first found data in LMDBs
        return None

class BaseDataset(Dataset):
    def __init__(
        self, vis_processor=None, text_processor=None, vis_root=None, ann_paths=[]
    ):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.vis_root = vis_root

        self.annotation = []
        for ann_path in ann_paths:
            self.annotation.extend(json.load(open(ann_path, "r")))

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self._add_instance_ids()

    def __len__(self):
        return len(self.annotation)

    def collater(self, samples):
        return default_collate(samples)

    def set_processors(self, vis_processor, text_processor):
        self.vis_processor = vis_processor
        self.text_processor = text_processor

    def _add_instance_ids(self, key="instance_id"):
        for idx, ann in enumerate(self.annotation):
            ann[key] = str(idx)


class ConcatDataset(ConcatDataset):
    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super().__init__(datasets)

    def collater(self, samples):
        # TODO For now only supports datasets with same underlying collater implementations

        all_keys = set()
        for s in samples:
            all_keys.update(s)

        shared_keys = all_keys
        for s in samples:
            shared_keys = shared_keys & set(s.keys())

        samples_shared_keys = []
        for s in samples:
            samples_shared_keys.append({k: s[k] for k in s.keys() if k in shared_keys})

        return self.datasets[0].collater(samples_shared_keys)


class BasePromptDataset(Dataset):
    def __init__(
        self, vis_processor=None, text_processor=None, vis_root=None, ann_paths=[]
    ):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.vis_root = vis_root

        self.annotation = json.load(open(ann_paths[0], "r")) # a list of dicts or just a dict 

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self._add_instance_ids()

    def __len__(self):
        return len(self.annotation['data'])

    def collater(self, samples):
        return default_collate(samples)

    def set_processors(self, vis_processor, text_processor):
        self.vis_processor = vis_processor
        self.text_processor = text_processor

    def _add_instance_ids(self, key="instance_id"):
        for idx, ann in enumerate(self.annotation['data']):
            ann[key] = str(idx)


