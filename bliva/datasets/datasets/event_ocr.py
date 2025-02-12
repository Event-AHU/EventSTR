import torch
from bliva.datasets.datasets.base_dataset import BaseDataset
import os
import json
import PIL
from PIL import Image
import numpy as np
# from torchvision.transforms.functional import resize, to_tensor
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import resize, to_tensor
class EVENTOCRDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        
        # Load annotation from JSON
        self.transform = ResizeNormalize(size=(224,224)) #128,32
        self.annotation = []
        for ann_path in ann_paths:
            with open(ann_path, 'r') as f:
                self.annotation.extend(json.load(f))

    def __getitem__(self, index):
        ann = self.annotation[index]

        # Get image path and open image
        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.transform(image).to(dtype=torch.float16)  #eventocr

        # image = self.vis_processor(image)  #WordArt and IC15
        conversations = ann["conversations"]
        question = ""
        answer = ""

        for conv in conversations:
            if conv["from"] == "human":
                question = conv["value"]
            elif conv["from"] == "gpt":
                answer = conv["value"]

        # Process text input (if needed) and set answer
        # question = self.text_processor(question)
        text_input = question
        text_output = answer

        return {
            "image": image,
            "text_input": text_input,
            "text_output": text_output,
        }

    def collater(self, samples):
        image_list, question_list, answer_list = [], [], []

        for sample in samples:
            image_list.append(sample["image"])
            question_list.append(sample["text_input"])
            answer_list.append(sample["text_output"])

        return {
            "image": torch.stack(image_list, dim=0),
            "text_input": question_list,
            "text_output": answer_list,
        }

class ResizeNormalize(object):
    def __init__(self, size, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)        
        self.normalize = transforms.Normalize(mean, std)

    def __call__(self, img):
        w, h = img.size
        target_w, target_h = self.size

        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        img = img.resize((new_w, new_h), self.interpolation)

        new_img = Image.new("RGB", self.size, (255, 255, 255))

        offset = ((target_w - new_w) // 2, (target_h - new_h) // 2)

        new_img.paste(img, offset)

        img = self.toTensor(new_img)
        img = self.normalize(img)
        
        return img