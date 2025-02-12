import argparse
import numpy as np
from PIL import Image
from bliva.models import load_model_and_preprocess
import logging
import torch
from torch.utils.data import DataLoader
from bliva.datasets.datasets.base_dataset import BaseDataset
import os
import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm
import torch
from bliva.datasets.datasets.base_dataset import BaseDataset
import os
import json
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import PIL

# eventocr
class ResizeNormalize(object):
    def __init__(self, size, interpolation=PIL.Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        return img

class EVENTOCRDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        self.transform = ResizeNormalize(size=(224,224))
        # Load annotation from JSON
        self.annotation = []
        for ann_path in ann_paths:
            with open(ann_path, 'r') as f:
                self.annotation.extend(json.load(f))

    def __getitem__(self, index):
        ann = self.annotation[index]

        # Get image path and open image
        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.transform(image).to(dtype=torch.float16)
        # image = self.vis_processor(image)
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
            "image_path": image_path,
            "text_input": text_input,
            "text_output": text_output,
        }

    def collater(self, samples):
        image_list, question_list, answer_list, image_path_list = [], [], [], []

        for sample in samples:
            image_list.append(sample["image"])
            image_path_list.append(sample['image_path'])
            question_list.append(sample["text_input"])
            answer_list.append(sample["text_output"])

        return {
            "image": torch.stack(image_list, dim=0),
            "image_path": image_path_list,
            "text_input": question_list,
            "text_output": answer_list,
        }

def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

def parse_args():
    """
    Parse arguments from command line.
    """
    parser = argparse.ArgumentParser(description="Arguments for Evaluation")
    parser.add_argument("--model_name", type=str, default="bliva_vicuna")
    parser.add_argument("--device", type=str, default="cuda:0", help="Specify which gpu device to use.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for evaluation")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--ann_paths", type=str, required=True, help="Path to the annotation files")
    args = parser.parse_args()
    return args

def reload_best_model(model):
    """
    Load the best checkpoint for evaluation.
    """
    checkpoint_path = '' #input your checkpoint_path
    logging.info("Loading checkpoint from {}.".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    try:
        model.load_state_dict(checkpoint["model"])
    except RuntimeError as e:
        logging.warning(
            """
            Key mismatch when loading checkpoint. This is expected if only part of the model is saved.
            Trying to load the model with strict=False.
            """
        )
        model.load_state_dict(checkpoint["model"], strict=False)
    return model
def is_chinese_char(char):
    """
    判断字符是否为中文字符
    """
    return '\u4e00' <= char <= '\u9fff'


def split_text(text):
    """
    将句子按中英文进行分割：中文按字符，英文按单词（不区分大小写）
    """
    tokens = []
    buffer = []
    
    # 遍历每个字符
    for char in text:
        if is_chinese_char(char):
            if buffer:
                tokens.extend(''.join(buffer).lower().split())  
                buffer = []
            tokens.append(char)  
        else:
            buffer.append(char)  
    
    if buffer:
        tokens.extend(''.join(buffer).lower().split()) 
    
    return tokens


def eval_batch(batch, model):
    """
    Evaluate a batch of images and questions
    """
    images = batch["image"].to(model.device)
    image_path = batch["image_path"]
    questions = batch["text_input"]
    targets = batch["text_output"]

    outputs = model.generate({"image": images, "prompt": questions, "image_path": image_path})

   
    references = [[split_text(target)] for target in targets]  
    hypotheses = [split_text(output) for output in outputs]  


    references_set = [set(split_text(target)) for target in targets]
    hypotheses_set = [set(split_text(output)) for output in outputs]

    # Initialize BLEU metric
    smooth_func = SmoothingFunction().method4

    # Calculate BLEU scores for each output
    bleu1_scores = []
    bleu2_scores = []
    bleu3_scores = []
    bleu4_scores = []

    for ref, hyp in zip(references, hypotheses):
        # Note that `ref` is a list of references
        bleu1_scores.append(sentence_bleu(ref, hyp, weights=(1, 0, 0, 0), smoothing_function=smooth_func))
        bleu2_scores.append(sentence_bleu(ref, hyp, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth_func))
        bleu3_scores.append(sentence_bleu(ref, hyp, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth_func))
        bleu4_scores.append(sentence_bleu(ref, hyp, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth_func))

    # Calculate average BLEU scores
    avg_bleu1 = sum(bleu1_scores) / len(bleu1_scores) if bleu1_scores else 0
    avg_bleu2 = sum(bleu2_scores) / len(bleu2_scores) if bleu2_scores else 0
    avg_bleu3 = sum(bleu3_scores) / len(bleu3_scores) if bleu3_scores else 0
    avg_bleu4 = sum(bleu4_scores) / len(bleu4_scores) if bleu4_scores else 0

    # Calculate accuracy
    correct = sum(1 for ref, hyp in zip(references_set, hypotheses_set) if ref == hyp)
    total = len(targets)

    return {
        "accuracy": 100 * correct / total,
        "bleu1": avg_bleu1,
        "bleu2": avg_bleu2,
        "bleu3": avg_bleu3,
        "bleu4": avg_bleu4
    }


def main(args):
    np.random.seed(0)
    disable_torch_init()

    # Load model
    if args.model_name == "bliva_vicuna":
        model, vis_processors, text_processors = load_model_and_preprocess(name=args.model_name, model_type="vicuna7b", is_eval=True, device=args.device)
        model = reload_best_model(model)
    elif args.model_name == "bliva_flant5":
        model, vis_processors, _ = load_model_and_preprocess(name=args.model_name, model_type="flant5xxl", is_eval=True, device=args.device)
    
    model.to(args.device)
    
    model.eval()

    # Load dataset
    dataset = EVENTOCRDataset(vis_processor=vis_processors["eval"], text_processor=text_processors["eval"], vis_root=args.dataset_path, sam_root=None, ann_paths=[args.ann_paths])
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize metrics
    total_accuracy = 0.0
    total_bleu1 = 0.0
    total_bleu2 = 0.0
    total_bleu3 = 0.0
    total_bleu4 = 0.0
    total_batches = 0

    # Open the output file
    output_file_path = "/wangx/BLIVA/bliva/output/visual.txt"

    with open(output_file_path, "w") as f:
        for batch in tqdm(dataloader, desc="Evaluating"):
            metrics = eval_batch(batch, model)

            total_accuracy += metrics["accuracy"]
            total_bleu1 += metrics["bleu1"]
            total_bleu2 += metrics["bleu2"]
            total_bleu3 += metrics["bleu3"]
            total_bleu4 += metrics["bleu4"]
            total_batches += 1

            # Write image path, label, and prediction to file
            for image_path, label, prediction in zip(batch["image_path"], batch["text_output"], model.generate({"image": batch["image"].to(model.device), "prompt": batch["text_input"], "image_path": batch["image_path"]})):
                f.write(f"{image_path}\t{[label]}\t{[prediction]}\n")

    # Calculate average metrics
    avg_accuracy = total_accuracy / total_batches
    avg_bleu1 = total_bleu1 / total_batches
    avg_bleu2 = total_bleu2 / total_batches
    avg_bleu3 = total_bleu3 / total_batches
    avg_bleu4 = total_bleu4 / total_batches

    print(f"saved result to {output_file_path}")

    print(f"Accuracy: {avg_accuracy:.2f}%")
    print(f"BLEU-1: {avg_bleu1:.4f}")
    print(f"BLEU-2: {avg_bleu2:.4f}")
    print(f"BLEU-3: {avg_bleu3:.4f}")
    print(f"BLEU-4: {avg_bleu4:.4f}")

if __name__ == "__main__":
    args = parse_args()
    main(args)
