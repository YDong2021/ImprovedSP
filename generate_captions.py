"""
Offline caption generation using LLaVA.
For each image in the dataset, generate a short description conditioned on its class label.
Output: data/captions/{dataset}_{split}_captions.json  {global_index: caption}
"""
import os
import json
import argparse
import torch
from PIL import Image
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration


def build_prompt(class_name: str) -> str:
    return (
        f"Describe this image of a {class_name} in one short sentence. "
        f"Focus on the visual appearance including shape, color, texture, and distinguishing features."
    )


def main(args):
    os.makedirs('data/captions', exist_ok=True)

    # dataset path (same mapping as data/dataset.py)
    split_map = {
        'train': 'base',
        'val': 'val',
        'test': 'novel',
    }
    dataset_dir = f'./dataset/{args.dataset_folder}/{split_map[args.split]}'
    dataset = ImageFolder(dataset_dir)
    classes = dataset.classes  # folder names
    print(f'Dataset: {dataset_dir}, {len(dataset)} images, {len(classes)} classes')

    # load LLaVA
    model_id = args.model_id
    processor = AutoProcessor.from_pretrained(model_id)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map='auto'
    )
    model.eval()

    captions = {}
    for idx in tqdm(range(len(dataset)), desc='Generating captions'):
        img_path, label = dataset.samples[idx]
        class_name = classes[label].replace('_', ' ')
        image = Image.open(img_path).convert('RGB')

        prompt_text = build_prompt(class_name)
        # LLaVA conversation format
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        text_input = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(text=text_input, images=image, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                do_sample=False,
            )
        # decode only generated part
        generated = processor.decode(output_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        captions[str(idx)] = generated.strip()

    out_path = f'data/captions/{args.dataset}_{args.split}_captions.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(captions, f, ensure_ascii=False, indent=2)
    print(f'Saved {len(captions)} captions to {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name for output file naming, e.g. miniImageNet')
    parser.add_argument('--dataset_folder', type=str, default='',
                        help='Folder name under ./dataset/, e.g. miniImagenet. Defaults to --dataset')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'])
    parser.add_argument('--model_id', type=str, default='llava-hf/llava-1.5-7b-hf')
    parser.add_argument('--max_tokens', type=int, default=77,
                        help='Max new tokens for generation (no artificial short limit)')
    args = parser.parse_args()
    if not args.dataset_folder:
        args.dataset_folder = args.dataset
    main(args)
