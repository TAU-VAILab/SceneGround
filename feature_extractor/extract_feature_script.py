import click
import json
import tqdm
import os
from typing import Literal, List
from pathlib import Path

@click.group()
def extract_features_group():
    pass

@extract_features_group.command()
@click.argument("image_dir")
@click.argument("features_type")
@click.argument("output_dir")
@click.option("--skip_existing", is_flag=True, default=False)
@click.option("-v",  "--visualize", is_flag=True, default=False)
def directory(image_dir: click.Path, features_type:Literal["lseg", "pyramid", "dift", "segformer", "lseg_mask", "dino", "clip_sam"], output_dir: click.Path, visualize:bool, skip_existing:bool):
    images = list(Path(image_dir).iterdir())
    extract_features(images, features_type, output_dir, visualize, skip_existing)

@extract_features_group.command()
@click.argument("cathedral_dir")
@click.argument("features_type")
@click.option("--skip_existing", is_flag=True, default=False)
@click.option("-v",  "--visualize", is_flag=True, default=False)
def cathedral(cathedral_dir: click.Path, features_type:Literal["lseg", "pyramid", "dift", "segformer", "lseg_mask", "dino", "clip_sam"], visualize:bool, skip_existing:bool):
    transform_json = Path(cathedral_dir) / "transforms.json"
    with open(transform_json) as f:
        transform = json.load(f)
    image_dir = Path(cathedral_dir) / "images"
    images = list(image_dir.iterdir())
    images_in_transform = [Path(image["file_path"]).name for image in transform["frames"]]
    images = [image for image in images if image.name in images_in_transform]
    output_dir = Path(cathedral_dir) / "images_{}_features".format(features_type)
    extract_features(images, features_type, output_dir, visualize, skip_existing)

def extract_features(images: List[Path], features_type:Literal["lseg", "pyramid", "dift", "segformer", "lseg_mask", "dino", "clip_sam"], output_dir: click.Path, visualize:bool, skip_existing:bool):
    print("Created feature extractor")
    lasy_import = False
    for image_path in tqdm.tqdm(images):
        # Load the input image
        image_name = image_path.name
        if not image_name.startswith("frame"):
            continue
        output_file = os.path.join(output_dir, f"{image_name}.pt")
        if skip_existing and os.path.exists(output_file):
            print(f"Skipping {image_name}")
            continue
        if not lasy_import:
            from feature_extractors import FeatureExtractor
            import torch
            import cv2
            feature_extractor = FeatureExtractor.get_feature_extractor(features_type)
            lasy_import = True
        try:
            features, image = feature_extractor.extract_from_path(image_path)
        except (cv2.error):
            print(f"Error in image {image_path}")
            continue
        os.makedirs(output_dir, exist_ok=True)
        torch.save(features, os.path.join(output_dir, f"{image_name}.pt"))
        visualization_path = os.path.join(output_dir, "visualizations")
        os.makedirs(visualization_path, exist_ok=True)
        if visualize:
            if features_type in ["segformer", "lseg_mask"]:
                feature_extractor.visualize_mask(image, features, os.path.join(visualization_path, f"{image_name}.png"))
            else:
                feature_extractor.visualize_features(features, os.path.join(visualization_path, f"{image_name}.png"))

if __name__ == "__main__":
    print("Finished imports - starting main")
    extract_features_group()
