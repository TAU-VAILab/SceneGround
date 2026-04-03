import click
import pycolmap
import os

# add utils to python path
import sys

sys.path.append("scripts")
import tqdm
import subprocess
import tempfile
import time

from utils import get_image_pairs, ImagePair, get_images_in_model, mapper
from typing import List, Optional
from pathlib import Path

GT_PATH_FORMAT = "./datasets/wikiearth_dataset/{}/sparse/"
ORIGINAL_META_IMAGE_FORMAT = "./datasets/WikiScenes3D/{}/{}/"
IMAGE_PATH = "./datasets/WikiScenes1200px/WikiScenes1200px/cathedrals"
IMAGE_PATH_IMC_PT = "./datasets/WikiScenes1200px/WikiScenes1200px/cathedrals/{}"
META_IMAGE_COLMAPS_PATH = "./local/colmap_models/{}//meta_image/{}"
DATABASE_PATH = os.path.join(META_IMAGE_COLMAPS_PATH, "database.db")
MODEL_PATH = os.path.join(META_IMAGE_COLMAPS_PATH, "sparse")


# On google earth consts
WILD_REGISTRATION_SPARSE = "./local/colmap_models/{}/wild_registration/{}/sparse"
WILD_REGISTRATION_DATABASE = "./local/colmap_models/{}/wild_registration/{}/database.db"
GOOGLE_EARTH_SPARSE_RECONSTRUCTION = "./local/colmap_models/{}/base/sparse/0/"


def get_image_path(cathedral_number):
    # If cathedral number >=100 it is the IMC_PT dataset else it is the WikiScenes1200px
    if cathedral_number >= 100:
        return IMAGE_PATH_IMC_PT.format(cathedral_number)
    return IMAGE_PATH
    
@click.command()
@click.argument("cathedral_number", type=int)
@click.argument("colmap_number", type=int)
@click.option("--on_google_earth", is_flag=True, default=False)
@click.option("--no_ground_truth", is_flag=True, default=False)
def create_meta_image(
    cathedral_number, colmap_number, on_google_earth, no_ground_truth
):
    print(f"Creating meta image for cathedral {cathedral_number}")
    gt_path = GT_PATH_FORMAT.format(cathedral_number)

    if cathedral_number >= 200 and cathedral_number < 300:
        meta_image_cathedral_number = cathedral_number - 200
    else:
        meta_image_cathedral_number = cathedral_number

    original_meta_image_path = ORIGINAL_META_IMAGE_FORMAT.format(
        meta_image_cathedral_number, colmap_number
    )
    google_earth_sparse_input = GOOGLE_EARTH_SPARSE_RECONSTRUCTION.format(
        cathedral_number, colmap_number
    )
    if on_google_earth:
        database_path = WILD_REGISTRATION_DATABASE.format(
            cathedral_number, colmap_number
        )
    else:
        database_path = DATABASE_PATH.format(cathedral_number, colmap_number)
    if on_google_earth:
        output_path = WILD_REGISTRATION_SPARSE.format(cathedral_number, colmap_number)
    else:
        output_path = MODEL_PATH.format(cathedral_number, colmap_number)
    meta_image_reconstruction = pycolmap.Reconstruction(original_meta_image_path)
    meta_image_images = get_images_in_model(original_meta_image_path)
    # FOR IMC_PT - because there are many images, but do not break backward compatibility
    if cathedral_number >= 100:
        meta_image_images = meta_image_images[:300]    
    meta_image_images_list = [image.name for image in meta_image_images]
    only_meta_image_images = meta_image_images
    gt_metaimage_pairs = []

    if not no_ground_truth:
        gt_reconstruction = pycolmap.Reconstruction(gt_path)
        gt_metaimage_pairs = get_image_pairs(
            gt_reconstruction, meta_image_reconstruction
        )

        if len(gt_metaimage_pairs) == 0:
            raise ValueError("No gt metaimage pairs")

        meta_image_in_gt_images_names = [
            pair.compare_image.image.name for pair in gt_metaimage_pairs
        ]
        only_meta_image_images = [
            image
            for image in meta_image_images
            if image.name not in meta_image_in_gt_images_names
        ]

    os.makedirs(os.path.dirname(database_path), exist_ok=True)
    os.makedirs(output_path, exist_ok=True)

    print("Extracting features")

    image_path = get_image_path(cathedral_number)

    extract_features(
        gt_metaimage_pairs, only_meta_image_images, database_path, image_path
    )
    print("Running exhaustive matcher")
    exhaustive_matcher(database_path)

    # 1st doing the mapping that refine intrinsics so it wont refine the intrinsics of the images that are in the gt
    mapper_input_path = None
    if on_google_earth:
        mapper_input_path = google_earth_sparse_input
    _ = mapper(
        database_path,
        meta_image_images_list,
        output_path,
        image_path=image_path,
        refine_intrinsics=False,
        input_path=mapper_input_path,
    )


def exhaustive_matcher(database_path):
    # Not using pycolmap because it does not support cuda
    cmdline = ["colmap", "exhaustive_matcher", "--database_path", database_path]
    print(f"running: {cmdline}")
    subprocess.run(cmdline)


def extract_features(
    pairs: List[ImagePair],
    images_without_intrinsics: List[pycolmap.Image],
    database_path,
    image_path,
):
    for pair in tqdm.tqdm(pairs, desc="Extracting features"):
        image_list = [pair.compare_image.image.name]

        # GT CAMERA
        gt_camera = pair.gt_image.get_camera()
        gt_params = gt_camera.params
        gt_params_str = ",".join([str(param) for param in gt_params])
        print(f"Extracting features for {pair.compare_image.image.name}")

        with tempfile.NamedTemporaryFile("w") as f:
            image_list_path = f.name
            f.write("\n".join(image_list))
            f.flush()
            # Using subprocess run because
            # 1. it uses gpu
            # 2. I think I can not combine pycolmap and colmap

            command = [
                "colmap",
                "feature_extractor",
                "--database_path",
                database_path,
                "--image_path",
                image_path,
                "--image_list_path",
                image_list_path,
                "--ImageReader.single_camera",
                "1",
                "--ImageReader.camera_params",
                gt_params_str,
                "--ImageReader.camera_model",
                "SIMPLE_RADIAL",
            ]
            run_command_with_retry(command)

    image_list = [image.name for image in images_without_intrinsics]
    with tempfile.NamedTemporaryFile("w") as f:
        image_list_path = f.name
        f.write("\n".join(image_list))
        f.flush()
        cmd = [
            "colmap",
            "feature_extractor",
            "--database_path",
            database_path,
            "--image_path",
            image_path,
            "--image_list_path",
            image_list_path,
        ]
        print("running: ", cmd)
        subprocess.run(cmd)


def run_command_with_retry(command, retries=3, timeout=10):
    for attempt in range(retries):
        try:
            result = subprocess.run(
                command,
                timeout=timeout,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return result  # Return the result if successful
        except subprocess.TimeoutExpired:
            print(f"Attempt {attempt + 1} timed out. Retrying...")
            # Optionally, add a delay before retrying
            time.sleep(1)
    raise RuntimeError(
        f"Command '{' '.join(command)}' failed after {retries} attempts."
    )


create_meta_image()
