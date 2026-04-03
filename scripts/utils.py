import numpy as np
from pytransform3d.rotations import matrix_from_quaternion
from pytransform3d.transformations import transform_from
from pathlib import Path
import pycolmap
from typing import List, Optional
from dataclasses import dataclass
import torch
import json
import re
import os
import tqdm
import functools
from PIL import Image
import tempfile
import subprocess

COLMAP_MODELS_PATH = "./local/colmap_models"
WIKISCENES_3D_PATH = "./datasets/WikiScenes3D/"
CATHEDRAL_NAME_MAPPING = "./datasets/WikiScenes/WikiScenes/cathedrals/category.json"
CATHEDRAL_CATEGORIES = "./datasets/WikiScenes/WikiScenes/cathedrals/{}/category.json"
GOOD_META_IMAGES_PATH = "./scripts/good_meta_images.json"
GOOD_META_IMAGE_IMAGE_LIMIT = 4


def transformation_to_translation_angle(transformation):
    translation = transformation[:3, 3]
    rotation = transformation[:3, :3]
    phi = np.arctan2(rotation[1, 0], rotation[0, 0])
    theta = np.arctan2(
        -rotation[2, 0], np.sqrt(rotation[2, 1] ** 2 + rotation[2, 2] ** 2)
    )
    psi = np.arctan2(rotation[2, 1], rotation[2, 2])
    return translation, np.array([phi, theta, psi])


def read_colmap_se3_transform(path):
    """
    We read the colmap SE3 transform we created using model_aligner,
    the format is specified here:
    https://github.com/colmap/colmap/blob/66fd8e56a0d160d68af2f29e9ac6941d442d2322/src/colmap/geometry/sim3.cc#L39

    return:
    scale, transformation matrix
    """
    with open(path) as registration_transform_file:
        transform_txt = registration_transform_file.read()
        splitted = [float(element) for element in transform_txt.split(" ")]

    scale, r_w, r_x, r_y, r_z, t_x, t_y, t_z = splitted
    rotation_matrix = matrix_from_quaternion(np.array([r_w, r_x, r_y, r_z])) * scale
    translation = np.array([t_x, t_y, t_z])
    transformation = transform_from(rotation_matrix, translation, strict_check=False)
    return scale, transformation


def is_google_earth_image(filename):
    """
    Checks if the filename matches the pattern '50-WellsCathedral-3_040.jpeg'.

    Args:
    filename (str): The name of the file to check.

    Returns:
    bool: True if the filename matches the pattern, False otherwise.
    """
    # Define the regular expression pattern
    pattern = r"^\d{2}-[A-Za-z\-]+-\d{1,2}_\d{3}\.jpeg$"

    # Use re.match to see if the filename matches the pattern
    return re.match(pattern, filename) is not None


def load_registration_transform(transform_path: str):
    transform = torch.load(transform_path, map_location="cpu")
    transformation = transform["registration"].detach().cpu().squeeze().numpy()
    scale = transform["scale"].detach().cpu().numpy()
    return transformation, scale

def load_registration_transform_per_image(transform_path:str):
    transform = torch.load(transform_path, map_location="cpu")
    ret = {}
    for image in transform:
        ret[image] = (transform[image]["registration"].detach().cpu().squeeze().numpy(), transform[image]["scale"].detach().cpu().numpy())
    return ret


def load_model_transform_json(model_transform_path: str):
    with open(model_transform_path) as f:
        model_dataparser_transformation_json = json.load(f)
    model_dataparser_transformation = np.array(
        model_dataparser_transformation_json["transform"]
    )
    model_dataparser_transformation = np.vstack(
        [model_dataparser_transformation, np.array([0, 0, 0, 1])]
    )
    model_dataparser_scale = model_dataparser_transformation_json["scale"]

    assert model_dataparser_scale == 1
    return model_dataparser_transformation


class ColmapImage:
    def __init__(self, image: pycolmap.Image, reconstruction):
        self.image = image  # pycolmap image
        self.nerfstudio_c2w = None
        self.reconstruction = reconstruction
        self.loss = None

    def set_nerfstudio_c2w(self, model_transform):
        """
        To get nerf studio c2w:
        1. inverse the colmap cam_from_world
        2. convert from opencv to opengl
        3. multiply with model transform (The base model transform)
        """
        image_cam_from_world = np.vstack(
            [self.image.cam_from_world.matrix(), np.array([0, 0, 0, 1])]
        )
        image_cam_to_world = np.linalg.inv(image_cam_from_world)
        image_cam_to_world[0:3, 1:3] *= -1  # Opencv to opengl (nerfstudio does that)
        nerfstudio_image_cam_to_world = model_transform @ image_cam_to_world
        self.nerfstudio_c2w = nerfstudio_image_cam_to_world

    def rotate_nerfstudio_c2w(self, transformation, scale):
        # add row to transformation matrix
        scale_matrix = np.eye(4) * scale
        scale_matrix[3, 3] = 1
        self.nerfstudio_c2w = transformation @ scale_matrix @ self.nerfstudio_c2w

    def get_camera(self):
        return self.reconstruction.cameras[self.image.camera_id]


@dataclass
class ImagePair:
    gt_image: ColmapImage
    compare_image: ColmapImage
    gt_reconstruction: pycolmap.Reconstruction
    compare_reconstruction: pycolmap.Reconstruction

    def __repr__(self):
        reprstr = f"Image Pair (Source: {self.gt_image.image.name}, Dest: {self.compare_image.image.name})\n\n"
        return reprstr

    def print_info(self):
        gt_transform = self.gt_image.nerfstudio_c2w
        compare_transform = self.compare_image.nerfstudio_c2w
        gt_translation, _ = transformation_to_translation_angle(gt_transform)
        compare_translation, _ = transformation_to_translation_angle(compare_transform)
        error = self.get_error()
        reprstr = f"Image Pair (Source: {self.gt_image.image.name}, Dest: {self.compare_image.image.name})\n gt_translation: {gt_translation}, compare_translation: {compare_translation} error {error}\n"
        source_ratio = self.get_focal_length_ratio(
            self.gt_reconstruction, self.gt_image.image
        )
        dest_ratio = self.get_focal_length_ratio(
            self.compare_reconstruction, self.compare_image.image
        )
        reprstr += f"source_focal_length_ratio: {source_ratio}, dest_focal_length_ratio: {dest_ratio}\n"
        reprstr += f"focal ratios ratio {source_ratio / dest_ratio}\n"
        reprstr += (
            f"z translation ratios {gt_translation[-1] / compare_translation[-1]}\n"
        )
        print(reprstr)

    def get_focal_length_ratio(self, reconstruction, image):
        return (
            reconstruction.cameras[image.camera_id].params[0]
            / reconstruction.cameras[image.camera_id].params[1]
        )

    def get_error(self):
        source_transform = self.gt_image.nerfstudio_c2w
        dest_transform = self.compare_image.nerfstudio_c2w
        source_translation, source_angles = transformation_to_translation_angle(
            source_transform
        )
        dest_translation, dest_angles = transformation_to_translation_angle(
            dest_transform
        )
        translation_errors = np.linalg.norm(source_translation - dest_translation)

        # Make sure angles in 0->2pi range
        source_angles = source_angles % (2 * np.pi)
        dest_angles = dest_angles % (2 * np.pi)

        # Distance between angles 0->2pi
        diff = (source_angles - dest_angles) % (2 * np.pi)

        # Distance 0 -> -pi -> 0
        angles_errors = (diff + np.pi) % (2 * np.pi) - np.pi
        angles_errors = np.linalg.norm(angles_errors)
        # convert to degrees
        angles_errors = np.rad2deg(angles_errors)
        return translation_errors, angles_errors


def set_pairs_nerfstudio_c2w(pairs, model_transform):
    for pair in pairs:
        pair.gt_image.set_nerfstudio_c2w(model_transform=model_transform)
        pair.compare_image.set_nerfstudio_c2w(model_transform=model_transform)
    return pairs


def get_all_meta_images(cathedral_number: int):
    cathedral_path = Path(WIKISCENES_3D_PATH) / str(cathedral_number)
    return list([dir for dir in cathedral_path.iterdir() if dir.is_dir()])


def get_all_meta_images_amount(cathedral_number: int):
    return len(get_all_meta_images(cathedral_number))


@functools.lru_cache
def get_cathedral_meta_images_from_json():
    with open(GOOD_META_IMAGES_PATH) as f:
        return json.load(f)


def get_all_cathedrals():
    colmap_mapping = json.load(open(GOOD_META_IMAGES_PATH))
    return list(colmap_mapping.keys())

def get_image_names_in_model(model: Path) -> List[Path]:
    reconstruction = pycolmap.Reconstruction(str(model))
    return [Path(image.name) for image_id, image in reconstruction.images.items()]


def get_images_in_model(model: Path) -> List[pycolmap.Image]:
    reconstruction = pycolmap.Reconstruction(str(model))
    images = []
    for image_id, image in reconstruction.images.items():
        images.append(image)
    return images


def reformat_normalized_name(name: str) -> str:
    before_pic, after_pic = name.split("pictures")
    before_pic = before_pic.replace("_", "/")
    return before_pic + "pictures/" + after_pic[1:]

def find_image_in_reconstruction(
    name: str, reconstruction: pycolmap.Reconstruction, not_exact_name=False
) -> Optional[pycolmap.Image]:
    name = strip_gt_name(name)
    for _, image in reconstruction.images.items():
        image_name = strip_gt_name(image.name)
        if not_exact_name:
            if is_google_earth_image(image_name):
                continue
            image_name = reformat_normalized_name(image_name)
        if name == str(Path(image_name)):
            return image
    return None


def strip_gt_name(gt_name: str) -> str:
    start_string = "datasets/WikiScenes1200px/WikiScenes1200px/cathedrals/"
    if gt_name.startswith("dataset"):
        return gt_name.split(start_string)[-1]
    return gt_name


def get_image_pairs_from_reconstruction(
    reconstruction: pycolmap.Reconstruction):
    image_pairs = []
    for _, gt_image in reconstruction.images.items():
        image_pairs.append(
            ImagePair(
                gt_image=ColmapImage(gt_image, reconstruction),
                compare_image=ColmapImage(gt_image, reconstruction),
                gt_reconstruction=reconstruction,
                compare_reconstruction=reconstruction,
            )
        )

    return image_pairs

def get_num_of_non_google_earth_images(
    reconstruction: pycolmap.Reconstruction
) -> int:
    num_images = 0
    for image in reconstruction.images.values():
        if is_google_earth_image(os.path.basename(image.name)):
            continue
        num_images += 1
    return num_images

def get_image_pairs(
    gt_reconstruction: pycolmap.Reconstruction,
    compare_reconstruction: pycolmap.Reconstruction,
    not_exact_names = False
) -> List[ImagePair]:
    
    if gt_reconstruction is None:
        return get_image_pairs_from_reconstruction(compare_reconstruction)
    image_pairs = []
    for _, gt_image in gt_reconstruction.images.items():
        if is_google_earth_image(os.path.basename(gt_image.name)):
            continue

        other_image = find_image_in_reconstruction(
            gt_image.name, compare_reconstruction, not_exact_names
        )
        if other_image is None:
            continue
        image_pairs.append(
            ImagePair(
                gt_image=ColmapImage(gt_image, gt_reconstruction),
                compare_image=ColmapImage(other_image, compare_reconstruction),
                gt_reconstruction=gt_reconstruction,
                compare_reconstruction=compare_reconstruction,
            )
        )
    return image_pairs


def get_pairs_from_models(
    gt_reconstruction,
    compare_reconstruction,
    model_transform,
    registration_transform=None,
) -> List[ImagePair]:
    pairs = get_image_pairs(
        gt_reconstruction=gt_reconstruction,
        compare_reconstruction=compare_reconstruction,
    )
    for pair in pairs:
        pair.compare_image.set_nerfstudio_c2w(model_transform=model_transform)
        pair.gt_image.set_nerfstudio_c2w(model_transform=model_transform)
        if registration_transform is not None:
            transformation, scale = registration_transform
            pair.compare_image.rotate_nerfstudio_c2w(transformation, scale=scale)
    return pairs


def get_exterior_meta_images(cathedral_number):
    good_meta_images = []
    categories = CATHEDRAL_CATEGORIES.format(cathedral_number)
    with open(categories) as f:
        categories = json.load(f)

    for name in categories["pairs"].keys():
        if "ext" in name.lower():
            key = name
            break
    else:
        raise ValueError("No exterior meta image found")
    exterior_category = categories["pairs"][key]
    model_dir = Path(WIKISCENES_3D_PATH) / str(cathedral_number)
    for dir in model_dir.iterdir():
        print("testing if dir is good", dir)
        if not dir.is_dir():
            continue
        reconstruction = pycolmap.Reconstruction(str(dir))
        for image_id, image in reconstruction.images.items():
            break
        if image.name.startswith("{}/{}".format(cathedral_number, exterior_category)):
            good_meta_images.append(int(dir.name))
    good_meta_images = sorted(good_meta_images)
    print(good_meta_images)
    return good_meta_images


def get_cathedrals_name_mapping():
    with open(CATHEDRAL_NAME_MAPPING) as f:
        mapping = json.load(f)
    # reverse the mapping
    reverse_mapping = {int(v): k for k, v in mapping["pairs"].items()}
    return reverse_mapping


def create_combined_image(
    image1_path, image2_path, slope, intercept_ratio, output_path
):
    """
    Combines IMAGE1 and IMAGE2 along a custom diagonal line and saves the result to OUTPUT_PATH.
    """
    # Load images and resize the second to match the first
    img1 = Image.open(image1_path)
    img2 = Image.open(image2_path)
    width, height = img1.size
    img2 = img2.resize((width, height))

    # Convert images to numpy arrays
    arr1 = np.array(img1)
    arr2 = np.array(img2)

    # Calculate the intercept in pixels
    intercept = int(intercept_ratio * width)

    # Create a mask based on the custom diagonal line
    mask = np.zeros((height, width), dtype=bool)
    for y in range(height):
        x_boundary = int(slope * y + intercept)
        mask[y, :x_boundary] = True

    # Apply the mask to blend images
    combined_arr = np.where(mask[..., None], arr1, arr2)

    # Convert array back to an image and save
    combined_img = Image.fromarray(combined_arr)
    combined_img.save(output_path)
    return combined_img


def mapper(
    database_path,
    image_list,
    output_path,
    image_path,
    refine_intrinsics=False,
    input_path: Optional[str] = None,
    ignore_watermark: bool = False,
) -> Path:
    with tempfile.NamedTemporaryFile("w") as f:
        image_list_path = f.name
        f.write("\n".join(image_list))
        f.flush()
        cmdline = [
            "colmap",
            "mapper",
            "--database_path",
            database_path,
            "--image_list_path",
            image_list_path,
            "--image_path",
            image_path,
            "--output_path",
            output_path,
        ]
        if input_path is not None:
            cmdline += ["--input_path", input_path]
            # If given an input path to change move the photos
            cmdline += ["--Mapper.fix_existing_images", "1"]

        if not refine_intrinsics:
            cmdline += ["--Mapper.ba_refine_focal_length", "0"]
            cmdline += ["--Mapper.ba_refine_extra_params", "0"]
        
        if ignore_watermark:
            cmdline += ["--Mapper.ignore_watermarks", "1"]

        print(f"running: {cmdline}")

        subprocess.run(cmdline)
    if input_path is not None:
        return Path(output_path)
    return Path(output_path) / "0"

CATHEDRAL_RENAME_MAPPING = "./local/nerfstudio_data/cathedrals/{}/registered_meta_image/{}/image_rename_map.json"
def get_name_mapping(cathedral_number, colmap_number):
    with open(CATHEDRAL_RENAME_MAPPING.format(cathedral_number, colmap_number)) as f:
        return  json.load(f)
