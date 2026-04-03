import pycolmap
import click
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional
import re



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
    is_google_earth = re.match(pattern, filename) is not None
    # Use re.match to see if the filename matches the pattern
    return is_google_earth


def create_geo_registration_row(image: pycolmap.Image, is_super_glue: bool = False) -> str:
    location = image.projection_center()
    if is_super_glue:
        new_name = reformat_super_glue_name(image.name)
    else:
        new_name = image.name
    return f"{new_name} {location[0]} {location[1]} {location[2]}\n"


def reformat_super_glue_name(name: str) -> str:
    before_pic, after_pic = name.split("pictures")
    before_pic = before_pic.replace("_", "/")
    return before_pic + "pictures/" + after_pic[1:]

MINIMUM_PAIRS = 5
def get_pairs_superglue(colmap_reconstruction, aligner_reconstruction):
    pairs = []
    for colmap_image in colmap_reconstruction.images.values():
        for aligner_image in aligner_reconstruction.images.values():
            formatted_colmap_name = colmap_image.name.replace("/", "_") 
            if formatted_colmap_name == aligner_image.name:
                pairs.append((colmap_image, aligner_image))
    
    
    sorted_pairs_by_focal_diff = sorted(pairs, key=lambda x: abs(colmap_reconstruction.cameras[x[0].camera_id].params[0] - aligner_reconstruction.cameras[x[1].camera_id].params[0]) / colmap_reconstruction.cameras[x[0].camera_id].params[0])
    num_pairs = max(int(len(sorted_pairs_by_focal_diff) / 5), MINIMUM_PAIRS)
    if len(sorted_pairs_by_focal_diff) > num_pairs:
        sorted_pairs_by_focal_diff = sorted_pairs_by_focal_diff[:num_pairs]
    print(f"using {len(sorted_pairs_by_focal_diff)} pairs for registration")
    return sorted_pairs_by_focal_diff 



def register_diffrent_intrinsics(colmap_to_align, aligner_colmap):
    colmap_to_align = pycolmap.Reconstruction(colmap_to_align)
    aligner_colmap = pycolmap.Reconstruction(aligner_colmap)
    pairs = get_pairs_superglue(colmap_to_align, aligner_colmap)
    images_to_align = [pair[1] for pair in pairs]
    return images_to_align



@click.command()
@click.argument("source_colmap")
@click.argument("out_image_location")
@click.option("--num_images", default=None, type=int)
@click.option("--keep_google_earth", is_flag=True, default=False)
@click.option("--is_super_glue", is_flag=True, default=False)
@click.option("--colmap_to_align", default=None)
def register(source_colmap: str, out_image_location: str, num_images: Optional[int], keep_google_earth: bool, is_super_glue: bool, colmap_to_align: Optional[str]):
    if is_super_glue and colmap_to_align:
        images_to_add = register_diffrent_intrinsics(colmap_to_align, source_colmap)
    else:
        images_to_add = []
        source_reconstruction = pycolmap.Reconstruction(source_colmap)
        for _, image in source_reconstruction.images.items():
            if not keep_google_earth and is_google_earth_image(image.name):
                continue
            images_to_add.append(image)
    write_images_to_file(images_to_add, out_image_location, is_super_glue)



def write_images_to_file(images: List[pycolmap.Image], out_image_location: str, is_super_glue: bool):
    with open(out_image_location, "w") as image_location_file:
        for image in images:
            image_location_file.write(create_geo_registration_row(image, is_super_glue))


if __name__ == "__main__":
    register()
