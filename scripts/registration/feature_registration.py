import click
import sys

sys.path.append("scripts")
import subprocess
from typing import Optional

MODEL_PATH_FORMAT = (
    "local/model_checkpoints/cathedrals/{}/base_google_earth/features_oriented_{}"
)

TRANSFORM_PATH_FORMAT = "./paper_results/transforms/{}/{}/percentile-50_iteration-12000_near-plane-0.7__l2_loss-False_feature_type-dino_loss_masking-100_transform.pt"

def register_nerfstudio(
    cathedral_number: int,
    colmap_number: int,
    iteration_number: int,
    feature_type: str,
    extra_transform_name: str = "",
    loss_percentile: Optional[int] = None,
    l2_loss: bool = False,
    near_plane: Optional[float] = None,
    loss_masking: Optional[int] = None,
    use_irls: bool = False,
    remove_half: bool = False,
    angle: Optional[float] = None,
    translation: Optional[float] = None,
    extra: Optional[str] = None,
    image_percent: Optional[float] = None,
    initialization: Optional[str] = None,
):
    transform_name = f"percentile-{loss_percentile}_iteration-{iteration_number}_near-plane-{near_plane}_{extra_transform_name}_l2_loss-{l2_loss}_feature_type-{feature_type}_loss_masking-{loss_masking}_angle-{angle}_translation-{translation}"
    if extra:
        transform_name += "_extra-" + extra
    if remove_half:
        transform_name += "_remove_half"
    if use_irls:
        transform_name += "_irls"
    if image_percent is not None:
        transform_name += "_image_percent-" + str(image_percent)
    
    if initialization is not None:
        transform_name += "_initilization-" + initialization
    if feature_type == "rgb":
        ns_method = "register_splatfacto"
        model_path = MODEL_PATH_FORMAT.format(
            cathedral_number, "dino"
        )  # Use the dino basemodel for rgb
    else:
        ns_method = "register_splatfacto_features"
        model_path = MODEL_PATH_FORMAT.format(cathedral_number, feature_type)
    
    ns_name = "registered_meta_image" if initialization is None else initialization 

    print("Starting transform: ", transform_name)
    cmd_line = ["ns-train", ns_method]
    if loss_percentile is not None:
        cmd_line += ["--pipeline.model.loss-percentile", str(loss_percentile)]
    if l2_loss:
        cmd_line += ["--pipeline.model.l2-loss", "True"]
    if near_plane is not None:
        cmd_line += ["--pipeline.model.near-plane", str(near_plane)]
    if loss_masking is not None:
        cmd_line += ["--pipeline.model.image_loss_masking", str(loss_masking)]
    if use_irls:
        cmd_line += ["--pipeline.model.use_irls", "True"]

    if remove_half:
        cmd_line += ["--pipeline.model.remove_half", "True"]

    if translation or angle:
        transform_path = TRANSFORM_PATH_FORMAT.format(cathedral_number, colmap_number)
        cmd_line += ["--pipeline.model.init_transform_path", transform_path]
        if angle is not None and angle != 0:
            cmd_line += ["--pipeline.model.angle", str(angle)]
        if translation is not None and translation != 0:
            cmd_line += ["--pipeline.model.translation", str(translation)]
    

    if feature_type != "rgb":
        cmd_line += ["--pipeline.model.feature_type", feature_type]

    experiment_name = (
        f"cathedral-{cathedral_number}_colmap_{colmap_number}_" + transform_name
    )
    cmd_line += [
        "--pipeline.model.num_downscales",
        "1",
        "--pipeline.model.resolution_schedule",
        "999999999",
        "--viewer.quit-on-train-completion",
        "True",
        "--transform_dir",
        transform_name,
        "--max_num_iterations",
        str(iteration_number),
        "--data",
        f"./local/nerfstudio_data/cathedrals/{cathedral_number}/{ns_name}/{colmap_number}",
        "--cathedral_number",
        str(cathedral_number),
        "--colmap_number",
        str(colmap_number),
        "--output-dir",
        "./registrations_output",
        "--experiment_name",
        experiment_name,
        "--load_dir",
        f"{model_path}/nerfstudio_models",
        "--vis",
        "viewer+tensorboard",
        "nerfstudio-data",
        "--registration",
        "True",
        "--auto_scale_poses",
        "False",
        "--orientation_method",
        "none",
        "--center_method",
        "none",
        "--model_transform_path",
        f"{model_path}/dataparser_transforms.json",
    ]
    if feature_type != "rgb":
        cmd_line += [
            "--feature-type",
            feature_type,
        ]

    if image_percent is not None:
        cmd_line += ["--frame_percent", str(image_percent)]
        cmd_line += ["--seed", extra]
    print("Running cmdline: ", cmd_line)
    try:
        subprocess.run(cmd_line, check=True)
    except KeyboardInterrupt as e:
        print("Finished transform: ", transform_name)
        raise e


@click.group()
def register():
    pass


@register.command()
@click.argument("cathedral_number", type=int)
@click.argument("colmap_number", type=int)
@click.option("--transform_name", type=str, default="")
@click.option("--iteration_number", type=int, default=6000)
@click.option("--loss_percentile", type=int, default=None)
@click.option("--near_plane", type=float, default=None)
@click.option("--l2_loss", type=bool, default=False)
@click.option("--feature_type", type=str, default="lseg")
@click.option("--loss_masking", type=int, default=None)
@click.option("--use_irls", type=bool, default=False)
@click.option("--angle", type=float, default=None)
@click.option("--translation", type=float, default=None)
@click.option("--extra", type=str, default=None)
@click.option("--image_percent", type=float, default=None)
@click.option("--initialization", type=str, default=None)
def single(
    cathedral_number: int,
    colmap_number: int,
    transform_name: str = "",
    iteration_number: int = 6000,
    loss_percentile: Optional[int] = None,
    near_plane: Optional[float] = None,
    l2_loss: bool = False,
    feature_type: str = "lseg",
    loss_masking: Optional[int] = None,
    use_irls: bool = False,
    angle: Optional[float] = None,
    translation: Optional[float] = None,
    extra: Optional[str] = None,
    image_percent: Optional[float] = None,
    initialization: Optional[str] = None,
):

    register_nerfstudio(
        cathedral_number=cathedral_number,
        colmap_number=colmap_number,
        extra_transform_name=transform_name,
        iteration_number=iteration_number,
        loss_percentile=loss_percentile,
        near_plane=near_plane,
        l2_loss=l2_loss,
        feature_type=feature_type,
        loss_masking=loss_masking,
        use_irls=use_irls,
        angle=angle,
        translation=translation,
        extra=extra,
        image_percent=image_percent,
        initialization=initialization
    )


@register.command()
@click.argument("transform_name", type=str, default="")
@click.option("--iteration_number", type=int, default=10000)
@click.option("--loss_percentile", type=int, default=None)
def all(
    iteration_number: int,
    transform_name: str = "",
    loss_percentile: Optional[int] = None,
    l2_loss: bool = False,
):
    raise NotImplementedError("This function is not implemented yet")
    for cathedral_number in CATHEDRALS_TO_REGISTER:
        colmap_number = CATHEDRALS_TO_COLMAP[cathedral_number]
        register_nerfstudio(
            cathedral_number=cathedral_number,
            colmap_number=colmap_number,
            iteration_number=iteration_number,
            extra_transform_name=transform_name,
            loss_percentile=loss_percentile,
            l2_loss=l2_loss,
        )


if __name__ == "__main__":
    register()
