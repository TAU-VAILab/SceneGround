import torch
from pathlib import Path
import cv2
from lseg import LSegNet
from lerf.data.utils.pyramid_embedding_dataloader import PyramidEmbeddingDataloader
from lerf.encoders.image_encoder import BaseImageEncoder
from lerf.data.lerf_datamanager import LERFDataManagerConfig
from lerf.lerf import LERFModelConfig
from lerf.data.lerf_datamanager import LERFDataManager
from lerf.lerf_pipeline import LERFPipelineConfig
from lerf.encoders.clip_encoder import CLIPNetworkConfig
from lerf.encoders.openclip_encoder import OpenCLIPNetworkConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from pathlib import Path
import cv2
import torch
import numpy as np
from sklearn.decomposition import PCA
from PIL import Image
import tempfile
from transformers import pipeline
import clip
import functools
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, AutoModel
import math

import open_clip
import torchvision.transforms as T

LSEG_PATH = "./local/model_checkpoints/lseg_minimal_e200.ckpt"
class FeatureExtractor:
    def __init__(self, feature_scale=(640, 480)) -> None:
        self.feature_scale = feature_scale

    @classmethod
    def get_feature_extractor(cls, features_type: str):
        if features_type == "pyramid" or features_type == "clip":
            feature_extractor = ClipPyramidFeatureExtractor()
        elif features_type == "lseg":
            feature_extractor = LSegFeatureExtractor(checkpoint_path=LSEG_PATH)
        elif features_type == "segformer":
            feature_extractor = SegFormerMaskExtractor()
        elif features_type == "lseg_mask":
            feature_extractor = LSegMaskExtractor(positive_prompts=["building", "cathedral", "sky"], negative_prompts=["object", "other", "ground", "trees"], checkpoint_path=LSEG_PATH)
        elif features_type == "dino":
            feature_extractor = DinoFeatureExtractor()
        elif features_type == "clip_sam":
            feature_extractor = ClipSAMFeatureExtractor()
        elif features_type == "clip_mask":
            feature_extractor = CLIPMaskExtractor(positive_prompts=["object", "things", "stuff", "texture", "sky"], negative_prompts=["tower"])
        elif features_type == "dino_lseg":
            feature_extractor = DinoLSegFeatureExtractor(lseg_checkpoint_path=LSEG_PATH)
        else:
            raise ValueError(f"Unknown feature extractor {features_type}")
        return feature_extractor

    def extract(self, image: torch.tensor) -> torch.tensor:
        """
        returns [1, FeatureDim, Height, Width]
        """
        raise NotImplementedError

    def extract_from_path(self, image_path: Path) -> torch.tensor:
        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.feature_scale)
        img = torch.from_numpy(img).float() / 255.0
        img = img[..., :3]  # drop alpha channel, if present
        img = img.cuda()
        img = img.permute(2, 0, 1)  # C, H, W
        return self.extract(img), img

    def features_to_pca(self, features:torch.tensor):
        """
        features: [1, FeatureDim, Height, Width]
        """
        reshaped_feature = (
            features.clone()
            .squeeze()
            .reshape(features.shape[1], -1)
            .permute(1, 0)
            .cpu()
            .detach()
            .numpy()
        )
        data_pca = PCA(n_components=3).fit_transform(reshaped_feature)
        data_pca_rescaled = (
            (data_pca - data_pca.min()) / (data_pca.max() - data_pca.min()) * 255
        )
        data_pca_rescaled = data_pca_rescaled.astype(np.uint8)

        # Reshape data_pca_rescaled into an RGB image
        img_shape = (
            features.shape[-2],
            features.shape[-1],
            3,
        )  # Define your image dimensions
        rgb_image = data_pca_rescaled.reshape(img_shape)
        return rgb_image
    
    def visualize_features(self, features: np.array, path: str):
        pca = self.features_to_pca(features)
        self.save_image(pca, path)
        
    def visualize_mask(self, image: torch.tensor, mask: torch.tensor, path: str):
        image = np.rollaxis(image.detach().cpu().numpy(), 0, 3)
        mask = torch.nn.functional.interpolate(mask.unsqueeze(0).unsqueeze(0).to(torch.float), size=(image.shape[0], image.shape[1]))
        mask = (mask > 0).squeeze()
        mask = mask.detach().cpu().numpy()

        # color black where masked
            # Make the original image half visible red where masked
        image[mask == 0, 0] = image[mask == 0, 0] * 0.5  # Red channel
        image[mask == 0, 1] = 0  # Green channel
        image[mask == 0, 2] = 0  # Blue channel
        
        plt.imsave(path, image)

    def save_image(self, image: np.array, path: str):
        im = Image.fromarray(image.astype(np.uint8))
        im.save(path)

class ClipSAMFeatureExtractor(FeatureExtractor):
    def __init__(self, feature_scale=(640, 480)) -> None:
        super().__init__(feature_scale)
        self.model = OpenCLIPNetwork(LangSplatOpenCLIPNetworkConfig)
        sam = sam_model_registry["vit_h"](checkpoint="./local/model_checkpoints/sam_vit_h_4b8939.pth").to('cuda')
        self.mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=32,
            pred_iou_thresh=0.7,
            box_nms_thresh=0.7,
            stability_score_thresh=0.85,
            crop_n_layers=1,
            crop_n_points_downscale_factor=1,
            min_mask_region_area=100,
        )
    
    def extract(self, image: torch.tensor) -> torch.tensor:
        print("extracting features of SamAutomaticMaskGeneratorimage with shape {}".format(image.shape))
        image = image * 255.0
        level = "m"
        self.mask_generator.predictor.model.to('cuda')
        img_embed, seg_map = _embed_clip_sam_tiles(image.unsqueeze(0), sam_encoder, self.model, self.mask_generator, level)
        img_embed_level = img_embed[level]
        seg_map_level = seg_map[level]
        
        features = self.convert_seg_to_embbeding(img_embed_level, seg_map_level)
        return features
    
    def convert_seg_to_embbeding(self, img_embed, seg_map):
        feature_size = img_embed.shape[1]
        H = seg_map.shape[0]
        W = seg_map.shape[1]
        features = torch.zeros((H, W, feature_size))
        segmented_pixels = seg_map != -1
        features[segmented_pixels, :] = img_embed[seg_map[segmented_pixels], :].to(torch.float)
        features = features.permute(2, 0, 1)
        features = features.unsqueeze(0)
        return features

class DinoFeatureExtractor(FeatureExtractor):
    def __init__(
        self,
        feature_scale=(100 * 14, 100 * 14),
    ) -> None:
        super().__init__(feature_scale)
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.normalize_image = T.Compose([
        T.Normalize(mean=[0.5], std=[0.5]),
        ])


    def extract(self, image: torch.tensor) -> torch.tensor:
        with torch.no_grad():
            image = self.normalize_image(image)[:3].unsqueeze(0)
            self.model.cuda()
            features = self.model.forward_features(image)["x_norm_patchtokens"].squeeze(0)
            features = features.permute(1, 0)
            features = features.reshape(-1, int(np.sqrt(features.shape[1])), int(np.sqrt(features.shape[1])))
            features = features.unsqueeze(0)
            # Normalize is good only when concatanting with different feature types, I believe that now we have a mix in the data...
            #features = torch.nn.functional.normalize(features, dim=1)
            return features

class LSegFeatureExtractor(FeatureExtractor):
    def __init__(
        self,
        checkpoint_path,
        feature_scale=(640, 480),
        backbone: str = "clip_vitl16_384",
        num_features: int = 256,
        arch_option: int = 0,
        block_depth: int = 0,
        activation: str = "lrelu",
        crop_size: int = 480,
    ) -> None:
        super().__init__(feature_scale)
        self.net = LSegNet(
            backbone=backbone,
            features=num_features,
            crop_size=crop_size,
            arch_option=arch_option,
            block_depth=block_depth,
            activation=activation,
        )
        self.net.load_state_dict(torch.load(str(checkpoint_path)))
        self.net.eval()
        self.net.cuda()

    def extract(self, image: torch.tensor) -> torch.tensor:
        """
        returns [1, FeatureDim, Height, Width]
        """
        image = image.unsqueeze(0)
        img_feat = self.net.forward(image)
        img_feat_norm = torch.nn.functional.normalize(img_feat, dim=1)
        return img_feat_norm

class DinoLSegFeatureExtractor(FeatureExtractor):
    def __init__(
        self,
        lseg_checkpoint_path,
        feature_scale=(100 * 14, 100 * 14),
    ) -> None:
        super().__init__(feature_scale)
        self.lseg_extractor = LSegFeatureExtractor(checkpoint_path = lseg_checkpoint_path)
        self.dino_extractor = DinoFeatureExtractor(feature_scale = feature_scale)

    def extract_from_path(self, image_path: Path) -> torch.tensor:
        lseg_fetures, image = self.lseg_extractor.extract_from_path(image_path)
        dino_features, _ = self.dino_extractor.extract_from_path(image_path)
        squeezed_lseg_features = torch.nn.functional.interpolate(lseg_fetures, size=dino_features.shape[-2:])
        features = torch.concatenate([squeezed_lseg_features, dino_features], axis=1)
        return features, image

class ClipPyramidFeatureExtractor(FeatureExtractor):
    def __init__(self, feature_scale=(250, 250)) -> None:
        self.pipeline = LERFPipelineConfig(
            datamanager=LERFDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(),
            ),
            model=LERFModelConfig(),
            network=OpenCLIPNetworkConfig(),
        )

        self.image_encoder = self.pipeline.network.setup()
        super().__init__(feature_scale)

    def extract(self, image: torch.tensor) -> torch.tensor:
        """
        returns [1, FeatureDim, Height, Width]
        """
        index = 0
        image_row_number = image.shape[1]
        image_col_number = image.shape[2]

        # the embedding should receive img_points: (B, 3) # (img_ind, x, y) (img_ind, row, col)
        row_grid, col_grid = torch.meshgrid(torch.arange(image_row_number), torch.arange(image_col_number))
        index_grid = torch.zeros_like(row_grid) + index
        combined_index_grid = torch.stack([index_grid, row_grid, col_grid])
        flattened_index_grid = combined_index_grid.reshape(combined_index_grid.shape[0], -1)
        # Permuting because it expects (B, 3) instead of (3, B)
        flattened_index_grid = flattened_index_grid.permute(1, 0)
        
        # Expected shape (N, 3, H, W)
        image_list = image.unsqueeze(0)
        with tempfile.TemporaryDirectory() as cache_dir:
            clip_interpolator = PyramidEmbeddingDataloader(
                image_list=image_list,
                device="cuda",
                cfg={
                    "tile_size_range": [0.05, 0.1], # size range of the tile of the pyramid
                    "tile_size_res": 7, # Number of sizes in the range
                    "stride_scaler": 0.5, # not sure exactly what this does
                    "image_shape": list(image_list.shape[2:4]),
                    "model_name": self.image_encoder.name,
                },
                cache_path=Path(cache_dir),
                model=self.image_encoder,
            )
            scales = np.linspace(0.05, 0.09, 7)
            features = None
            for scale in scales:
                clip_features, _ = clip_interpolator(flattened_index_grid, scale=scale)
                if features is None:
                    features = clip_features
                else:
                    features += clip_features
            features = features / scales.shape[0]
            clip_features = clip_features.reshape(image_row_number, image_col_number, -1)
            return clip_features.permute(2, 0 ,1).unsqueeze(0)

class SegFormerMaskExtractor(FeatureExtractor):
    def __init__(self, feature_scale=(640, 480)):
        super().__init__(feature_scale)
        self.labels = ["sky", "building"]
        self.semantic_segmentation = pipeline("image-segmentation", "nvidia/segformer-b1-finetuned-cityscapes-1024-1024")

    def extract(self, image: torch.tensor) -> torch.tensor:
        """
        output shape [H, W]
        """
        # plot picture: plt.imsave("image.png", np.rollaxis(image.detach().cpu().numpy(), 0, 3))
        pil_image = Image.fromarray(np.rollaxis((image.detach().cpu().numpy() * 255).astype(np.uint8), 0, 3))
        masks = self.semantic_segmentation(pil_image)
        # Take only the masks relevant for out labels
        masks = [np.array(mask["mask"]) for mask in masks if mask["label"] in self.labels]
        masks = [mask > 0 for mask in masks]
        combined_mask = functools.reduce(np.logical_or, masks)
        # plot mask plt.imsave("mask.png", combined_mask)
        return torch.from_numpy(combined_mask).cuda()

class LSegMaskExtractor(LSegFeatureExtractor):
    def __init__(self, checkpoint_path, positive_prompts, negative_prompts):
        super().__init__(checkpoint_path)
        self.positive_prompts = positive_prompts
        self.negative_prompts = negative_prompts
        self.prompts = self.positive_prompts + self.negative_prompts
        self.tokenized_prompts = clip.tokenize(self.prompts)
        self.tokenized_prompts = self.tokenized_prompts.cuda()
        clip_text_encoder = self.net.clip_pretrained.encode_text
        self.prompt_features = clip_text_encoder(self.tokenized_prompts)


    def extract(self, image: torch.tensor) -> torch.tensor:
        image_features = super().extract(image)
        cosine_similarity = torch.nn.CosineSimilarity(dim=1)
        # the prompts are positive_prompts + negative_prompts
        similarity = cosine_similarity(image_features, self.prompt_features.unsqueeze(-1).unsqueeze(-1))
        positive_similarity = similarity[:len(self.positive_prompts)]
        negative_similarity = similarity[len(self.positive_prompts):]
        max_positive_similarity = torch.max(positive_similarity, dim=0).values
        max_negative_similarity = torch.max(negative_similarity, dim=0).values
        mask = max_positive_similarity > max_negative_similarity
        return mask

class CLIPMaskExtractor(ClipPyramidFeatureExtractor):
    def __init__(self, positive_prompts, negative_prompts):
        super().__init__()
        self.positive_prompts = positive_prompts
        self.negative_prompts = negative_prompts
        self.prompts = self.positive_prompts + self.negative_prompts
        self.tokenized_prompts = open_clip.get_tokenizer(self.pipeline.network.clip_model_type)(self.prompts)
        self.tokenized_prompts = self.tokenized_prompts.cuda()
        clip_model, preprocess, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="laion2b_s34b_b88k", device="cuda")
        self.prompt_features = clip_model.encode_text(self.tokenized_prompts)


    def extract(self, image: torch.tensor) -> torch.tensor:
        image_features = super().extract(image)
        cosine_similarity = torch.nn.CosineSimilarity(dim=1)
        # the prompts are positive_prompts + negative_prompts
        similarity = cosine_similarity(image_features, self.prompt_features.unsqueeze(-1).unsqueeze(-1))
        positive_similarity = similarity[:len(self.positive_prompts)]
        negative_similarity = similarity[len(self.positive_prompts):]
        max_positive_similarity = torch.max(positive_similarity, dim=0).values
        max_negative_similarity = torch.max(negative_similarity, dim=0).values
        mask = max_positive_similarity > max_negative_similarity
        return mask

def get_loss_image(feature_a, feature_b):
    colormap = plt.cm.plasma
    
    diff_image = np.abs(feature_a.squeeze().cpu().detach().numpy() - feature_b.squeeze().cpu().detach().numpy())
    image_means = np.mean(diff_image, axis=0)
    normalized_image = (image_means - np.min(image_means)) / (np.max(image_means) - np.min(image_means))
    return (colormap(normalized_image)[..., :-1] * 255).astype(np.uint8)

def get_loss(feature_a, feature_b):
    return torch.abs((feature_a - feature_b)).mean()
