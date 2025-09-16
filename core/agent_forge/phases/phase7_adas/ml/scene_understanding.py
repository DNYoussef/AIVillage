"""
ADAS Phase 7 - Scene Understanding Module
Automotive-grade semantic scene analysis with 3D reconstruction and depth estimation
"""

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging
import time
from scipy.spatial.transform import Rotation
from sklearn.cluster import DBSCAN


class WeatherCondition(Enum):
    """Weather conditions affecting scene understanding"""
    CLEAR = "clear"
    RAIN = "rain"
    SNOW = "snow"
    FOG = "fog"
    NIGHT = "night"
    DAWN_DUSK = "dawn_dusk"


class LightingCondition(Enum):
    """Lighting conditions for adaptation"""
    DAY_BRIGHT = "day_bright"
    DAY_OVERCAST = "day_overcast"
    NIGHT_DARK = "night_dark"
    NIGHT_LIT = "night_lit"
    TUNNEL = "tunnel"
    BACKLIT = "backlit"


@dataclass
class Object3D:
    """3D object detection result"""
    class_id: int
    class_name: str
    confidence: float
    bbox_2d: np.ndarray  # [x1, y1, x2, y2]
    bbox_3d: np.ndarray  # [x, y, z, w, h, l, rx, ry, rz]
    center_3d: np.ndarray  # [x, y, z]
    velocity: np.ndarray  # [vx, vy, vz]
    distance: float
    track_id: Optional[int] = None


@dataclass
class SceneSegmentation:
    """Semantic segmentation result"""
    semantic_map: np.ndarray  # H x W class IDs
    instance_map: np.ndarray  # H x W instance IDs
    confidence_map: np.ndarray  # H x W confidence scores
    class_names: List[str]
    drivable_area: np.ndarray  # Binary mask for drivable area
    lane_markings: List[np.ndarray]  # List of lane line points


@dataclass
class DepthEstimation:
    """Depth estimation result"""
    depth_map: np.ndarray  # H x W depth in meters
    confidence_map: np.ndarray  # H x W confidence scores
    point_cloud: np.ndarray  # N x 3 3D points
    normals: np.ndarray  # N x 3 surface normals
    occlusion_map: np.ndarray  # Binary occlusion mask


@dataclass
class SceneContext:
    """Complete scene understanding result"""
    objects_3d: List[Object3D]
    segmentation: SceneSegmentation
    depth: DepthEstimation
    weather: WeatherCondition
    lighting: LightingCondition
    road_geometry: Dict[str, np.ndarray]
    traffic_signs: List[Dict]
    lane_structure: Dict[str, List[np.ndarray]]
    free_space: np.ndarray  # Occupancy grid
    timestamp: float


class EfficientNet3D(nn.Module):
    """EfficientNet-based 3D object detection backbone"""

    def __init__(self,
                 num_classes: int = 10,
                 input_channels: int = 3,
                 depth_estimation: bool = True):
        super().__init__()

        self.num_classes = num_classes
        self.depth_estimation = depth_estimation

        # EfficientNet-B0 inspired backbone (simplified for automotive)
        self.stem = nn.Conv2d(input_channels, 32, 3, 2, 1)

        # Inverted residual blocks
        self.blocks = nn.ModuleList([
            self._make_mb_block(32, 16, 1, 1, 1),   # Stage 1
            self._make_mb_block(16, 24, 6, 2, 2),   # Stage 2
            self._make_mb_block(24, 40, 6, 2, 2),   # Stage 3
            self._make_mb_block(40, 80, 6, 2, 3),   # Stage 4
            self._make_mb_block(80, 112, 6, 1, 3),  # Stage 5
            self._make_mb_block(112, 192, 6, 2, 4), # Stage 6
            self._make_mb_block(192, 320, 6, 1, 1)  # Stage 7
        ])

        # Feature Pyramid Network for multi-scale features
        self.fpn = nn.ModuleDict({
            'p3': nn.Conv2d(40, 256, 1),
            'p4': nn.Conv2d(112, 256, 1),
            'p5': nn.Conv2d(320, 256, 1)
        })

        # 3D object detection heads
        self.cls_head = nn.Conv2d(256, num_classes * 9, 3, 1, 1)  # 9 anchors per location
        self.bbox_2d_head = nn.Conv2d(256, 4 * 9, 3, 1, 1)
        self.bbox_3d_head = nn.Conv2d(256, 7 * 9, 3, 1, 1)  # [x, y, z, w, h, l, ry]

        # Depth estimation head
        if depth_estimation:
            self.depth_head = nn.Sequential(
                nn.Conv2d(256, 128, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 64, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 1, 1),
                nn.Sigmoid()
            )

        # Semantic segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 19, 1)  # Cityscapes classes
        )

    def _make_mb_block(self, in_channels, out_channels, expand_ratio, stride, num_blocks):
        """Create Mobile Inverted Residual Block"""
        blocks = []
        for i in range(num_blocks):
            blocks.append(
                MBConvBlock(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    expand_ratio,
                    stride if i == 0 else 1
                )
            )
        return nn.Sequential(*blocks)

    def forward(self, x):
        # Backbone
        x = F.silu(self.stem(x))

        features = []
        for block in self.blocks:
            x = block(x)
            if x.shape[1] in [40, 112, 320]:  # Collect FPN features
                features.append(x)

        # Feature Pyramid Network
        fpn_features = []
        for i, (key, conv) in enumerate(self.fpn.items()):
            feat = conv(features[i])
            fpn_features.append(feat)

        # Combine FPN features
        combined_feat = fpn_features[0]  # Start with P3
        for feat in fpn_features[1:]:
            # Upsample to match P3 resolution
            upsampled = F.interpolate(feat, size=combined_feat.shape[2:],
                                    mode='bilinear', align_corners=False)
            combined_feat = combined_feat + upsampled

        outputs = {}

        # Object detection outputs
        outputs['cls'] = self.cls_head(combined_feat)
        outputs['bbox_2d'] = self.bbox_2d_head(combined_feat)
        outputs['bbox_3d'] = self.bbox_3d_head(combined_feat)

        # Depth estimation
        if self.depth_estimation:
            outputs['depth'] = self.depth_head(combined_feat) * 80.0  # Scale to 80m max

        # Semantic segmentation (upsample to input resolution)
        seg_logits = self.seg_head(combined_feat)
        outputs['segmentation'] = F.interpolate(
            seg_logits, size=x.shape[2:], mode='bilinear', align_corners=False
        )

        return outputs


class MBConvBlock(nn.Module):
    """Mobile Inverted Residual Block"""

    def __init__(self, in_channels, out_channels, expand_ratio, stride):
        super().__init__()

        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels

        hidden_dim = in_channels * expand_ratio

        self.conv = nn.Sequential(
            # Pointwise expansion
            nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),

            # Depthwise convolution
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),

            # Pointwise linear projection
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)


class StereoDepthNet(nn.Module):
    """Stereo depth estimation network for automotive cameras"""

    def __init__(self, max_disparity: int = 192):
        super().__init__()

        self.max_disparity = max_disparity

        # Feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # Cost volume construction and aggregation
        self.cost_aggregation = nn.Sequential(
            nn.Conv3d(128, 64, 3, 1, 1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            nn.Conv3d(64, 32, 3, 1, 1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            nn.Conv3d(32, 1, 3, 1, 1)
        )

    def forward(self, left_img, right_img):
        # Extract features
        left_feat = self.feature_extractor(left_img)
        right_feat = self.feature_extractor(right_img)

        # Build cost volume
        batch_size, channels, height, width = left_feat.shape
        cost_volume = torch.zeros(
            batch_size, channels, self.max_disparity//4, height, width,
            device=left_feat.device
        )

        for d in range(self.max_disparity//4):
            if d == 0:
                cost_volume[:, :, d, :, :] = left_feat - right_feat
            else:
                cost_volume[:, :, d, :, :-d] = left_feat[:, :, :, :-d] - right_feat[:, :, :, d:]

        # Cost aggregation
        cost_volume = self.cost_aggregation(cost_volume)
        cost_volume = cost_volume.squeeze(1)

        # Disparity regression
        disparity_indices = torch.arange(0, self.max_disparity//4, dtype=torch.float32, device=cost_volume.device)
        disparity_indices = disparity_indices.view(1, -1, 1, 1)

        prob_volume = F.softmax(-cost_volume, dim=1)
        disparity = torch.sum(prob_volume * disparity_indices, dim=1, keepdim=True)

        # Convert disparity to depth (assuming known camera parameters)
        # depth = focal_length * baseline / (disparity + epsilon)
        focal_length = 721.5  # Typical automotive camera focal length
        baseline = 0.54  # Typical stereo baseline in meters
        depth = focal_length * baseline / (disparity * 4 + 1e-6)  # Scale back disparity

        return depth, prob_volume


class WeatherAdaptationNet(nn.Module):
    """Neural network for weather condition adaptation"""

    def __init__(self, input_channels: int = 3):
        super().__init__()

        # Weather classification branch
        self.weather_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(input_channels * 49, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, len(WeatherCondition))
        )

        # Image enhancement for different conditions
        self.enhancement_networks = nn.ModuleDict({
            'rain': self._create_enhancement_net(),
            'snow': self._create_enhancement_net(),
            'fog': self._create_enhancement_net(),
            'night': self._create_enhancement_net()
        })

    def _create_enhancement_net(self):
        """Create image enhancement network"""
        return nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Classify weather condition
        weather_logits = self.weather_classifier(x)
        weather_probs = F.softmax(weather_logits, dim=1)

        # Enhance image based on weather
        enhanced_images = {}
        for condition, enhancer in self.enhancement_networks.items():
            enhanced_images[condition] = enhancer(x)

        return weather_probs, enhanced_images


class SceneUnderstandingSystem:
    """Main scene understanding system for ADAS Phase 7"""

    def __init__(self,
                 model_path: Optional[str] = None,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 use_stereo: bool = True):

        self.device = device
        self.use_stereo = use_stereo
        self.logger = logging.getLogger(__name__)

        # Initialize models
        self.detection_model = EfficientNet3D(num_classes=10, depth_estimation=not use_stereo).to(device)

        if use_stereo:
            self.stereo_model = StereoDepthNet().to(device)

        self.weather_model = WeatherAdaptationNet().to(device)

        # Load pretrained weights
        if model_path:
            self.load_models(model_path)

        # Camera calibration (typical automotive stereo setup)
        self.camera_matrix = np.array([
            [721.5377, 0.0, 609.5593],
            [0.0, 721.5377, 172.854],
            [0.0, 0.0, 1.0]
        ])

        self.baseline = 0.54  # meters

        # Semantic class names (Cityscapes)
        self.class_names = [
            'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
            'traffic_light', 'traffic_sign', 'vegetation', 'terrain',
            'sky', 'person', 'rider', 'car', 'truck', 'bus',
            'train', 'motorcycle', 'bicycle'
        ]

        # Object detection class names
        self.object_classes = [
            'car', 'pedestrian', 'cyclist', 'truck', 'bus',
            'motorcycle', 'traffic_sign', 'traffic_light',
            'construction_vehicle', 'barrier'
        ]

        # Performance tracking
        self.processing_times = []
        self.frame_count = 0

    def process_frame(self,
                     left_image: np.ndarray,
                     right_image: Optional[np.ndarray] = None,
                     camera_params: Optional[Dict] = None) -> SceneContext:
        """
        Process single frame for complete scene understanding

        Args:
            left_image: Main camera image (RGB, H x W x 3)
            right_image: Right stereo image (RGB, H x W x 3)
            camera_params: Camera calibration parameters

        Returns:
            Complete scene understanding result
        """

        start_time = time.time()

        # Preprocess images
        left_tensor = self._preprocess_image(left_image)

        # Weather adaptation
        weather_probs, enhanced_images = self.weather_model(left_tensor)
        weather_condition = self._classify_weather(weather_probs)
        lighting_condition = self._classify_lighting(left_image)

        # Use enhanced image if needed
        if weather_condition != WeatherCondition.CLEAR:
            enhanced_tensor = enhanced_images.get(weather_condition.value, left_tensor)
        else:
            enhanced_tensor = left_tensor

        # Main detection and segmentation
        with torch.no_grad():
            detection_outputs = self.detection_model(enhanced_tensor)

        # Parse detection results
        objects_3d = self._parse_detections(detection_outputs, left_image.shape)
        segmentation = self._parse_segmentation(detection_outputs, left_image.shape)

        # Depth estimation
        if self.use_stereo and right_image is not None:
            right_tensor = self._preprocess_image(right_image)
            with torch.no_grad():
                depth_map, depth_confidence = self.stereo_model(enhanced_tensor, right_tensor)

            depth_estimation = self._create_depth_result(
                depth_map, depth_confidence, left_image.shape
            )
        else:
            # Use monocular depth from detection model
            depth_map = detection_outputs['depth']
            depth_estimation = self._create_depth_result(
                depth_map, torch.ones_like(depth_map), left_image.shape
            )

        # Advanced scene analysis
        road_geometry = self._extract_road_geometry(segmentation, depth_estimation)
        traffic_signs = self._detect_traffic_signs(objects_3d, segmentation)
        lane_structure = self._analyze_lane_structure(segmentation, depth_estimation)
        free_space = self._compute_free_space(objects_3d, depth_estimation, left_image.shape)

        # Create complete scene context
        scene_context = SceneContext(
            objects_3d=objects_3d,
            segmentation=segmentation,
            depth=depth_estimation,
            weather=weather_condition,
            lighting=lighting_condition,
            road_geometry=road_geometry,
            traffic_signs=traffic_signs,
            lane_structure=lane_structure,
            free_space=free_space,
            timestamp=time.time()
        )

        # Update performance metrics
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        self.frame_count += 1

        # Keep only last 100 times
        if len(self.processing_times) > 100:
            self.processing_times.pop(0)

        self.logger.debug(f"Frame processed in {processing_time:.3f}s")

        return scene_context

    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for neural network"""

        # Resize to model input size
        image_resized = cv2.resize(image, (640, 384))

        # Normalize
        image_normalized = image_resized.astype(np.float32) / 255.0

        # Convert to tensor
        image_tensor = torch.from_numpy(image_normalized.transpose(2, 0, 1)).unsqueeze(0)

        return image_tensor.to(self.device)

    def _classify_weather(self, weather_probs: torch.Tensor) -> WeatherCondition:
        """Classify weather condition from probabilities"""

        weather_idx = torch.argmax(weather_probs, dim=1).item()
        weather_conditions = list(WeatherCondition)

        return weather_conditions[weather_idx]

    def _classify_lighting(self, image: np.ndarray) -> LightingCondition:
        """Classify lighting condition from image statistics"""

        # Convert to grayscale for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Calculate statistics
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)

        # Simple rule-based classification
        if mean_intensity > 150:
            return LightingCondition.DAY_BRIGHT
        elif mean_intensity > 100:
            return LightingCondition.DAY_OVERCAST
        elif mean_intensity > 50:
            return LightingCondition.NIGHT_LIT
        else:
            return LightingCondition.NIGHT_DARK

    def _parse_detections(self, outputs: Dict, image_shape: Tuple) -> List[Object3D]:
        """Parse neural network outputs to 3D objects"""

        objects = []

        # Extract predictions
        cls_pred = outputs['cls'].squeeze(0).cpu().numpy()
        bbox_2d_pred = outputs['bbox_2d'].squeeze(0).cpu().numpy()
        bbox_3d_pred = outputs['bbox_3d'].squeeze(0).cpu().numpy()

        # Apply NMS and extract detections
        # Simplified implementation - in production use proper anchor generation
        height, width = cls_pred.shape[1:]

        for y in range(0, height, 8):  # Stride of 8
            for x in range(0, width, 8):
                for anchor in range(9):  # 9 anchors per location
                    cls_idx = anchor * len(self.object_classes)
                    bbox_2d_idx = anchor * 4
                    bbox_3d_idx = anchor * 7

                    # Get maximum class probability
                    class_scores = cls_pred[cls_idx:cls_idx+len(self.object_classes), y, x]
                    max_class = np.argmax(class_scores)
                    confidence = class_scores[max_class]

                    if confidence > 0.5:  # Confidence threshold
                        # Extract 2D bbox
                        bbox_2d = bbox_2d_pred[bbox_2d_idx:bbox_2d_idx+4, y, x]

                        # Extract 3D bbox
                        bbox_3d = bbox_3d_pred[bbox_3d_idx:bbox_3d_idx+7, y, x]

                        # Convert to image coordinates
                        scale_x = image_shape[1] / width
                        scale_y = image_shape[0] / height

                        bbox_2d_scaled = bbox_2d * np.array([scale_x, scale_y, scale_x, scale_y])

                        # Create 3D object
                        obj = Object3D(
                            class_id=max_class,
                            class_name=self.object_classes[max_class],
                            confidence=confidence,
                            bbox_2d=bbox_2d_scaled,
                            bbox_3d=bbox_3d,
                            center_3d=bbox_3d[:3],
                            velocity=np.zeros(3),  # TODO: Track velocity
                            distance=np.linalg.norm(bbox_3d[:3])
                        )

                        objects.append(obj)

        # Apply NMS to remove duplicate detections
        objects = self._apply_nms(objects)

        return objects

    def _apply_nms(self, objects: List[Object3D], iou_threshold: float = 0.5) -> List[Object3D]:
        """Apply Non-Maximum Suppression to object detections"""

        if not objects:
            return objects

        # Sort by confidence
        objects.sort(key=lambda x: x.confidence, reverse=True)

        keep = []
        while objects:
            # Keep highest confidence object
            best = objects.pop(0)
            keep.append(best)

            # Remove overlapping objects
            remaining = []
            for obj in objects:
                iou = self._calculate_iou(best.bbox_2d, obj.bbox_2d)
                if iou < iou_threshold:
                    remaining.append(obj)

            objects = remaining

        return keep

    def _calculate_iou(self, bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """Calculate IoU between two 2D bounding boxes"""

        # Calculate intersection
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)

        # Calculate union
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def _parse_segmentation(self, outputs: Dict, image_shape: Tuple) -> SceneSegmentation:
        """Parse segmentation output"""

        seg_logits = outputs['segmentation'].squeeze(0).cpu()
        seg_probs = F.softmax(seg_logits, dim=0)
        seg_pred = torch.argmax(seg_probs, dim=0).numpy()

        # Resize to original image size
        seg_resized = cv2.resize(
            seg_pred.astype(np.uint8),
            (image_shape[1], image_shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

        # Extract confidence map
        confidence_map = torch.max(seg_probs, dim=0)[0].numpy()
        confidence_resized = cv2.resize(confidence_map, (image_shape[1], image_shape[0]))

        # Extract drivable area (road + sidewalk classes)
        drivable_classes = [0, 1]  # road, sidewalk
        drivable_area = np.isin(seg_resized, drivable_classes)

        # Extract lane markings (simple edge detection on road class)
        road_mask = (seg_resized == 0).astype(np.uint8)
        edges = cv2.Canny(road_mask * 255, 50, 150)

        # Find lane line contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        lane_markings = [contour.squeeze() for contour in contours if len(contour) > 10]

        return SceneSegmentation(
            semantic_map=seg_resized,
            instance_map=np.zeros_like(seg_resized),  # TODO: Implement instance segmentation
            confidence_map=confidence_resized,
            class_names=self.class_names,
            drivable_area=drivable_area,
            lane_markings=lane_markings
        )

    def _create_depth_result(self,
                           depth_tensor: torch.Tensor,
                           confidence_tensor: torch.Tensor,
                           image_shape: Tuple) -> DepthEstimation:
        """Create depth estimation result"""

        # Convert to numpy
        depth_map = depth_tensor.squeeze().cpu().numpy()
        confidence_map = confidence_tensor.squeeze().cpu().numpy()

        # Resize to original image size
        depth_resized = cv2.resize(depth_map, (image_shape[1], image_shape[0]))
        confidence_resized = cv2.resize(confidence_map, (image_shape[1], image_shape[0]))

        # Generate 3D point cloud
        point_cloud = self._depth_to_pointcloud(depth_resized, self.camera_matrix)

        # Calculate surface normals
        normals = self._calculate_surface_normals(depth_resized)

        # Detect occlusions
        occlusion_map = self._detect_occlusions(depth_resized)

        return DepthEstimation(
            depth_map=depth_resized,
            confidence_map=confidence_resized,
            point_cloud=point_cloud,
            normals=normals,
            occlusion_map=occlusion_map
        )

    def _depth_to_pointcloud(self, depth_map: np.ndarray, camera_matrix: np.ndarray) -> np.ndarray:
        """Convert depth map to 3D point cloud"""

        height, width = depth_map.shape

        # Create coordinate grids
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        u = u.flatten()
        v = v.flatten()
        depth = depth_map.flatten()

        # Valid depth points
        valid = depth > 0
        u = u[valid]
        v = v[valid]
        depth = depth[valid]

        # Convert to 3D coordinates
        fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
        cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth

        point_cloud = np.stack([x, y, z], axis=1)

        return point_cloud

    def _calculate_surface_normals(self, depth_map: np.ndarray) -> np.ndarray:
        """Calculate surface normals from depth map"""

        # Calculate gradients
        grad_x = cv2.Sobel(depth_map, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=3)

        # Calculate normals
        normals = np.zeros((depth_map.shape[0], depth_map.shape[1], 3))
        normals[:, :, 0] = -grad_x
        normals[:, :, 1] = -grad_y
        normals[:, :, 2] = 1.0

        # Normalize
        norm = np.linalg.norm(normals, axis=2, keepdims=True)
        normals = normals / (norm + 1e-6)

        return normals.reshape(-1, 3)

    def _detect_occlusions(self, depth_map: np.ndarray) -> np.ndarray:
        """Detect occlusion boundaries in depth map"""

        # Calculate depth discontinuities
        grad_x = np.abs(cv2.Sobel(depth_map, cv2.CV_64F, 1, 0, ksize=3))
        grad_y = np.abs(cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=3))

        # Combine gradients
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # Threshold for occlusion detection
        occlusion_threshold = 2.0  # meters
        occlusion_map = grad_magnitude > occlusion_threshold

        return occlusion_map

    def _extract_road_geometry(self,
                             segmentation: SceneSegmentation,
                             depth: DepthEstimation) -> Dict[str, np.ndarray]:
        """Extract 3D road geometry"""

        # Get road points from segmentation and depth
        road_mask = (segmentation.semantic_map == 0)  # Road class

        # Sample road points
        y_coords, x_coords = np.where(road_mask)
        if len(y_coords) == 0:
            return {}

        # Get 3D coordinates of road points
        depths = depth.depth_map[y_coords, x_coords]
        valid = depths > 0

        if np.sum(valid) == 0:
            return {}

        y_coords = y_coords[valid]
        x_coords = x_coords[valid]
        depths = depths[valid]

        # Convert to 3D coordinates
        fx, fy = self.camera_matrix[0, 0], self.camera_matrix[1, 1]
        cx, cy = self.camera_matrix[0, 2], self.camera_matrix[1, 2]

        x_3d = (x_coords - cx) * depths / fx
        y_3d = (y_coords - cy) * depths / fy
        z_3d = depths

        road_points = np.stack([x_3d, y_3d, z_3d], axis=1)

        # Fit road plane
        if len(road_points) > 100:
            # RANSAC plane fitting
            from sklearn.linear_model import RANSACRegressor

            X = road_points[:, [0, 2]]  # x, z coordinates
            y = road_points[:, 1]       # y coordinate (height)

            ransac = RANSACRegressor(random_state=42)
            ransac.fit(X, y)

            # Road plane parameters: y = ax + bz + c
            road_plane = np.array([ransac.estimator_.coef_[0],
                                 ransac.estimator_.coef_[1],
                                 ransac.estimator_.intercept_])
        else:
            road_plane = np.array([0.0, 0.0, 0.0])

        return {
            'road_points': road_points,
            'road_plane': road_plane,
            'road_normal': np.array([road_plane[0], -1.0, road_plane[1]])
        }

    def _detect_traffic_signs(self,
                            objects_3d: List[Object3D],
                            segmentation: SceneSegmentation) -> List[Dict]:
        """Detect and classify traffic signs"""

        traffic_signs = []

        for obj in objects_3d:
            if obj.class_name in ['traffic_sign', 'traffic_light']:
                # Extract sign region from segmentation
                x1, y1, x2, y2 = obj.bbox_2d.astype(int)

                # Ensure bounds are valid
                h, w = segmentation.semantic_map.shape
                x1 = max(0, min(x1, w-1))
                x2 = max(0, min(x2, w-1))
                y1 = max(0, min(y1, h-1))
                y2 = max(0, min(y2, h-1))

                if x2 > x1 and y2 > y1:
                    sign_region = segmentation.semantic_map[y1:y2, x1:x2]

                    # Simple classification based on size and shape
                    height = y2 - y1
                    width = x2 - x1
                    aspect_ratio = width / height if height > 0 else 1.0

                    # Classify sign type
                    if aspect_ratio > 1.5:
                        sign_type = "rectangular"
                    elif 0.8 <= aspect_ratio <= 1.2:
                        sign_type = "square"
                    else:
                        sign_type = "vertical"

                    traffic_sign = {
                        'object_id': id(obj),
                        'type': sign_type,
                        'class': obj.class_name,
                        'confidence': obj.confidence,
                        'position_3d': obj.center_3d,
                        'bbox_2d': obj.bbox_2d,
                        'distance': obj.distance
                    }

                    traffic_signs.append(traffic_sign)

        return traffic_signs

    def _analyze_lane_structure(self,
                              segmentation: SceneSegmentation,
                              depth: DepthEstimation) -> Dict[str, List[np.ndarray]]:
        """Analyze lane structure from segmentation and depth"""

        lane_structure = {
            'left_lanes': [],
            'right_lanes': [],
            'center_lines': []
        }

        # Process lane markings
        for lane_marking in segmentation.lane_markings:
            if len(lane_marking) > 20:  # Minimum points for valid lane
                # Convert to 3D coordinates
                lane_3d = []
                for point in lane_marking:
                    if len(point) >= 2:
                        x, y = point[0], point[1]
                        if 0 <= y < depth.depth_map.shape[0] and 0 <= x < depth.depth_map.shape[1]:
                            depth_val = depth.depth_map[y, x]
                            if depth_val > 0:
                                # Convert to 3D
                                fx, fy = self.camera_matrix[0, 0], self.camera_matrix[1, 1]
                                cx, cy = self.camera_matrix[0, 2], self.camera_matrix[1, 2]

                                x_3d = (x - cx) * depth_val / fx
                                y_3d = (y - cy) * depth_val / fy
                                z_3d = depth_val

                                lane_3d.append([x_3d, y_3d, z_3d])

                if len(lane_3d) > 10:
                    lane_3d = np.array(lane_3d)

                    # Classify lane based on position relative to ego vehicle
                    mean_x = np.mean(lane_3d[:, 0])

                    if mean_x < -1.0:  # Left side
                        lane_structure['left_lanes'].append(lane_3d)
                    elif mean_x > 1.0:   # Right side
                        lane_structure['right_lanes'].append(lane_3d)
                    else:               # Center
                        lane_structure['center_lines'].append(lane_3d)

        return lane_structure

    def _compute_free_space(self,
                          objects_3d: List[Object3D],
                          depth: DepthEstimation,
                          image_shape: Tuple) -> np.ndarray:
        """Compute free space occupancy grid"""

        # Define grid parameters
        grid_size = 0.2  # 20cm resolution
        grid_width = int(40 / grid_size)  # 40m width
        grid_height = int(80 / grid_size)  # 80m depth

        # Initialize occupancy grid (0 = free, 1 = occupied, 0.5 = unknown)
        occupancy_grid = np.ones((grid_height, grid_width)) * 0.5

        # Mark free space from depth information
        point_cloud = depth.point_cloud

        for point in point_cloud[::100]:  # Subsample for efficiency
            x, y, z = point

            # Convert to grid coordinates
            grid_x = int((x + 20) / grid_size)  # Shift by 20m to center
            grid_y = int(z / grid_size)

            # Check bounds
            if 0 <= grid_x < grid_width and 0 <= grid_y < grid_height:
                # Mark as free if on ground plane
                if abs(y + 1.5) < 0.3:  # Assuming camera 1.5m above ground
                    occupancy_grid[grid_y, grid_x] = 0.0

        # Mark occupied space from detected objects
        for obj in objects_3d:
            x, y, z = obj.center_3d

            # Get object dimensions from 3D bbox
            w, h, l = obj.bbox_3d[3:6] if len(obj.bbox_3d) > 6 else [2.0, 2.0, 4.0]

            # Mark object footprint as occupied
            x_min = int((x - w/2 + 20) / grid_size)
            x_max = int((x + w/2 + 20) / grid_size)
            y_min = int((z - l/2) / grid_size)
            y_max = int((z + l/2) / grid_size)

            # Clamp to grid bounds
            x_min = max(0, min(x_min, grid_width-1))
            x_max = max(0, min(x_max, grid_width-1))
            y_min = max(0, min(y_min, grid_height-1))
            y_max = max(0, min(y_max, grid_height-1))

            occupancy_grid[y_min:y_max+1, x_min:x_max+1] = 1.0

        return occupancy_grid

    def load_models(self, model_path: str):
        """Load pretrained model weights"""

        try:
            checkpoint = torch.load(model_path, map_location=self.device)

            if 'detection_model' in checkpoint:
                self.detection_model.load_state_dict(checkpoint['detection_model'])

            if 'stereo_model' in checkpoint and self.use_stereo:
                self.stereo_model.load_state_dict(checkpoint['stereo_model'])

            if 'weather_model' in checkpoint:
                self.weather_model.load_state_dict(checkpoint['weather_model'])

            self.logger.info(f"Models loaded from {model_path}")

        except Exception as e:
            self.logger.warning(f"Could not load models: {e}")

    def get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics"""

        if not self.processing_times:
            return {}

        avg_time = np.mean(self.processing_times)
        max_time = np.max(self.processing_times)
        fps = 1.0 / avg_time if avg_time > 0 else 0.0

        return {
            'average_processing_time_ms': avg_time * 1000,
            'max_processing_time_ms': max_time * 1000,
            'fps': fps,
            'frames_processed': self.frame_count,
            'memory_usage_mb': torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        }


if __name__ == "__main__":
    # Example usage for automotive testing
    logging.basicConfig(level=logging.INFO)

    # Initialize scene understanding system
    scene_system = SceneUnderstandingSystem(use_stereo=True)

    # Create dummy test images
    left_image = np.random.randint(0, 255, (384, 640, 3), dtype=np.uint8)
    right_image = np.random.randint(0, 255, (384, 640, 3), dtype=np.uint8)

    # Process frame
    scene_context = scene_system.process_frame(left_image, right_image)

    # Print results
    print(f"Detected {len(scene_context.objects_3d)} objects")
    print(f"Weather: {scene_context.weather.value}")
    print(f"Lighting: {scene_context.lighting.value}")
    print(f"Drivable area: {np.sum(scene_context.segmentation.drivable_area)} pixels")
    print(f"Lane markings: {len(scene_context.segmentation.lane_markings)}")
    print(f"Traffic signs: {len(scene_context.traffic_signs)}")

    # Print performance metrics
    metrics = scene_system.get_performance_metrics()
    for key, value in metrics.items():
        print(f"{key}: {value:.2f}")