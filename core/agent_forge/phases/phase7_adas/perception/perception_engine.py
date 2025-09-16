"""
Perception Engine - Object detection, tracking, and scene understanding for ADAS
Implements multi-object tracking, lane detection, traffic sign recognition, and pedestrian detection
Automotive Safety Integrity Level (ASIL-D) compliant
"""

import asyncio
import logging
import time
import threading
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import numpy as np
from abc import ABC, abstractmethod
import uuid
import math

class ObjectType(Enum):
    """Types of objects that can be detected"""
    VEHICLE = "vehicle"
    PEDESTRIAN = "pedestrian"
    CYCLIST = "cyclist"
    MOTORCYCLE = "motorcycle"
    TRUCK = "truck"
    BUS = "bus"
    TRAFFIC_SIGN = "traffic_sign"
    TRAFFIC_LIGHT = "traffic_light"
    OBSTACLE = "obstacle"
    UNKNOWN = "unknown"

class TrackState(Enum):
    """Object tracking states"""
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    LOST = "lost"
    DELETED = "deleted"

class LaneType(Enum):
    """Lane marking types"""
    SOLID = "solid"
    DASHED = "dashed"
    DOUBLE_SOLID = "double_solid"
    SOLID_DASHED = "solid_dashed"
    DASHED_SOLID = "dashed_solid"

@dataclass
class BoundingBox:
    """2D bounding box representation"""
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    confidence: float = 1.0

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x_min + self.x_max) / 2, (self.y_min + self.y_max) / 2)

    @property
    def area(self) -> float:
        return (self.x_max - self.x_min) * (self.y_max - self.y_min)

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

@dataclass
class Detection:
    """Object detection result"""
    detection_id: str
    object_type: ObjectType
    bbox: BoundingBox
    position_3d: Optional[Tuple[float, float, float]]
    confidence: float
    timestamp: float
    features: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Track:
    """Object tracking state"""
    track_id: str
    object_type: ObjectType
    state: TrackState
    positions: deque  # Historical positions
    velocities: deque  # Historical velocities
    bbox_history: deque  # Historical bounding boxes
    confidence_history: deque  # Historical confidence scores
    created_time: float
    last_update_time: float
    hits: int  # Number of detections associated
    misses: int  # Number of frames without detection
    predicted_position: Optional[Tuple[float, float, float]] = None
    predicted_velocity: Optional[Tuple[float, float, float]] = None

    def __post_init__(self):
        if not hasattr(self, 'positions') or self.positions is None:
            self.positions = deque(maxlen=30)
        if not hasattr(self, 'velocities') or self.velocities is None:
            self.velocities = deque(maxlen=30)
        if not hasattr(self, 'bbox_history') or self.bbox_history is None:
            self.bbox_history = deque(maxlen=30)
        if not hasattr(self, 'confidence_history') or self.confidence_history is None:
            self.confidence_history = deque(maxlen=30)

@dataclass
class LaneMarking:
    """Lane marking representation"""
    lane_id: str
    points: List[Tuple[float, float]]
    lane_type: LaneType
    confidence: float
    width: float
    color: str = "white"

@dataclass
class LaneInfo:
    """Complete lane information"""
    left_markings: List[LaneMarking]
    right_markings: List[LaneMarking]
    center_line: Optional[List[Tuple[float, float]]]
    lane_width: float
    curvature: float
    confidence: float

@dataclass
class TrafficSign:
    """Traffic sign detection"""
    sign_id: str
    sign_type: str
    bbox: BoundingBox
    position_3d: Optional[Tuple[float, float, float]]
    confidence: float
    text_content: Optional[str] = None
    speed_limit: Optional[int] = None
    distance_estimate: Optional[float] = None

class ObjectDetector:
    """Object detection module using AI models"""

    def __init__(self, model_config: Dict):
        self.model_config = model_config
        self.confidence_threshold = model_config.get('confidence_threshold', 0.5)
        self.nms_threshold = model_config.get('nms_threshold', 0.4)
        self.detection_classes = self._load_detection_classes()

        # Performance metrics
        self.detection_metrics = {
            'total_detections': 0,
            'average_confidence': 0.0,
            'processing_time_ms': 0.0,
            'fps': 0.0
        }

    def _load_detection_classes(self) -> Dict[int, ObjectType]:
        """Load object detection class mappings"""
        # Standard COCO classes mapped to ObjectType
        return {
            0: ObjectType.VEHICLE,    # car
            1: ObjectType.VEHICLE,    # truck
            2: ObjectType.BUS,        # bus
            3: ObjectType.MOTORCYCLE, # motorcycle
            4: ObjectType.CYCLIST,    # bicycle
            5: ObjectType.PEDESTRIAN, # person
            # Add more mappings as needed
        }

    async def detect_objects(self, image: np.ndarray, timestamp: float) -> List[Detection]:
        """Detect objects in image"""
        start_time = time.perf_counter()

        try:
            # Placeholder for actual object detection inference
            # In production, this would use YOLO, SSD, or similar models
            detections = await self._run_detection_model(image)

            # Post-process detections
            filtered_detections = self._post_process_detections(detections, timestamp)

            # Update metrics
            processing_time = (time.perf_counter() - start_time) * 1000
            self._update_detection_metrics(filtered_detections, processing_time)

            return filtered_detections

        except Exception as e:
            logging.error(f"Object detection failed: {e}")
            return []

    async def _run_detection_model(self, image: np.ndarray) -> List[Dict]:
        """Run the actual detection model (placeholder)"""
        # Simulate processing delay
        await asyncio.sleep(0.02)  # 20ms processing time

        # Simulate detection results
        height, width = image.shape[:2]

        # Generate synthetic detections for demonstration
        detections = []

        # Vehicle detection
        if np.random.random() > 0.3:  # 70% chance of vehicle detection
            detections.append({
                'class_id': 0,
                'confidence': 0.85 + np.random.random() * 0.15,
                'bbox': [
                    width * 0.3, height * 0.4,  # x_min, y_min
                    width * 0.7, height * 0.8   # x_max, y_max
                ]
            })

        # Pedestrian detection
        if np.random.random() > 0.6:  # 40% chance of pedestrian detection
            detections.append({
                'class_id': 5,
                'confidence': 0.75 + np.random.random() * 0.2,
                'bbox': [
                    width * 0.1, height * 0.3,
                    width * 0.25, height * 0.9
                ]
            })

        return detections

    def _post_process_detections(self, raw_detections: List[Dict], timestamp: float) -> List[Detection]:
        """Post-process raw detections"""
        detections = []

        for raw_det in raw_detections:
            # Filter by confidence
            if raw_det['confidence'] < self.confidence_threshold:
                continue

            # Create detection object
            class_id = raw_det['class_id']
            object_type = self.detection_classes.get(class_id, ObjectType.UNKNOWN)

            bbox = BoundingBox(
                x_min=raw_det['bbox'][0],
                y_min=raw_det['bbox'][1],
                x_max=raw_det['bbox'][2],
                y_max=raw_det['bbox'][3],
                confidence=raw_det['confidence']
            )

            detection = Detection(
                detection_id=str(uuid.uuid4()),
                object_type=object_type,
                bbox=bbox,
                position_3d=self._estimate_3d_position(bbox),
                confidence=raw_det['confidence'],
                timestamp=timestamp
            )

            detections.append(detection)

        # Apply Non-Maximum Suppression
        return self._apply_nms(detections)

    def _estimate_3d_position(self, bbox: BoundingBox) -> Tuple[float, float, float]:
        """Estimate 3D position from 2D bounding box (simplified)"""
        # This is a simplified estimation
        # In production, use camera calibration and stereo vision or depth estimation

        center_x, center_y = bbox.center

        # Estimate distance based on object size (rough approximation)
        object_height_pixels = bbox.height
        assumed_real_height = 1.7  # Assume 1.7m for vehicles/pedestrians

        # Simple distance estimation (requires camera calibration)
        focal_length = 800  # Placeholder focal length
        distance = (assumed_real_height * focal_length) / max(object_height_pixels, 1)

        # Convert to 3D coordinates (simplified)
        x = distance  # Forward distance
        y = (center_x - 960) / 960 * distance * 0.5  # Lateral offset
        z = 0.0  # Ground level

        return (x, y, z)

    def _apply_nms(self, detections: List[Detection]) -> List[Detection]:
        """Apply Non-Maximum Suppression to remove duplicate detections"""
        if not detections:
            return []

        # Sort by confidence
        detections.sort(key=lambda x: x.confidence, reverse=True)

        filtered_detections = []

        for detection in detections:
            # Check overlap with already selected detections
            overlaps = False
            for selected in filtered_detections:
                if self._calculate_iou(detection.bbox, selected.bbox) > self.nms_threshold:
                    overlaps = True
                    break

            if not overlaps:
                filtered_detections.append(detection)

        return filtered_detections

    def _calculate_iou(self, bbox1: BoundingBox, bbox2: BoundingBox) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes"""
        # Calculate intersection
        x_left = max(bbox1.x_min, bbox2.x_min)
        y_top = max(bbox1.y_min, bbox2.y_min)
        x_right = min(bbox1.x_max, bbox2.x_max)
        y_bottom = min(bbox1.y_max, bbox2.y_max)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # Calculate union
        bbox1_area = bbox1.area
        bbox2_area = bbox2.area
        union_area = bbox1_area + bbox2_area - intersection_area

        return intersection_area / union_area if union_area > 0 else 0.0

    def _update_detection_metrics(self, detections: List[Detection], processing_time: float):
        """Update detection performance metrics"""
        self.detection_metrics['total_detections'] += len(detections)
        self.detection_metrics['processing_time_ms'] = processing_time
        self.detection_metrics['fps'] = 1000.0 / processing_time if processing_time > 0 else 0.0

        if detections:
            avg_conf = sum(det.confidence for det in detections) / len(detections)
            self.detection_metrics['average_confidence'] = avg_conf

class MultiObjectTracker:
    """Multi-object tracking using Kalman filter and data association"""

    def __init__(self, tracker_config: Dict):
        self.tracker_config = tracker_config
        self.tracks = {}  # track_id -> Track
        self.next_track_id = 1

        # Tracking parameters
        self.max_distance_threshold = tracker_config.get('max_distance_threshold', 50.0)
        self.min_hits = tracker_config.get('min_hits', 3)
        self.max_age = tracker_config.get('max_age', 30)

        # Kalman filter parameters (simplified)
        self.process_noise = tracker_config.get('process_noise', 1.0)
        self.measurement_noise = tracker_config.get('measurement_noise', 10.0)

    def update(self, detections: List[Detection], timestamp: float) -> List[Track]:
        """Update tracker with new detections"""
        try:
            # Predict existing tracks
            self._predict_tracks(timestamp)

            # Associate detections with tracks
            matched_tracks, unmatched_detections = self._associate_detections(detections)

            # Update matched tracks
            for track_id, detection in matched_tracks:
                self._update_track(track_id, detection, timestamp)

            # Create new tracks for unmatched detections
            for detection in unmatched_detections:
                self._create_new_track(detection, timestamp)

            # Manage track lifecycle
            self._manage_track_lifecycle(timestamp)

            # Return confirmed tracks
            return [track for track in self.tracks.values()
                   if track.state == TrackState.CONFIRMED]

        except Exception as e:
            logging.error(f"Tracking update failed: {e}")
            return list(self.tracks.values())

    def _predict_tracks(self, timestamp: float):
        """Predict track positions using motion model"""
        for track in self.tracks.values():
            if track.state == TrackState.DELETED:
                continue

            # Simple constant velocity model
            if track.positions and track.velocities:
                dt = timestamp - track.last_update_time
                last_pos = track.positions[-1]
                last_vel = track.velocities[-1] if track.velocities else (0, 0, 0)

                # Predict next position
                predicted_x = last_pos[0] + last_vel[0] * dt
                predicted_y = last_pos[1] + last_vel[1] * dt
                predicted_z = last_pos[2] + last_vel[2] * dt

                track.predicted_position = (predicted_x, predicted_y, predicted_z)

    def _associate_detections(self, detections: List[Detection]) -> Tuple[List[Tuple[str, Detection]], List[Detection]]:
        """Associate detections with existing tracks using Hungarian algorithm (simplified)"""
        if not detections or not self.tracks:
            return [], detections

        # Build cost matrix
        active_tracks = [t for t in self.tracks.values() if t.state != TrackState.DELETED]
        cost_matrix = np.full((len(active_tracks), len(detections)), np.inf)

        for i, track in enumerate(active_tracks):
            for j, detection in enumerate(detections):
                # Calculate association cost (distance + appearance)
                if detection.position_3d and track.predicted_position:
                    distance = self._calculate_3d_distance(
                        detection.position_3d, track.predicted_position
                    )

                    if distance < self.max_distance_threshold:
                        # Cost combines distance and confidence
                        cost_matrix[i, j] = distance / (detection.confidence + 0.1)

        # Simplified assignment (greedy approach)
        # In production, use Hungarian algorithm or similar
        matched_tracks = []
        used_detections = set()
        used_tracks = set()

        # Find minimum cost assignments
        for _ in range(min(len(active_tracks), len(detections))):
            min_cost = np.inf
            best_track_idx, best_det_idx = -1, -1

            for i in range(len(active_tracks)):
                if i in used_tracks:
                    continue
                for j in range(len(detections)):
                    if j in used_detections:
                        continue
                    if cost_matrix[i, j] < min_cost:
                        min_cost = cost_matrix[i, j]
                        best_track_idx, best_det_idx = i, j

            if min_cost < np.inf:
                track = active_tracks[best_track_idx]
                detection = detections[best_det_idx]
                matched_tracks.append((track.track_id, detection))
                used_tracks.add(best_track_idx)
                used_detections.add(best_det_idx)

        # Unmatched detections
        unmatched_detections = [det for i, det in enumerate(detections)
                              if i not in used_detections]

        return matched_tracks, unmatched_detections

    def _calculate_3d_distance(self, pos1: Tuple[float, float, float],
                              pos2: Tuple[float, float, float]) -> float:
        """Calculate 3D Euclidean distance"""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(pos1, pos2)))

    def _update_track(self, track_id: str, detection: Detection, timestamp: float):
        """Update track with new detection"""
        track = self.tracks[track_id]

        # Update position history
        if detection.position_3d:
            track.positions.append(detection.position_3d)

            # Calculate velocity
            if len(track.positions) >= 2:
                dt = timestamp - track.last_update_time
                if dt > 0:
                    last_pos = track.positions[-2]
                    current_pos = track.positions[-1]
                    velocity = (
                        (current_pos[0] - last_pos[0]) / dt,
                        (current_pos[1] - last_pos[1]) / dt,
                        (current_pos[2] - last_pos[2]) / dt
                    )
                    track.velocities.append(velocity)

        # Update other attributes
        track.bbox_history.append(detection.bbox)
        track.confidence_history.append(detection.confidence)
        track.last_update_time = timestamp
        track.hits += 1
        track.misses = 0

        # Update track state
        if track.state == TrackState.TENTATIVE and track.hits >= self.min_hits:
            track.state = TrackState.CONFIRMED
        elif track.state == TrackState.LOST:
            track.state = TrackState.CONFIRMED

    def _create_new_track(self, detection: Detection, timestamp: float):
        """Create new track from detection"""
        track_id = f"track_{self.next_track_id}"
        self.next_track_id += 1

        track = Track(
            track_id=track_id,
            object_type=detection.object_type,
            state=TrackState.TENTATIVE,
            positions=deque(maxlen=30),
            velocities=deque(maxlen=30),
            bbox_history=deque(maxlen=30),
            confidence_history=deque(maxlen=30),
            created_time=timestamp,
            last_update_time=timestamp,
            hits=1,
            misses=0
        )

        # Initialize with detection
        if detection.position_3d:
            track.positions.append(detection.position_3d)
        track.bbox_history.append(detection.bbox)
        track.confidence_history.append(detection.confidence)

        self.tracks[track_id] = track

    def _manage_track_lifecycle(self, timestamp: float):
        """Manage track creation, confirmation, and deletion"""
        tracks_to_delete = []

        for track_id, track in self.tracks.items():
            age = timestamp - track.last_update_time

            if track.state == TrackState.CONFIRMED:
                if age > self.max_age:
                    track.state = TrackState.LOST
                    track.misses += 1
            elif track.state == TrackState.TENTATIVE:
                if age > self.max_age / 2:  # Shorter timeout for tentative tracks
                    tracks_to_delete.append(track_id)
            elif track.state == TrackState.LOST:
                if age > self.max_age * 2:  # Extended timeout for lost tracks
                    tracks_to_delete.append(track_id)

        # Delete old tracks
        for track_id in tracks_to_delete:
            del self.tracks[track_id]

class LaneDetector:
    """Lane detection and tracking module"""

    def __init__(self, lane_config: Dict):
        self.lane_config = lane_config
        self.confidence_threshold = lane_config.get('confidence_threshold', 0.6)
        self.lane_width_estimate = lane_config.get('lane_width', 3.7)  # Standard lane width

    async def detect_lanes(self, image: np.ndarray, timestamp: float) -> LaneInfo:
        """Detect lane markings in image"""
        try:
            # Placeholder for lane detection model
            # In production, use specialized lane detection models
            lane_points = await self._run_lane_detection_model(image)

            # Process lane points into lane markings
            left_markings, right_markings = self._process_lane_points(lane_points)

            # Calculate lane center line
            center_line = self._calculate_center_line(left_markings, right_markings)

            # Calculate lane curvature
            curvature = self._calculate_curvature(center_line)

            # Calculate overall confidence
            confidence = self._calculate_lane_confidence(left_markings, right_markings)

            return LaneInfo(
                left_markings=left_markings,
                right_markings=right_markings,
                center_line=center_line,
                lane_width=self.lane_width_estimate,
                curvature=curvature,
                confidence=confidence
            )

        except Exception as e:
            logging.error(f"Lane detection failed: {e}")
            return LaneInfo(
                left_markings=[],
                right_markings=[],
                center_line=None,
                lane_width=0.0,
                curvature=0.0,
                confidence=0.0
            )

    async def _run_lane_detection_model(self, image: np.ndarray) -> Dict:
        """Run lane detection model (placeholder)"""
        await asyncio.sleep(0.015)  # Simulate processing time

        height, width = image.shape[:2]

        # Generate synthetic lane points for demonstration
        lane_points = {
            'left_lane': [
                (width * 0.2, height),
                (width * 0.3, height * 0.7),
                (width * 0.4, height * 0.4),
                (width * 0.45, height * 0.1)
            ],
            'right_lane': [
                (width * 0.8, height),
                (width * 0.7, height * 0.7),
                (width * 0.6, height * 0.4),
                (width * 0.55, height * 0.1)
            ]
        }

        return lane_points

    def _process_lane_points(self, lane_points: Dict) -> Tuple[List[LaneMarking], List[LaneMarking]]:
        """Process detected lane points into lane markings"""
        left_markings = []
        right_markings = []

        # Left lane
        if 'left_lane' in lane_points and lane_points['left_lane']:
            left_marking = LaneMarking(
                lane_id="left_main",
                points=lane_points['left_lane'],
                lane_type=LaneType.DASHED,  # Default assumption
                confidence=0.8,
                width=0.1,  # 10cm typical marking width
                color="white"
            )
            left_markings.append(left_marking)

        # Right lane
        if 'right_lane' in lane_points and lane_points['right_lane']:
            right_marking = LaneMarking(
                lane_id="right_main",
                points=lane_points['right_lane'],
                lane_type=LaneType.SOLID,  # Default assumption
                confidence=0.8,
                width=0.1,
                color="white"
            )
            right_markings.append(right_marking)

        return left_markings, right_markings

    def _calculate_center_line(self, left_markings: List[LaneMarking],
                              right_markings: List[LaneMarking]) -> Optional[List[Tuple[float, float]]]:
        """Calculate lane center line from left and right markings"""
        if not left_markings or not right_markings:
            return None

        left_points = left_markings[0].points
        right_points = right_markings[0].points

        if len(left_points) != len(right_points):
            return None

        center_points = []
        for left_pt, right_pt in zip(left_points, right_points):
            center_x = (left_pt[0] + right_pt[0]) / 2
            center_y = (left_pt[1] + right_pt[1]) / 2
            center_points.append((center_x, center_y))

        return center_points

    def _calculate_curvature(self, center_line: Optional[List[Tuple[float, float]]]) -> float:
        """Calculate lane curvature"""
        if not center_line or len(center_line) < 3:
            return 0.0

        # Simplified curvature calculation
        # In production, use polynomial fitting
        total_angle_change = 0.0

        for i in range(1, len(center_line) - 1):
            p1 = center_line[i - 1]
            p2 = center_line[i]
            p3 = center_line[i + 1]

            # Calculate angle between consecutive segments
            v1 = (p2[0] - p1[0], p2[1] - p1[1])
            v2 = (p3[0] - p2[0], p3[1] - p2[1])

            dot_product = v1[0] * v2[0] + v1[1] * v2[1]
            magnitude1 = math.sqrt(v1[0]**2 + v1[1]**2)
            magnitude2 = math.sqrt(v2[0]**2 + v2[1]**2)

            if magnitude1 > 0 and magnitude2 > 0:
                cos_angle = dot_product / (magnitude1 * magnitude2)
                cos_angle = max(-1, min(1, cos_angle))  # Clamp to valid range
                angle = math.acos(cos_angle)
                total_angle_change += angle

        return total_angle_change / max(1, len(center_line) - 2)

    def _calculate_lane_confidence(self, left_markings: List[LaneMarking],
                                  right_markings: List[LaneMarking]) -> float:
        """Calculate overall lane detection confidence"""
        confidences = []

        for marking in left_markings + right_markings:
            confidences.append(marking.confidence)

        if not confidences:
            return 0.0

        return sum(confidences) / len(confidences)

class TrafficSignDetector:
    """Traffic sign detection and recognition module"""

    def __init__(self, sign_config: Dict):
        self.sign_config = sign_config
        self.confidence_threshold = sign_config.get('confidence_threshold', 0.7)
        self.sign_classes = self._load_sign_classes()

    def _load_sign_classes(self) -> Dict[int, str]:
        """Load traffic sign class mappings"""
        return {
            0: "stop",
            1: "yield",
            2: "speed_limit",
            3: "no_entry",
            4: "warning",
            5: "mandatory",
            6: "information"
        }

    async def detect_traffic_signs(self, image: np.ndarray, timestamp: float) -> List[TrafficSign]:
        """Detect traffic signs in image"""
        try:
            # Placeholder for traffic sign detection
            raw_detections = await self._run_sign_detection_model(image)

            # Process detections
            traffic_signs = []
            for detection in raw_detections:
                if detection['confidence'] >= self.confidence_threshold:
                    sign = self._create_traffic_sign(detection, timestamp)
                    traffic_signs.append(sign)

            return traffic_signs

        except Exception as e:
            logging.error(f"Traffic sign detection failed: {e}")
            return []

    async def _run_sign_detection_model(self, image: np.ndarray) -> List[Dict]:
        """Run traffic sign detection model (placeholder)"""
        await asyncio.sleep(0.01)  # Simulate processing time

        height, width = image.shape[:2]

        # Generate synthetic sign detections
        detections = []

        # Stop sign
        if np.random.random() > 0.7:  # 30% chance
            detections.append({
                'class_id': 0,
                'confidence': 0.9,
                'bbox': [width * 0.6, height * 0.2, width * 0.8, height * 0.4],
                'text': 'STOP'
            })

        # Speed limit sign
        if np.random.random() > 0.8:  # 20% chance
            detections.append({
                'class_id': 2,
                'confidence': 0.85,
                'bbox': [width * 0.1, height * 0.1, width * 0.3, height * 0.35],
                'text': '50',
                'speed_limit': 50
            })

        return detections

    def _create_traffic_sign(self, detection: Dict, timestamp: float) -> TrafficSign:
        """Create traffic sign object from detection"""
        bbox = BoundingBox(
            x_min=detection['bbox'][0],
            y_min=detection['bbox'][1],
            x_max=detection['bbox'][2],
            y_max=detection['bbox'][3],
            confidence=detection['confidence']
        )

        class_id = detection['class_id']
        sign_type = self.sign_classes.get(class_id, 'unknown')

        # Estimate 3D position
        position_3d = self._estimate_sign_position(bbox)

        # Estimate distance
        distance = math.sqrt(position_3d[0]**2 + position_3d[1]**2) if position_3d else None

        return TrafficSign(
            sign_id=str(uuid.uuid4()),
            sign_type=sign_type,
            bbox=bbox,
            position_3d=position_3d,
            confidence=detection['confidence'],
            text_content=detection.get('text'),
            speed_limit=detection.get('speed_limit'),
            distance_estimate=distance
        )

    def _estimate_sign_position(self, bbox: BoundingBox) -> Tuple[float, float, float]:
        """Estimate 3D position of traffic sign"""
        # Simplified estimation based on sign size and position
        center_x, center_y = bbox.center
        sign_height_pixels = bbox.height

        # Typical traffic sign dimensions
        typical_sign_height = 0.6  # 60cm typical sign height

        # Estimate distance
        focal_length = 800  # Placeholder
        distance = (typical_sign_height * focal_length) / max(sign_height_pixels, 1)

        # Convert to 3D coordinates
        x = distance
        y = (center_x - 960) / 960 * distance * 0.5
        z = 2.5  # Typical sign mounting height

        return (x, y, z)

class PerceptionEngine:
    """Main perception engine coordinating all perception modules"""

    def __init__(self, perception_config: Dict):
        self.perception_config = perception_config

        # Initialize modules
        self.object_detector = ObjectDetector(perception_config.get('object_detection', {}))
        self.tracker = MultiObjectTracker(perception_config.get('tracking', {}))
        self.lane_detector = LaneDetector(perception_config.get('lane_detection', {}))
        self.sign_detector = TrafficSignDetector(perception_config.get('traffic_signs', {}))

        # Performance monitoring
        self.perception_metrics = {
            'frames_processed': 0,
            'average_objects_per_frame': 0.0,
            'tracking_success_rate': 0.0,
            'lane_detection_rate': 0.0,
            'processing_time_ms': 0.0
        }

    async def process_frame(self, image: np.ndarray, timestamp: float) -> Dict:
        """Process a single frame through all perception modules"""
        start_time = time.perf_counter()

        try:
            # Run all perception modules concurrently
            detection_task = self.object_detector.detect_objects(image, timestamp)
            lane_task = self.lane_detector.detect_lanes(image, timestamp)
            sign_task = self.sign_detector.detect_traffic_signs(image, timestamp)

            # Wait for all tasks to complete
            detections, lane_info, traffic_signs = await asyncio.gather(
                detection_task, lane_task, sign_task, return_exceptions=True
            )

            # Handle exceptions
            if isinstance(detections, Exception):
                logging.error(f"Object detection failed: {detections}")
                detections = []

            if isinstance(lane_info, Exception):
                logging.error(f"Lane detection failed: {lane_info}")
                lane_info = LaneInfo([], [], None, 0.0, 0.0, 0.0)

            if isinstance(traffic_signs, Exception):
                logging.error(f"Traffic sign detection failed: {traffic_signs}")
                traffic_signs = []

            # Update object tracking
            tracks = self.tracker.update(detections, timestamp)

            # Calculate processing time
            processing_time = (time.perf_counter() - start_time) * 1000

            # Update metrics
            self._update_perception_metrics(detections, tracks, lane_info, processing_time)

            # Create perception result
            result = {
                'timestamp': timestamp,
                'detections': detections,
                'tracks': tracks,
                'lane_info': lane_info,
                'traffic_signs': traffic_signs,
                'processing_time_ms': processing_time,
                'metrics': self.get_perception_metrics()
            }

            return result

        except Exception as e:
            logging.error(f"Perception processing failed: {e}")
            return {
                'timestamp': timestamp,
                'detections': [],
                'tracks': [],
                'lane_info': LaneInfo([], [], None, 0.0, 0.0, 0.0),
                'traffic_signs': [],
                'processing_time_ms': 0.0,
                'metrics': self.get_perception_metrics()
            }

    def _update_perception_metrics(self, detections: List[Detection], tracks: List[Track],
                                  lane_info: LaneInfo, processing_time: float):
        """Update perception performance metrics"""
        self.perception_metrics['frames_processed'] += 1
        count = self.perception_metrics['frames_processed']

        # Update averages
        self.perception_metrics['processing_time_ms'] = (
            (self.perception_metrics['processing_time_ms'] * (count - 1) + processing_time) / count
        )

        self.perception_metrics['average_objects_per_frame'] = (
            (self.perception_metrics['average_objects_per_frame'] * (count - 1) + len(detections)) / count
        )

        # Tracking success rate (tracks vs detections)
        if detections:
            tracking_rate = len(tracks) / len(detections)
            self.perception_metrics['tracking_success_rate'] = (
                (self.perception_metrics['tracking_success_rate'] * (count - 1) + tracking_rate) / count
            )

        # Lane detection rate
        lane_detected = 1.0 if lane_info.confidence > 0.5 else 0.0
        self.perception_metrics['lane_detection_rate'] = (
            (self.perception_metrics['lane_detection_rate'] * (count - 1) + lane_detected) / count
        )

    def get_perception_metrics(self) -> Dict:
        """Get current perception metrics"""
        return self.perception_metrics.copy()

    def get_active_tracks(self) -> List[Track]:
        """Get currently active tracks"""
        return [track for track in self.tracker.tracks.values()
               if track.state == TrackState.CONFIRMED]

    def reset_tracking(self):
        """Reset all tracking state"""
        self.tracker.tracks.clear()
        self.tracker.next_track_id = 1

# Example usage
if __name__ == "__main__":
    import asyncio

    # Example configuration
    perception_config = {
        'object_detection': {
            'confidence_threshold': 0.5,
            'nms_threshold': 0.4
        },
        'tracking': {
            'max_distance_threshold': 50.0,
            'min_hits': 3,
            'max_age': 30
        },
        'lane_detection': {
            'confidence_threshold': 0.6,
            'lane_width': 3.7
        },
        'traffic_signs': {
            'confidence_threshold': 0.7
        }
    }

    async def main():
        # Initialize perception engine
        perception = PerceptionEngine(perception_config)

        # Create test image
        test_image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

        # Process frame
        result = await perception.process_frame(test_image, time.time())

        print(f"Perception Results:")
        print(f"  Detections: {len(result['detections'])}")
        print(f"  Tracks: {len(result['tracks'])}")
        print(f"  Lane confidence: {result['lane_info'].confidence:.2f}")
        print(f"  Traffic signs: {len(result['traffic_signs'])}")
        print(f"  Processing time: {result['processing_time_ms']:.1f}ms")

        # Print metrics
        metrics = perception.get_perception_metrics()
        print(f"Perception metrics: {metrics}")

    # Run example
    asyncio.run(main())