"""Video tracking service with persistent IDs and people counting."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict

import cv2
import numpy as np
from loguru import logger

from src.core.detector import Detector
from src.services.inference_service import standardize_predictions
from src.utils.video_utils import (
    build_output_video_path,
    create_video_writer,
    read_video_metadata,
    validate_video_path,
)


@dataclass
class ActiveTrack:
    """Mutable tracking state for one tracked object."""

    track_id: int
    class_name: str
    confidence: float
    bbox: list[float]
    history: list[list[float]] = field(default_factory=list)
    frames_seen: int = 0
    missed_frames: int = 0


class TrackResult(TypedDict):
    """Structured tracking output for one tracked object in one frame."""

    track_id: int
    class_name: str
    confidence: float
    bbox: list[float]
    history: list[list[float]]


class FrameTrackingResult(TypedDict):
    """Structured tracking output for one processed frame."""

    frame_index: int
    tracks: list[TrackResult]


class TrackSummary(TypedDict):
    """Summary output for offline tracking."""

    success: bool
    input_video: str
    output_video: str
    frames_processed: int
    fps: float
    people_count: int
    total_tracks: int
    frame_results: list[FrameTrackingResult]


class TrackingService:
    """Service layer orchestration for video tracking workflows."""

    def __init__(
        self,
        detector: Detector | None = None,
        *,
        iou_threshold: float = 0.3,
        max_missing_frames: int = 5,
        max_history_points: int = 30,
    ) -> None:
        """Initialize the tracking service with detector and tracker settings."""
        self._detector = detector or Detector()
        self._iou_threshold = iou_threshold
        self._max_missing_frames = max_missing_frames
        self._max_history_points = max_history_points

    def track_video(
        self,
        video_path: str | Path,
        model_source: str | Path,
        outputs_dir: str | Path,
    ) -> TrackSummary:
        """Track objects in a video and write an annotated output video."""
        input_video_path = validate_video_path(video_path)
        output_video_path = build_output_video_path(input_video_path, outputs_dir)

        capture = cv2.VideoCapture(str(input_video_path))
        if not capture.isOpened():
            message = f"Unable to open video: {input_video_path}"
            logger.error(message)
            raise ValueError(message)

        active_tracks: list[ActiveTrack] = []
        frame_results: list[FrameTrackingResult] = []
        seen_person_track_ids: set[int] = set()
        next_track_id = 1

        try:
            metadata = read_video_metadata(capture)
            writer = create_video_writer(output_video_path, metadata)
            if not writer.isOpened():
                message = f"Unable to create output video: {output_video_path}"
                logger.error(message)
                raise ValueError(message)

            try:
                frame_index = 0

                while True:
                    success, frame = capture.read()
                    if not success:
                        break

                    raw_results = self._detector.predict_frame(model_source=model_source, frame=frame, verbose=False)
                    detections = standardize_predictions(raw_results)
                    tracked_objects, next_track_id = self._update_tracks(
                        active_tracks=active_tracks,
                        detections=detections,
                        next_track_id=next_track_id,
                    )

                    for tracked_object in tracked_objects:
                        if tracked_object["class_name"] == "person":
                            seen_person_track_ids.add(tracked_object["track_id"])

                    annotated_frame = self._annotate_frame(frame=frame, tracked_objects=tracked_objects)
                    writer.write(annotated_frame)
                    frame_results.append(
                        {
                            "frame_index": frame_index,
                            "tracks": tracked_objects,
                        }
                    )
                    frame_index += 1
            finally:
                writer.release()

            logger.info("Tracked {} frames into {}", frame_index, output_video_path)
            return {
                "success": True,
                "input_video": str(input_video_path),
                "output_video": str(output_video_path),
                "frames_processed": frame_index,
                "fps": metadata["fps"],
                "people_count": len(seen_person_track_ids),
                "total_tracks": next_track_id - 1,
                "frame_results": frame_results,
            }
        finally:
            capture.release()

    def _update_tracks(
        self,
        *,
        active_tracks: list[ActiveTrack],
        detections: list[dict[str, object]],
        next_track_id: int,
    ) -> tuple[list[TrackResult], int]:
        """Match detections to active tracks and update persistent state."""
        matched_track_ids: set[int] = set()
        frame_tracks: list[TrackResult] = []

        for detection in detections:
            matched_track = self._find_best_track(active_tracks=active_tracks, detection=detection, matched_track_ids=matched_track_ids)
            bbox = [float(value) for value in detection["bbox"]]
            class_name = str(detection["class"])
            confidence = float(detection["confidence"])
            center = self._compute_center(bbox)

            if matched_track is None:
                matched_track = ActiveTrack(
                    track_id=next_track_id,
                    class_name=class_name,
                    confidence=confidence,
                    bbox=bbox,
                )
                active_tracks.append(matched_track)
                next_track_id += 1

            matched_track.class_name = class_name
            matched_track.confidence = confidence
            matched_track.bbox = bbox
            matched_track.frames_seen += 1
            matched_track.missed_frames = 0
            matched_track.history.append(center)
            matched_track.history = matched_track.history[-self._max_history_points :]
            matched_track_ids.add(matched_track.track_id)

            frame_tracks.append(
                {
                    "track_id": matched_track.track_id,
                    "class_name": matched_track.class_name,
                    "confidence": round(matched_track.confidence, 4),
                    "bbox": [round(value, 2) for value in matched_track.bbox],
                    "history": [[round(value, 2) for value in point] for point in matched_track.history],
                }
            )

        self._age_unmatched_tracks(active_tracks=active_tracks, matched_track_ids=matched_track_ids)
        return frame_tracks, next_track_id

    def _find_best_track(
        self,
        *,
        active_tracks: list[ActiveTrack],
        detection: dict[str, object],
        matched_track_ids: set[int],
    ) -> ActiveTrack | None:
        """Find the best unmatched active track for one detection."""
        best_track: ActiveTrack | None = None
        best_iou = 0.0
        detection_bbox = [float(value) for value in detection["bbox"]]
        detection_class = str(detection["class"])

        for track in active_tracks:
            if track.track_id in matched_track_ids:
                continue
            if track.class_name != detection_class:
                continue

            iou = self._compute_iou(track.bbox, detection_bbox)
            if iou >= self._iou_threshold and iou > best_iou:
                best_iou = iou
                best_track = track

        return best_track

    def _age_unmatched_tracks(self, *, active_tracks: list[ActiveTrack], matched_track_ids: set[int]) -> None:
        """Age unmatched tracks and remove stale ones."""
        stale_track_ids: set[int] = set()

        for track in active_tracks:
            if track.track_id not in matched_track_ids:
                track.missed_frames += 1
                if track.missed_frames > self._max_missing_frames:
                    stale_track_ids.add(track.track_id)

        active_tracks[:] = [track for track in active_tracks if track.track_id not in stale_track_ids]

    @staticmethod
    def _compute_iou(first_bbox: list[float], second_bbox: list[float]) -> float:
        """Compute IoU between two bounding boxes in xyxy format."""
        x_left = max(first_bbox[0], second_bbox[0])
        y_top = max(first_bbox[1], second_bbox[1])
        x_right = min(first_bbox[2], second_bbox[2])
        y_bottom = min(first_bbox[3], second_bbox[3])

        if x_right <= x_left or y_bottom <= y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        first_area = max(first_bbox[2] - first_bbox[0], 0.0) * max(first_bbox[3] - first_bbox[1], 0.0)
        second_area = max(second_bbox[2] - second_bbox[0], 0.0) * max(second_bbox[3] - second_bbox[1], 0.0)
        union_area = first_area + second_area - intersection_area

        if union_area <= 0:
            return 0.0

        return intersection_area / union_area

    @staticmethod
    def _compute_center(bbox: list[float]) -> list[float]:
        """Compute the center point of a bounding box."""
        return [
            (bbox[0] + bbox[2]) / 2.0,
            (bbox[1] + bbox[3]) / 2.0,
        ]

    @staticmethod
    def _annotate_frame(frame: np.ndarray, tracked_objects: list[TrackResult]) -> np.ndarray:
        """Draw tracking results, IDs, and history trails on a frame."""
        annotated_frame = frame.copy()

        for tracked_object in tracked_objects:
            bbox = tracked_object["bbox"]
            x1, y1, x2, y2 = [int(value) for value in bbox]
            label = f"ID {tracked_object['track_id']} {tracked_object['class_name']} {tracked_object['confidence']:.2f}"
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (30, 144, 255), 2)
            cv2.putText(
                annotated_frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (30, 144, 255),
                2,
                cv2.LINE_AA,
            )

            history = tracked_object["history"]
            for index in range(1, len(history)):
                start = tuple(int(value) for value in history[index - 1])
                end = tuple(int(value) for value in history[index])
                cv2.line(annotated_frame, start, end, (0, 215, 255), 2, cv2.LINE_AA)

        return annotated_frame
