# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

# /// script
# requires-python = ">=3.12"
# dependencies = ["lap", "muna", "torchvision", "ultralytics"]
# ///

from argparse import Namespace
from muna import compile, Parameter, Sandbox
from numpy import array, float32
from pydantic import BaseModel, Field
from torch import from_numpy
from torchvision.ops import box_convert
from typing import Annotated
from ultralytics.trackers.byte_tracker import BYTETracker

class Detection(BaseModel):
    x_min: float = Field(description="Normalized minimum X coordinate.")
    y_min: float = Field(description="Normalized minimum Y coordinate.")
    x_max: float = Field(description="Normalized maximum X coordinate.")
    y_max: float = Field(description="Normalized maximum Y coordinate.")
    label: str = Field(description="Detection label.")
    confidence: float = Field(description="Detection confidence score.")

class TrackedDetection(Detection):
    track_id: int = Field(description="Unique track identifier that persists across frames.")

# Create tracker
tracker_args = Namespace(
    track_high_thresh=0.5,
    track_low_thresh=0.1,
    new_track_thresh=0.5,
    track_buffer=30,
    match_thresh=0.8,
    fuse_score=False
)
tracker = BYTETracker(tracker_args, frame_rate=30)
label_to_id = { "": -1 }
id_to_label = { -1: "" }

@compile(
    tag="@yusuf/bytetrack",
    description="Track multiple objects using the ByteTrack algorithm.",
    sandbox=Sandbox()
        .pip_install("torchvision", index_url="https://download.pytorch.org/whl/cpu")
        .pip_install("lap", "ultralytics")
        .pip_install("opencv-python-headless")
)
def track_objects(
    detections: Annotated[
        list[Detection],
        Parameter.BoundingBoxes(description="Detections for the current frame.")
    ],
    *,
    reset: Annotated[
        bool,
        Parameter.Generic(description="Whether to reset the tracker state before tracking.")
    ]=False
) -> Annotated[
    list[TrackedDetection],
    Parameter.BoundingBoxes(description="Tracked objects for the current frame with persistent track IDs.")
]:
    """
    Track multiple objects using the ByteTrack algorithm.
    
    ByteTrack associates detections across video frames to produce tracked objects 
    with persistent IDs. It uses a two-stage matching strategy: first matching 
    high-confidence detections, then matching remaining low-confidence detections 
    to improve tracking continuity.
    """
    # Reset
    if reset:
        tracker.reset()
        label_to_id.clear()
        id_to_label.clear()
    # Build label mappings from detections
    for det in detections:
        if det.label not in label_to_id:
            label_id = len(label_to_id)
            label_to_id[det.label] = label_id
            id_to_label[label_id] = det.label
    # Run tracker
    xyxy = array([[d.x_min, d.y_min, d.x_max, d.y_max] for d in detections], dtype=float32)
    cxcywh = box_convert(from_numpy(xyxy), in_fmt="xyxy", out_fmt="cxcywh").numpy()
    conf = array([d.confidence for d in detections], dtype=float32)
    cls = array([label_to_id.get(d.label, 0) for d in detections], dtype=float32)
    state = Namespace(xywh=cxcywh, conf=conf, cls=cls)
    tracked = tracker.update(state)
    # Create tracked detections
    tracked_detections = [TrackedDetection(
        x_min=row[0].item(),
        y_min=row[1].item(),
        x_max=row[2].item(),
        y_max=row[3].item(),
        label=id_to_label.get(int(row[6]), "unknown"),
        confidence=row[5].item(),
        track_id=int(row[4].item())
    ) for row in tracked]
    # Return
    return tracked_detections

if __name__ == "__main__":
    # Simulate a simple tracking scenario with 3 frames
    # Frame 1: Two people detected
    frame1_detections = [
        Detection(x_min=0.1, y_min=0.2, x_max=0.2, y_max=0.5, label="person", confidence=0.9),
        Detection(x_min=0.5, y_min=0.3, x_max=0.6, y_max=0.6, label="person", confidence=0.85),
    ]
    # Frame 2: Same two people, slightly moved
    frame2_detections = [
        Detection(x_min=0.52, y_min=0.32, x_max=0.62, y_max=0.62, label="person", confidence=0.87),
        Detection(x_min=0.12, y_min=0.22, x_max=0.22, y_max=0.52, label="person", confidence=0.88),
    ]
    # Frame 3: One person left, one new person appeared
    frame3_detections = [
        Detection(x_min=0.14, y_min=0.24, x_max=0.24, y_max=0.54, label="person", confidence=0.91),
        Detection(x_min=0.7, y_min=0.1, x_max=0.8, y_max=0.4, label="person", confidence=0.82),
    ]
    # Process each frame
    print("Frame 1:")
    tracks1 = track_objects(frame1_detections)
    for t in tracks1:
        print(f"  Track {t.track_id}: {t.label} @ ({t.x_min:.2f}, {t.y_min:.2f}) conf={t.confidence:.2f}")
    print("\nFrame 2:")
    tracks2 = track_objects(frame2_detections)
    for t in tracks2:
        print(f"  Track {t.track_id}: {t.label} @ ({t.x_min:.2f}, {t.y_min:.2f}) conf={t.confidence:.2f}")
    print("\nFrame 3:")
    tracks3 = track_objects(frame3_detections)
    for t in tracks3:
        print(f"  Track {t.track_id}: {t.label} @ ({t.x_min:.2f}, {t.y_min:.2f}) conf={t.confidence:.2f}")
    # Verify tracking consistency
    print("\n--- Verification ---")
    print(f"Frame 1 track IDs: {[t.track_id for t in tracks1]}")
    print(f"Frame 2 track IDs: {[t.track_id for t in tracks2]}")
    print(f"Frame 3 track IDs: {[t.track_id for t in tracks3]}")