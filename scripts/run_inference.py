#!/usr/bin/env python3
"""
Run the full SO-101 world model pipeline on a real or mock robot.

Example:
    # Mock mode (no hardware, no cloud WM needed):
    python scripts/run_inference.py --mock --idm_ckpt checkpoints/idm/best.pt

    # Real robot + real world model:
    python scripts/run_inference.py \
        --idm_ckpt checkpoints/idm/best.pt \
        --wm_lora checkpoints/wm/lora_final \
        --instruction "pick up the red block and place it in the box" \
        --num_candidates 3
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--idm_ckpt", type=Path, required=True)
    parser.add_argument("--wm_lora", type=Path, default=None)
    parser.add_argument("--instruction", type=str, default="pick up the object")
    parser.add_argument("--num_candidates", type=int, default=3)
    parser.add_argument("--num_frames", type=int, default=13)
    parser.add_argument("--wm_steps", type=int, default=50)
    parser.add_argument("--replan_every", type=int, default=6)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mock", action="store_true", help="Use mock robot + mock world model")
    # Robot connection
    parser.add_argument("--port", default="/dev/tty.usbmodem*")
    parser.add_argument("--cam", type=int, default=0)
    # Flywheel
    parser.add_argument("--contribute", action="store_true", help="Upload episode to HF flywheel")
    parser.add_argument("--hf_repo", default="so101-community/world-model-data")
    parser.add_argument("--contributor", default="anonymous")
    args = parser.parse_args()

    from wm.models.pipeline import SO101Pipeline
    from wm.models.world_model import MockWorldModel

    if args.mock:
        from wm.models.world_model import MockWorldModel
        from wm.models.idm import InverseDynamicsModel
        import torch

        print("Mock mode — no hardware or cloud model needed")
        idm = InverseDynamicsModel()
        if args.idm_ckpt.exists():
            state = torch.load(args.idm_ckpt, map_location="cpu")
            idm.load_state_dict(state.get("model", state))
            print(f"Loaded IDM from {args.idm_ckpt}")

        pipeline = SO101Pipeline(
            world_model=MockWorldModel(),
            idm=idm,
            device="cpu",
            num_candidates=1,
            use_clip_scoring=False,
        )

        # Mock execution
        mock_frame = Image.fromarray(np.random.randint(0, 255, (480, 480, 3), dtype=np.uint8))
        print(f"\nPlanning: '{args.instruction}'")
        result = pipeline.plan(
            mock_frame,
            args.instruction,
            num_frames=args.num_frames,
            wm_inference_steps=1,
        )
        print(f"Generated {len(result.video_frames)} frames")
        print(f"Extracted {len(result.actions)} actions, shape: {result.actions.shape}")
        print(f"First action (degrees): {result.actions[0]}")
        return

    # Real pipeline
    pipeline = SO101Pipeline.from_checkpoints(
        idm_ckpt=args.idm_ckpt,
        wm_lora=args.wm_lora,
        device=args.device,
        num_candidates=args.num_candidates,
    )

    # Set up flywheel if contributing
    flywheel = None
    if args.contribute:
        from wm.data.flywheel import DataFlywheel
        flywheel = DataFlywheel(
            hf_repo=args.hf_repo,
            contributor=args.contributor,
        )
        flywheel.start_episode(task=args.instruction)
        print(f"Flywheel active — successful episodes will be uploaded to {args.hf_repo}")

    # Connect to SO-101
    try:
        import cv2
        cap = cv2.VideoCapture(args.cam)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {args.cam}")
        print(f"Camera {args.cam} opened")
    except Exception as e:
        print(f"Camera error: {e}")
        return

    print(f"\nRunning: '{args.instruction}'")
    print(f"Candidates: {args.num_candidates}, Replan every: {args.replan_every} steps")

    step = 0
    total_reward = 0.0

    try:
        while step < args.max_steps:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            frame = Image.fromarray(frame_bgr[:, :, ::-1])

            result = pipeline.plan(
                frame, args.instruction,
                num_frames=args.num_frames,
                wm_inference_steps=args.wm_steps,
            )
            print(f"  Step {step}: goal_score={result.goal_score:.3f}, actions={len(result.actions)}")

            for action in result.actions:
                if step >= args.max_steps:
                    break
                # TODO: send action to SO-101 hardware via LeRobot
                # robot.send_action(action)
                if flywheel:
                    _, latest_frame_bgr = cap.read()
                    latest_frame = Image.fromarray(latest_frame_bgr[:, :, ::-1])
                    flywheel.record_step(latest_frame, action)
                step += 1

            if step % args.replan_every == 0:
                continue  # replan with fresh frame

    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        cap.release()

    if flywheel:
        result_info = flywheel.submit_episode()
        print(f"\nFlywheel result: {result_info}")


if __name__ == "__main__":
    main()
