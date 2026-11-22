"""
Convert .npz blendshape files (51 ARKit camelCase) to Furhat .txt format (49 UPPER_SNAKE + 3 neck).

Usage:
    python npz_to_furhat_txt.py input.npz output.txt --fps 30
"""

import argparse
import numpy as np

# Mapping from your 51 ARKit blendshapes to the 49 Furhat expects.
# Furhat's txt is missing browDownLeft and browDownRight (indices 0, 1 in your list).
# The rest map 1:1 but with UPPER_SNAKE_CASE naming.

ARKIT_51 = [
    'browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight',
    'cheekPuff', 'cheekSquintLeft', 'cheekSquintRight',
    'eyeBlinkLeft', 'eyeBlinkRight', 'eyeLookDownLeft', 'eyeLookDownRight',
    'eyeLookInLeft', 'eyeLookInRight', 'eyeLookOutLeft', 'eyeLookOutRight',
    'eyeLookUpLeft', 'eyeLookUpRight', 'eyeSquintLeft', 'eyeSquintRight',
    'eyeWideLeft', 'eyeWideRight',
    'jawForward', 'jawLeft', 'jawOpen', 'jawRight',
    'mouthClose', 'mouthDimpleLeft', 'mouthDimpleRight',
    'mouthFrownLeft', 'mouthFrownRight', 'mouthFunnel',
    'mouthLeft', 'mouthLowerDownLeft', 'mouthLowerDownRight',
    'mouthPressLeft', 'mouthPressRight', 'mouthPucker',
    'mouthRight', 'mouthRollLower', 'mouthRollUpper',
    'mouthShrugLower', 'mouthShrugUpper', 'mouthSmileLeft', 'mouthSmileRight',
    'mouthStretchLeft', 'mouthStretchRight', 'mouthUpperUpLeft', 'mouthUpperUpRight',
    'noseSneerLeft', 'noseSneerRight'
]

# The 49 Furhat blendshape names (in order from your .txt header), excluding the 3 neck params
FURHAT_49 = [
    'BROW_INNER_UP', 'BROW_OUTER_UP_LEFT', 'BROW_OUTER_UP_RIGHT',
    'CHEEK_PUFF', 'CHEEK_SQUINT_LEFT', 'CHEEK_SQUINT_RIGHT',
    'EYE_BLINK_LEFT', 'EYE_BLINK_RIGHT', 'EYE_LOOK_DOWN_LEFT', 'EYE_LOOK_DOWN_RIGHT',
    'EYE_LOOK_IN_LEFT', 'EYE_LOOK_IN_RIGHT', 'EYE_LOOK_OUT_LEFT', 'EYE_LOOK_OUT_RIGHT',
    'EYE_LOOK_UP_LEFT', 'EYE_LOOK_UP_RIGHT', 'EYE_SQUINT_LEFT', 'EYE_SQUINT_RIGHT',
    'EYE_WIDE_LEFT', 'EYE_WIDE_RIGHT',
    'JAW_FORWARD', 'JAW_LEFT', 'JAW_OPEN', 'JAW_RIGHT',
    'MOUTH_CLOSE', 'MOUTH_DIMPLE_LEFT', 'MOUTH_DIMPLE_RIGHT',
    'MOUTH_FROWN_LEFT', 'MOUTH_FROWN_RIGHT', 'MOUTH_FUNNEL',
    'MOUTH_LEFT', 'MOUTH_LOWER_DOWN_LEFT', 'MOUTH_LOWER_DOWN_RIGHT',
    'MOUTH_PRESS_LEFT', 'MOUTH_PRESS_RIGHT', 'MOUTH_PUCKER',
    'MOUTH_RIGHT', 'MOUTH_ROLL_LOWER', 'MOUTH_ROLL_UPPER',
    'MOUTH_SHRUG_LOWER', 'MOUTH_SHRUG_UPPER', 'MOUTH_SMILE_LEFT', 'MOUTH_SMILE_RIGHT',
    'MOUTH_STRETCH_LEFT', 'MOUTH_STRETCH_RIGHT', 'MOUTH_UPPER_UP_LEFT', 'MOUTH_UPPER_UP_RIGHT',
    'NOSE_SNEER_LEFT', 'NOSE_SNEER_RIGHT'
]

NECK_NAMES = ['NECK_PAN', 'NECK_TILT', 'NECK_ROLL']

def camel_to_upper_snake(name):
    """browInnerUp -> BROW_INNER_UP"""
    import re
    return re.sub(r'(?<=[a-z])(?=[A-Z])', '_', name).upper()

def build_mapping(arkit_names, furhat_names):
    """
    Returns list of (furhat_index, arkit_index) pairs.
    Skips ARKit blendshapes that have no Furhat equivalent.
    """
    furhat_lookup = {name: i for i, name in enumerate(furhat_names)}
    mapping = []
    for arkit_idx, arkit_name in enumerate(arkit_names):
        snake = camel_to_upper_snake(arkit_name)
        if snake in furhat_lookup:
            mapping.append((furhat_lookup[snake], arkit_idx))
    return mapping

def convert(npz_path, txt_path, fps=30):
    data = np.load(npz_path)

    # Find the blendshape array
    for key in ['combined_features', 'expressions', 'arkit_blendshapes']:
        if key in data.files:
            blendshapes = data[key]
            break
    else:
        raise ValueError(f"No blendshape data found. Keys: {data.files}")

    num_frames, num_bs = blendshapes.shape
    print(f"Loaded {npz_path}: {num_frames} frames, {num_bs} blendshapes")

    mapping = build_mapping(ARKIT_51, FURHAT_49)
    print(f"Mapped {len(mapping)}/{len(FURHAT_49)} Furhat blendshapes")

    # Check which Furhat blendshapes are unmapped
    mapped_furhat = {fi for fi, _ in mapping}
    unmapped = [FURHAT_49[i] for i in range(len(FURHAT_49)) if i not in mapped_furhat]
    if unmapped:
        print(f"Unmapped (will be 0.0): {unmapped}")

    with open(txt_path, 'w') as f:
        # Header line
        header = FURHAT_49 + NECK_NAMES
        f.write(' '.join(header) + '\n')

        for frame_idx in range(num_frames):
            timestamp = frame_idx / fps

            # Build the 49 blendshape values
            values = [0.0] * len(FURHAT_49)
            for furhat_idx, arkit_idx in mapping:
                values[furhat_idx] = float(blendshapes[frame_idx, arkit_idx])

            # No head pose in .npz, so neck = 0
            neck = [0.0, 0.0, 0.0]

            # Format: timestamp val1 val2 ... val49 neck_pan neck_tilt neck_roll
            parts = [f'{timestamp:.3f}'] + [f'{v:.3f}' for v in values + neck]
            f.write(' '.join(parts) + '\n')

    print(f"Written {num_frames} frames to {txt_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_npz', help='Path to .npz blendshape file')
    parser.add_argument('--output_txt', help='Path to output .txt file')
    parser.add_argument('--fps', type=int, default=30, help='Frame rate (default: 30)')
    args = parser.parse_args()
    convert(args.input_npz, args.output_txt, args.fps)