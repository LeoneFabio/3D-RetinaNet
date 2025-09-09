"""
Visualize tubes on frames.
"""

import os
import argparse
import pickle
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.cm as cmx
import matplotlib.colors as colors
import hashlib


MIN_SIZE = 512
MAX_SIZE = int(MIN_SIZE * 1.35)  # 512 * 1.35 ≈ 691

# --- Label mappings ---
LABEL_TYPES = ['agent', 'action', 'loc', 'duplex', 'triplet']

LABEL_MAP = {
    'agent': ['Ped', 'Car', 'Cyc', 'Mobike', 'MedVeh', 'LarVeh', 'Bus', 'EmVeh', 'TL', 'OthTL'],
    'action': ['Red', 'Amber', 'Green', 'MovAway', 'MovTow', 'Mov', 'Brake', 'Stop', 'IncatLft', 'IncatRht', 'HazLit',
               'TurLft', 'TurRht', 'Ovtak', 'Wait2X', 'XingFmLft', 'XingFmRht', 'Xing', 'PushObj'],
    'loc': ['VehLane', 'OutgoLane', 'OutgoCycLane', 'IncomLane', 'IncomCycLane', 'Pav', 'LftPav', 'RhtPav', 'Jun', 'xing',
            'BusStop', 'parking'],
    'duplex': ['Bus-MovAway', 'Bus-MovTow', 'Bus-Stop', 'Bus-XingFmLft', 'Car-Brake', 'Car-IncatLft', 'Car-IncatRht',
               'Car-MovAway', 'Car-MovTow', 'Car-Stop', 'Car-TurLft', 'Car-TurRht', 'Car-XingFmLft', 'Car-XingFmRht',
               'Cyc-MovAway', 'Cyc-MovTow', 'Cyc-Stop', 'Cyc-TurLft', 'Cyc-XingFmLft', 'Cyc-XingFmRht', 'LarVeh-Stop',
               'MedVeh-IncatLft', 'MedVeh-MovTow', 'MedVeh-Stop', 'MedVeh-TurRht', 'OthTL-Green', 'OthTL-Red', 'Ped-Mov',
               'Ped-MovAway', 'Ped-MovTow', 'Ped-PushObj', 'Ped-Stop', 'Ped-Wait2X', 'Ped-Xing', 'Ped-XingFmLft',
               'Ped-XingFmRht', 'TL-Amber', 'TL-Green', 'TL-Red'],
    'triplet': ['Bus-MovTow-IncomLane', 'Bus-MovTow-Jun', 'Bus-Stop-IncomLane', 'Bus-Stop-VehLane', 'Bus-XingFmLft-Jun',
                'Car-Brake-Jun', 'Car-Brake-VehLane', 'Car-IncatLft-Jun', 'Car-IncatLft-VehLane', 'Car-IncatRht-IncomLane',
                'Car-IncatRht-Jun', 'Car-MovAway-Jun', 'Car-MovAway-OutgoLane', 'Car-MovAway-VehLane', 'Car-MovTow-IncomLane',
                'Car-MovTow-Jun', 'Car-Stop-IncomLane', 'Car-Stop-Jun', 'Car-Stop-VehLane', 'Car-TurLft-Jun',
                'Car-TurLft-VehLane', 'Car-TurRht-IncomLane', 'Car-TurRht-Jun', 'Car-XingFmLft-Jun', 'Cyc-MovAway-Jun',
                'Cyc-MovAway-LftPav', 'Cyc-MovAway-OutgoCycLane', 'Cyc-MovAway-OutgoLane', 'Cyc-MovAway-VehLane',
                'Cyc-MovTow-IncomCycLane', 'Cyc-MovTow-IncomLane', 'Cyc-MovTow-Jun', 'Cyc-MovTow-LftPav',
                'Cyc-Stop-IncomCycLane', 'Cyc-Stop-IncomLane', 'Cyc-Stop-Jun', 'Cyc-TurLft-Jun', 'Cyc-XingFmLft-Jun',
                'MedVeh-MovTow-IncomLane', 'MedVeh-MovTow-Jun', 'MedVeh-Stop-IncomLane', 'MedVeh-Stop-Jun',
                'MedVeh-TurRht-Jun', 'Ped-Mov-Pav', 'Ped-MovAway-LftPav', 'Ped-MovAway-Pav', 'Ped-MovAway-RhtPav',
                'Ped-MovTow-IncomLane', 'Ped-MovTow-LftPav', 'Ped-MovTow-RhtPav', 'Ped-MovTow-VehLane', 'Ped-PushObj-LftPav',
                'Ped-PushObj-RhtPav', 'Ped-Stop-BusStop', 'Ped-Stop-LftPav', 'Ped-Stop-Pav', 'Ped-Stop-RhtPav',
                'Ped-Stop-VehLane', 'Ped-Wait2X-LftPav', 'Ped-Wait2X-RhtPav', 'Ped-XingFmLft-IncomLane', 'Ped-XingFmLft-Jun',
                'Ped-XingFmLft-VehLane', 'Ped-XingFmLft-xing', 'Ped-XingFmRht-IncomLane', 'Ped-XingFmRht-Jun',
                'Ped-XingFmRht-RhtPav', 'Ped-XingFmRht-VehLane']
}

# --- helpers for color ---
def deterministic_color(key, cmap_name='tab20'):
    """
    Generate an RGB color tuple (0-255) deterministically from key (string or tuple).
    """
    if not isinstance(key, str):
        key = str(key)
    # hash to integer
    digest = hashlib.md5(key.encode('utf8')).hexdigest()
    h = int(digest[:8], 16)
    cmap = cmx.get_cmap(cmap_name)
    # map hash to [0,1)
    v = (h % 1000) / 1000.0
    rgba = cmap(v)
    rgb = tuple(int(255 * c) for c in rgba[:3])
    return rgb

# --- label overlap avoidance utility ---
def rects_intersect(r1, r2):
    # r = (x1,y1,x2,y2)
    return not (r2[0] >= r1[2] or r2[2] <= r1[0] or r2[1] >= r1[3] or r2[3] <= r1[1])

def text_bbox(x, y, w, h):
    return (x, y, x + w, y + h)

# --- main visualization function ---
def visualize_tubes_on_frames(
    tube_file,
    videoname,
    frames_root,
    out_root,
    label_map=LABEL_MAP,
    font_path=None,
    top_k=3,  # number of top concepts to show per box
    threshold=1.35  # threshold for score summation
):
    with open(tube_file, 'rb') as f:
        tubes_data = pickle.load(f)

    detection_tubes = tubes_data
    frames_dir = os.path.join(frames_root, videoname)
    if not os.path.isdir(frames_dir):
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")

    save_dir = os.path.join(out_root, videoname)
    os.makedirs(save_dir, exist_ok=True)

    try:
        font = ImageFont.truetype(font_path, size=14) if font_path and os.path.isfile(font_path) else ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    # --- collect all tubes ---
    all_tubes = []
    for ltype, videomap in detection_tubes.items():
        if isinstance(videomap, dict) and videoname in videomap:
            for t in videomap[videoname]:
                t['_label_type'] = ltype
                all_tubes.append(t)

    if not all_tubes and videoname in detection_tubes:
        for t in detection_tubes[videoname]:
            t['_label_type'] = t.get('label_type', 'agent')
            all_tubes.append(t)

    if not all_tubes:
        raise ValueError(f"No tubes found for video {videoname} in {tube_file}")

    # --- build frame dict ---
    frames_dict = {}
    for tube in all_tubes:
        label_type = tube['_label_type']
        label_id = tube.get('label_id', None)
        frames_arr = tube.get('frames')
        boxes_arr = tube.get('boxes')
        score = tube.get('score', None)
        if frames_arr is None or boxes_arr is None:
            continue

        label_name = label_map.get(label_type, [f"{label_type}:{label_id}"])[int(label_id)] if (
            label_id is not None and label_type in label_map and 0 <= int(label_id) < len(label_map[label_type])
        ) else f"{label_type}:{label_id}"

        tube_color = deterministic_color((label_type, label_id))

        for idx_in_tube, frame_num in enumerate(frames_arr):
            box = boxes_arr[idx_in_tube]
            fnum = int(frame_num)

            ann = {
                'box': [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                'label': label_name,
                'label_type': label_type,
                'label_id': int(label_id) if label_id is not None else None,
                'tube_score': float(score) if score is not None else None,
                'frame_score': float(tube.get('scores', np.zeros(len(frames_arr)))[idx_in_tube])
                if tube.get('scores') is not None else None
            }
            frames_dict.setdefault(fnum, []).append((ann, tube_color))

    frame_ids = sorted(frames_dict.keys())

    for fnum in frame_ids:
        img_path = os.path.join(frames_dir, f"{fnum:05d}.jpg")
        if not os.path.isfile(img_path):
            print(f"Warning: missing frame image {img_path}; skipping")
            continue

        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        img_w, img_h = img.size
        occupied = []

        ann_list = frames_dict.get(fnum, [])
        # --- group by box to find top K concepts ---
        box_groups = {}
        for ann, color in ann_list:
            key = tuple(round(v, 1) for v in ann['box'])  # small rounding to avoid float diff
            box_groups.setdefault(key, {'anns': [], 'color': color})
            box_groups[key]['anns'].append(ann)

        for box_key, data in box_groups.items():
            anns_sorted = sorted(
                data['anns'],
                key=lambda a: -(a.get('frame_score') or a.get('tube_score') or 0)
            )

            # --- thresholding with topk+1 ---
            topk_plus_1 = anns_sorted[:(top_k+1)]
            score_sum = sum([(a.get('frame_score') or a.get('tube_score') or 0) for a in topk_plus_1])

            if score_sum <= threshold:
                continue  # skip this box entirely

            # keep only topk for visualization
            anns = anns_sorted[:top_k]
            color = data['color']
            
            # --- RESCALE BOXES TO ORIGINAL IMAGE SIZE ---
            x1, y1, x2, y2 = box_key
            scale_x = img_w / MAX_SIZE   # img_w = 1280
            scale_y = img_h / MIN_SIZE   # img_h = 960
            x1 *= scale_x
            x2 *= scale_x
            y1 *= scale_y
            y2 *= scale_y

            # draw box
            stroke = 3
            for s in range(stroke):
                rect = [x1 - s, y1 - s, x2 + s, y2 + s]
                draw.rectangle(rect, outline=color)

            # stack concepts vertically
            ty = max(0, y1 - 20 * len(anns))
            for ann in anns:
                label_text = f"{ann['label']} {ann['frame_score']:.2f}" if ann.get('frame_score') is not None else ann['label']
                bbox = draw.textbbox((0, 0), label_text, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
                pad = 3

                if ty < 0:
                    ty = y1 + 2

                bg_rect = [x1 - pad, ty - pad, x1 + text_w + pad, ty + text_h + pad]
                draw.rectangle(bg_rect, fill=color)  # <--- same color as box
                draw.text((x1, ty), label_text, fill=(255, 255, 255), font=font)  # white text
                occupied.append(bg_rect)
                ty += text_h + 2 * pad + 2

        out_path = os.path.join(save_dir, f"{fnum:05d}.jpg")
        img.save(out_path)

    print(f"Visualization complete. Saved to {save_dir}")



# --- CLI ---
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--tube_file', required=True, help='Path to tubes pickle file (e.g. tubes_none_0.pkl)')
    ap.add_argument('--videoname', required=True, help='Video folder name inside frames root')
    ap.add_argument('--frames_root', default='./comma/rgb-images', help='Root folder containing video folders')
    ap.add_argument('--out_root', default='./tubes_visualizations', help='Output folder')
    ap.add_argument('--font', default=None, help='Optional path to TTF font')
    args = ap.parse_args()
    
    visualize_tubes_on_frames(args.tube_file, args.videoname, args.frames_root, args.out_root, font_path=args.font)