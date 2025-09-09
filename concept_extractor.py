"""
Concept Extraction Module for 3D-RetinaNet to GridLock Integration

This module extract concept representations
that can replace CLIP in the GridLock model, while maintaining the same output format
as gen_dets for consistency and debugging.
"""

import os
import torch
import numpy as np
import pickle
import torch.utils.data as data_utils
from data import custom_collate_comma
from data import custum_collate
from modules import utils
from modules.box_utils import nms
from models.retinanet import build_retinanet

logger = utils.get_logger(__name__)

def extract_concepts_for_gridlock(args, net, val_dataset, output_dir):
    """
    Extract concept representations from 3D-RetinaNet for GridLock integration.
    Version with 240-frame videos with proper sequence reconstruction.
    
    Args:
        args: Argument namespace
        net: Trained 3D-RetinaNet model
        val_dataset: Dataset (e.g., Comma2k19)
        output_dir: Directory to save concept representations
    
    Returns:
        Saves detections + concept logits 
    """
    collate_fn = custom_collate_comma if args.DATASET == 'comma' else custum_collate

    net.eval()
    val_data_loader = data_utils.DataLoader(
        val_dataset, int(args.TEST_BATCH_SIZE), num_workers=args.NUM_WORKERS,
        shuffle=False, pin_memory=True, collate_fn=collate_fn
    )
    
    # Load trained model weights
    epoch = int(args.EVAL_EPOCHS[0])
    args.MODEL_PATH = args.SAVE_ROOT + 'model_{:06d}.pth'.format(epoch)
    net.load_state_dict(torch.load(args.MODEL_PATH))
    logger.info('Loaded model from %s' % args.MODEL_PATH)
    
    # Setup save directories
    concept_save_dir = os.path.join(output_dir, "concepts-{it:02d}-{sq:02d}/".format(
        it=epoch, sq=args.TEST_SEQ_LEN))
    os.makedirs(concept_save_dir, exist_ok=True)
    logger.info('Concept extraction saving dir: ' + concept_save_dir)

    batch_concepts_dir = os.path.join(output_dir, "batch_concepts-{it:02d}-{sq:02d}/".format(
        it=epoch, sq=args.TEST_SEQ_LEN))
    os.makedirs(batch_concepts_dir, exist_ok=True)
    logger.info('Batch concept tensors saving dir: ' + batch_concepts_dir)
    
    activation = torch.nn.Sigmoid().cuda()
    processed_videos = []

    # Dictionary to reconstruct full 240-frame videos
    video_reconstructions = {}  # {video_name: {frame_num: concept_logits}}
    
    with torch.no_grad():
        for val_itr, (images, gt_boxes, gt_targets, ego_labels, batch_counts, img_indexs, wh) in enumerate(val_data_loader):
            
            batch_size = images.size(0)
            images = images.cuda(0, non_blocking=True)
            
            # Forward pass - get raw outputs
            decoded_boxes, flat_conf, ego_preds = net(images)
            
            # Apply activation to get probabilities
            confidence = activation(flat_conf)  # [batch, seq_len, num_anchors, num_classes]
            ego_probs = activation(ego_preds)   # [batch, seq_len, num_ego_classes]
            
            seq_len = ego_probs.shape[1]
            num_concepts = confidence.shape[-1]
            effective_seq_len = seq_len - getattr(args, 'skip_ending', 0)

            # Process each sample in the batch
            for b in range(batch_size):
                index = img_indexs[b]
                annot_info = val_dataset.ids[index]
                
                if args.DATASET != 'ava':
                    video_id, frame_num, step_size = annot_info
                else:
                    video_id, frame_num, step_size, keyframe = annot_info
                    frame_num = keyframe - 1
                
                videoname = val_dataset.video_list[video_id]
                save_dir = '{:s}/{}'.format(concept_save_dir, videoname)
                
                # Initialize video reconstruction dictionary
                if videoname not in video_reconstructions:
                    video_reconstructions[videoname] = {
                        'concepts': {},  # {frame_number: concept_logits}
                        'video_id': video_id,
                        'frames_collected': set()
                    }
                
                # Track processed videos for gen_dets consistency
                if videoname not in processed_videos:
                    processed_videos.append(videoname)
                    

                if not os.path.isdir(save_dir):
                    os.makedirs(save_dir)
                
                # Process each frame in the sequence
                current_frame_num = frame_num

                for si in range(seq_len):

                    # Extract frame-level data
                    decoded_boxes_batch = decoded_boxes[b, si]
                    confidence_batch = confidence[b, si]  # [num_anchors, num_classes]
                    ego_frame = ego_probs[b, si, :].cpu().numpy()
                    
                    # Get detection scores and apply filtering
                    scores = confidence_batch[:, 0].squeeze().clone()
                    
                    # Filter detections and get concept logits
                    cls_dets, save_data, frame_concept_logits = filter_detections_with_concepts(
                        args, scores, decoded_boxes_batch, confidence_batch
                    )

                    # Store frame concept in reconstruction (1-indexed frame numbers)
                    frame_key = current_frame_num + 1
                    if frame_key <= 240:  # Only store valid frame numbers
                        video_reconstructions[videoname]['concepts'][frame_key] = frame_concept_logits.copy()
                        video_reconstructions[videoname]['frames_collected'].add(frame_key)
                    
                    # Save pickle files (following original gen_dets logic)
                    save_name = '{:s}/{:05d}.pkl'.format(save_dir, frame_key)
                    
                    # Create save data
                    complete_save_data = {
                        'ego': ego_frame,
                        'main': save_data,
                        'video_name': videoname,
                        'concepts': val_dataset.concepts_labels,
                    }

                    # Save logic
                    should_save = si < effective_seq_len

                    if should_save:
                        with open(save_name, 'wb') as ff:
                            pickle.dump(complete_save_data, ff)
                    
                    current_frame_num += step_size

            if val_itr % 10 == 0:
                logger.info(f'Processed {val_itr + 1} batches')
    
    # Now reconstruct complete 240-frame videos and save them
    logger.info("Reconstructing complete 240-frame videos...")
    
    complete_videos = []
    incomplete_videos = []
    
    for videoname, video_data in video_reconstructions.items():
        frames_collected = video_data['frames_collected']
        concepts_dict = video_data['concepts']
        
        if len(frames_collected) == 240:
            # Complete video - reconstruct full sequence
            complete_videos.append(videoname)
            
            # Create ordered sequence [240, num_concepts]
            video_concepts = np.zeros((240, num_concepts))
            for frame_num in range(1, 241):  # Frames 1-240
                if frame_num in concepts_dict:
                    video_concepts[frame_num - 1] = concepts_dict[frame_num]  # Convert to 0-indexed
                else:
                    logger.warning(f"Missing frame {frame_num} for video {videoname}")

            
            
            # Save complete video
            video_save_path = os.path.join(batch_concepts_dir, f'{videoname}.pt')
            torch.save({
                'concepts': torch.from_numpy(video_concepts),  # [240, num_concepts]
                'video_name': videoname,
                'video_id': video_data['video_id'],
                'total_frames': 240,
                'num_concepts': num_concepts,
                'textual_concepts': val_dataset.concepts_labels,
                'frame_coverage': len(frames_collected)
            }, video_save_path)
            
            logger.info(f"Saved complete 240-frame video: {videoname}")
            
        else:
            # Incomplete video
            incomplete_videos.append(videoname)
            logger.warning(f"Incomplete video {videoname}: {len(frames_collected)}/240 frames")
            
            # Still save what we have
            max_frame = max(frames_collected) if frames_collected else 0
            video_concepts = np.zeros((max_frame, num_concepts))
            
            for frame_num in frames_collected:
                video_concepts[frame_num - 1] = concepts_dict[frame_num]
            
            video_save_path = os.path.join(batch_concepts_dir, f'{videoname}_incomplete_{len(frames_collected)}frames.pt')
            torch.save({
                'concepts': torch.from_numpy(video_concepts),
                'video_name': videoname,
                'video_id': video_data['video_id'],
                'total_frames': max_frame,
                'num_concepts': num_concepts,
                'textual_concepts': val_dataset.concepts_labels,
                'frame_coverage': len(frames_collected),
                'missing_frames': set(range(1, max_frame + 1)) - frames_collected
            }, video_save_path)
    
    # Summary
    logger.info(f'Concept extraction completed. Saved to {concept_save_dir}')
    logger.info(f'Complete videos: {len(complete_videos)}')
    logger.info(f'Incomplete videos: {len(incomplete_videos)}')
    logger.info(f'Batch concept tensors saved to {batch_concepts_dir}')


def filter_detections_with_concepts(args, scores, decoded_boxes_batch, confidences):
    """
    Filter detections and extract concept logits.
    
    Args:
        args: Arguments
        scores: Detection confidence scores [num_anchors]
        decoded_boxes_batch: Decoded bounding boxes [num_anchors, 4]  
        confidences: Full confidence tensor [num_anchors, num_classes]
    
    Returns:
        cls_dets: Filtered detections [N, 5] (boxes + scores)
        save_data: Combined data for saving [N, 4 + num_classes] (boxes + all confidences)  
        frame_concept_logits: Frame-level concept representation [num_classes]
    """
    
    # Apply confidence threshold
    c_mask = scores.gt(args.GEN_CONF_THRESH)
    scores = scores[c_mask].squeeze()
    
    if scores.dim() == 0 or scores.shape[0] == 0:
        num_classes = confidences.shape[-1]
        return (np.zeros((0, 5)), 
                np.zeros((0, 4 + num_classes)), 
                np.zeros(num_classes))
    
    # Filter boxes and confidences
    boxes = decoded_boxes_batch[c_mask, :].clone().view(-1, 4)
    numc = confidences.shape[-1]
    filtered_confidences = confidences[c_mask, :].clone().view(-1, numc)
    
    # Apply NMS
    max_k = min(args.GEN_TOPK * 500, scores.shape[0])
    ids, counts = nms(boxes, scores, args.GEN_NMS, max_k)
    
    # Get top detections
    top_k = min(args.GEN_TOPK, counts)
    final_ids = ids[:top_k]
    
    final_scores = scores[final_ids].cpu().numpy()
    final_boxes = boxes[final_ids, :].cpu().numpy()
    final_confidences = filtered_confidences[final_ids, :].cpu().numpy()
    
    # Create detection output (same as gen_dets)
    cls_dets = np.hstack((final_boxes, final_scores[:, np.newaxis])).astype(np.float32, copy=True)
    
    # Create save data: boxes + all confidence scores (including concept logits)
    save_data = np.hstack((final_boxes, final_confidences)).astype(np.float32)
    
    # Extract frame-level concept representation
    frame_concept_logits = extract_frame_level_concepts(final_confidences)
    
    return cls_dets, save_data, frame_concept_logits


def extract_frame_level_concepts(final_confidences):
    """
    Extract frame-level concept representation from filtered detections.
    
    Args:
        final_confidences: Already filtered confidence tensor [num_final_detections, num_classes]
    
    Returns:
        frame_concepts: Frame-level concept logits [num_classes]
    """
    
    if len(final_confidences) == 0:
        return np.zeros(final_confidences.shape[-1] if len(final_confidences.shape) > 1 else 0)
    
    # Convert to tensor if it's numpy
    if isinstance(final_confidences, np.ndarray):
        final_confidences = torch.from_numpy(final_confidences)
    
    # Aggregate using max pooling
    frame_concepts_max = torch.max(final_confidences, dim=0)[0]
    
    frame_concepts = frame_concepts_max.cpu().numpy()
    
    return frame_concepts



def extract_concepts(args, val_dataset):
    """
    Concept extraction mode integration

    """
    # Build and load model
    net = build_retinanet(args).cuda()
    if args.MULTI_GPUS:
        logger.info('\nLets do dataparallel\n')
        net = torch.nn.DataParallel(net)
    
    # Extract concepts
    output_dir = args.SAVE_ROOT
    extract_concepts_for_gridlock(args, net, val_dataset, output_dir)
 