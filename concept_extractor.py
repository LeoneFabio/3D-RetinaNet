"""
Concept Extraction Module for 3D-RetinaNet to GridLock Integration

This module modifies the existing gen_dets pipeline to extract concept representations
that can replace CLIP in the GridLock model, while maintaining the same output format
as gen_dets for consistency and debugging.
"""

import os
import time
import torch
import numpy as np
import pickle
import torch.utils.data as data_utils
from data import custom_collate_comma
from data import custum_collate
from modules import utils
from modules.box_utils import decode, nms

logger = utils.get_logger(__name__)

def extract_concepts_for_gridlock(args, net, val_dataset, output_dir):
    """
    Extract concept representations from 3D-RetinaNet for GridLock integration.
    Fixed version for 240-frame videos with proper sequence reconstruction.
    
    Args:
        args: Argument namespace
        net: Trained 3D-RetinaNet model
        val_dataset: Dataset (e.g., Comma2k19)
        output_dir: Directory to save concept representations
    
    Returns:
        Saves detections + concept logits in gen_dets compatible format + reconstructed 240-frame sequences
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

            logger.info(f'Processing batch {val_itr}: seq_len={seq_len}, effective_seq_len={effective_seq_len}')

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

                    # Save logic: save all frames except the last skip_ending frames
                    should_save = (frame_key > 238) or (si < effective_seq_len and not ((frame_key == 233 and si == 0) or (frame_key == 234 and si == 1)))

                    if should_save:
                        with open(save_name, 'wb') as ff:
                            pickle.dump(complete_save_data, ff)
                    
                    logger.info(f"Frame {frame_key} (seq_idx {si}), video: {videoname}, saved: {should_save}")
                    
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
            video_save_path = os.path.join(batch_concepts_dir, f'{videoname}_240frames.pt')
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
    
    # Create batch-style tensor from complete videos
    if complete_videos:
        logger.info(f"Creating batch tensor from {len(complete_videos)} complete videos...")
        
        # Load all complete videos and stack them
        batch_concepts = []
        batch_video_names = []

        batch_counter = 0 
        
        for videoname in complete_videos:
            video_path = os.path.join(batch_concepts_dir, f'{videoname}_240frames.pt')
            video_data = torch.load(video_path)
            batch_concepts.append(video_data['concepts'])  # [240, num_concepts]
            batch_video_names.append(videoname)
            
            # Create chunks of videos for batch processing
            if len(batch_concepts) == args.TEST_BATCH_SIZE or videoname == complete_videos[-1]:
                # Stack videos to create batch tensor
                batch_tensor = torch.stack(batch_concepts, dim=0)  # [batch_size, 240, num_concepts]
                
                batch_save_name = os.path.join(batch_concepts_dir, f'batch_{batch_counter:03d}.pt')
                
                torch.save({
                    'concepts': batch_tensor,  # [batch_size, 240, num_concepts]
                    'video_names': batch_video_names.copy(),
                    'batch_size': batch_tensor.shape[0],
                    'seq_len': 240,
                    'num_concepts': num_concepts,
                    'textual_concepts': val_dataset.concepts_labels
                }, batch_save_name)
                
                logger.info(f"Saved batch tensor with shape {batch_tensor.shape}: {batch_save_name}")
                
                # Reset for next batch
                batch_concepts = []
                batch_video_names = []
                batch_counter += 1
    
    # Summary
    logger.info(f'Concept extraction completed. Saved to {concept_save_dir}')
    logger.info(f'Complete videos: {len(complete_videos)}')
    logger.info(f'Incomplete videos: {len(incomplete_videos)}')
    logger.info(f'Batch concept tensors saved to {batch_concepts_dir}')

'''
def extract_concepts_for_gridlock(args, net, val_dataset, output_dir):
    """
    Extract concept representations from 3D-RetinaNet for GridLock integration.
    Maintains the same output format as gen_dets but adds concept logits.
    
    Args:
        args: Argument namespace
        net: Trained 3D-RetinaNet model
        val_dataset: Dataset (e.g., Comma2k19)
        output_dir: Directory to save concept representations
    
    Returns:
        Saves detections + concept logits in gen_dets compatible format + per batch logits
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
    
    # Setup save directory similar to gen_dets
    concept_save_dir = os.path.join(output_dir, "concepts-{it:02d}-{sq:02d}/".format(
        it=epoch, sq=args.TEST_SEQ_LEN))
    os.makedirs(concept_save_dir, exist_ok=True)
    logger.info('Concept extraction saving dir: ' + concept_save_dir)


    # Setup batch concept tensors save directory
    batch_concepts_dir = os.path.join(output_dir, "batch_concepts-{it:02d}-{sq:02d}/".format(
        it=epoch, sq=args.TEST_SEQ_LEN))
    os.makedirs(batch_concepts_dir, exist_ok=True)
    logger.info('Batch concept tensors saving dir: ' + batch_concepts_dir)
    
    activation = torch.nn.Sigmoid().cuda()
    
    processed_videos = []

    all_concepts = []
    first_batch_info = None  
    accumulated_seq_len = 0
    target_seq_len = 240
    chunk_idx = 0
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


            # Initialize batch-level concept tensor
            batch_concept_logits = torch.zeros(batch_size, seq_len, num_concepts, device=images.device)
            
            # Collect video information for this batch
            batch_video_info = []

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

                batch_video_info.append({
                    'video_name': videoname,
                    'frame_start': frame_num+1  # it starts from 1 rather than 0
                })
                
                # Track processed videos for consistency with gen_dets
                store_last = False
                if videoname not in processed_videos:
                    processed_videos.append(videoname)
                    store_last = True
                
                if not os.path.isdir(save_dir):
                    os.makedirs(save_dir)
                
                # Process each frame in the sequence
                for si in range(seq_len):
                    
                    # Extract frame-level data
                    decoded_boxes_batch = decoded_boxes[b, si]
                    confidence_batch = confidence[b, si]  # [num_anchors, num_classes]
                    ego_frame = ego_probs[b, si, :].cpu().numpy()
                    
                    # Get detection scores and apply filtering (similar to gen_dets)
                    scores = confidence_batch[:, 0].squeeze().clone()  # Use first class as objectness
                    
                    # Filter detections and get concept logits
                    cls_dets, save_data, frame_concept_logits = filter_detections_with_concepts(
                        args, scores, decoded_boxes_batch, confidence_batch
                    )
                      
                    # Save in gen_dets format with added concept information
                    save_name = '{:s}/{:05d}.pkl'.format(save_dir, frame_num + 1)
                    frame_num += step_size
                    
                    # Create save data with same structure as gen_dets
                    complete_save_data = {
                        'ego': ego_frame,
                        'main': save_data,  # Contains boxes + confidence + concept logits
                        'video_name': videoname,  # Video name
                        'concepts': val_dataset.concepts_labels,  # All concepts used in the video
                    }

                    # Store frame concept logits in batch tensor
                    batch_concept_logits[b, si] = torch.from_numpy(frame_concept_logits).to(images.device)
                    
                    # Save following gen_dets logic for sequence handling
                    if si < seq_len - getattr(args, 'skip_ending', 0) or store_last:
                       
                        with open(save_name, 'wb') as ff:
                            pickle.dump(complete_save_data, ff)
                    logger.info(f"Saving frame {frame_num-step_size+1} (seq index {si}), will_save={si < seq_len - getattr(args, 'skip_ending', 0) or store_last}")

            
            """# Save batch concept logits tensor
            batch_save_name = os.path.join(batch_concepts_dir, f'batch_{val_itr:06d}_concepts.pt')
            torch.save({
                'concepts': batch_concept_logits.cpu(),  # [batch_size, seq_len, num_concepts]
                'textual_concepts': val_dataset.concepts_labels,
                'num_concepts': num_concepts-1,
                'video_info': batch_video_info,  # Detailed info per batch element
                'unique_videos': list(set([info['video_name'] for info in batch_video_info]))
            }, batch_save_name)"""
            
            # Accumulate
            all_concepts.append(batch_concept_logits.cpu())
            accumulated_seq_len += seq_len 

            # Save info about first batch only once
            if first_batch_info is None:
                first_batch_info = {
                    'textual_concepts': val_dataset.concepts_labels,
                    'num_concepts': num_concepts - 1,
                    'video_info': batch_video_info,
                    'unique_videos': list(set([info['video_name'] for info in batch_video_info]))
                }

            # If 240 frames (total seq_len) reached
            if accumulated_seq_len >= target_seq_len:
                merged = torch.cat(all_concepts, dim=1)  # [batch_size, accumulated_seq_len, num_concepts]

                batch_save_name = os.path.join(batch_concepts_dir, f'batch_{chunk_idx:03d}_concepts.pt')
                torch.save({
                    'concepts': merged,
                    **first_batch_info
                }, batch_save_name)

                logger.info(f"Saved chunk {chunk_idx} with shape {merged.shape} to {batch_save_name}")

                # reset for next chunk
                all_concepts = []
                accumulated_seq_len = 0
                chunk_idx += 1

            if val_itr % 10 == 0:
                logger.info(f'Processed {val_itr + 1} batches')
    
    # at the end, if there is any remaining data not multiple of 240
    if all_concepts:
        merged = torch.cat(all_concepts, dim=1)
        batch_save_name = os.path.join(batch_concepts_dir, f'batch_{chunk_idx:03d}_concepts.pt')
        torch.save({
            'concepts': merged,
            **first_batch_info
        }, batch_save_name)
        logger.info(f"Saved final chunk {chunk_idx} with shape {merged.shape} to {batch_save_name}")
    logger.info(f'Concept extraction completed. Saved to {concept_save_dir}')
    logger.info(f'Batch concept tensors saved to {batch_concepts_dir}')
'''

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
    
    # Alternative: weighted average
    detection_weights = torch.softmax(final_confidences.max(dim=1)[0], dim=0)
    frame_concepts_weighted = torch.sum(
        final_confidences * detection_weights.unsqueeze(1), dim=0
    )
    
    frame_concepts = frame_concepts_max.cpu().numpy()
    
    return frame_concepts


# Modified main function integration
def add_concept_extraction_mode(args, val_dataset):
    """
    Add concept extraction mode to the existing main.py argument parser.
    
    This should be integrated into your main.py file.
    """
    
    if args.MODE == 'extract_concepts':
        from models.retinanet import build_retinanet
        
        # Build and load model
        net = build_retinanet(args).cuda()
        if args.MULTI_GPUS:
            logger.info('\nLets do dataparallel\n')
            net = torch.nn.DataParallel(net)
        
        # Extract concepts
        output_dir = args.SAVE_ROOT
        extract_concepts_for_gridlock(args, net, val_dataset, output_dir)
        
        return True  # Indicates concept extraction was performed
    
    return False  # Continue with normal processing
