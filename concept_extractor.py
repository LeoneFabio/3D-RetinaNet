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
    Maintains the same output format as gen_dets but adds concept logits.
    
    Args:
        args: Argument namespace
        net: Trained 3D-RetinaNet model
        val_dataset: Dataset (e.g., Comma2k19)
        output_dir: Directory to save concept representations
    
    Returns:
        Saves detections + concept logits in gen_dets compatible format
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
    
    activation = torch.nn.Sigmoid().cuda()
    
    processed_videos = []
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
                        #'frame_concepts': frame_concept_logits,  # Frame-level aggregated concepts
                        #'raw_concepts': confidence_batch.cpu().numpy()  # Raw per-anchor concepts
                    }
                    
                    # Save following gen_dets logic for sequence handling
                    if si < seq_len - getattr(args, 'skip_ending', 0) or store_last:
                        with open(save_name, 'wb') as ff:
                            pickle.dump(complete_save_data, ff)
                logger.info(f"Saving frame {frame_num+1} (seq index {si}), will_save={si < seq_len - getattr(args, 'skip_ending', 0) or store_last}")

            if val_itr % 10 == 0:
                logger.info(f'Processed {val_itr + 1} batches')
    
    logger.info(f'Concept extraction completed. Saved to {concept_save_dir}')


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
    frame_concept_logits = extract_frame_level_concepts(confidences, c_mask, final_ids)
    
    return cls_dets, save_data, frame_concept_logits


def extract_frame_level_concepts(confidences, c_mask, final_ids):
    """
    Extract frame-level concept representation from filtered detections.
    
    Args:
        confidences: Full confidence tensor [num_anchors, num_classes]
        c_mask: Confidence mask for filtering
        final_ids: Final detection IDs after NMS
    
    Returns:
        frame_concepts: Frame-level concept logits [num_classes]
    """
    
    if len(final_ids) == 0:
        return np.zeros(confidences.shape[-1])
    
    # Get concepts from high-confidence detections
    high_conf_concepts = confidences[c_mask][final_ids]  # [num_detections, num_classes]
    
    # Aggregate using max pooling (strongest concept evidence)
    frame_concepts_max = torch.max(high_conf_concepts, dim=0)[0]
    
    # Alternative: weighted average by detection confidence
    detection_weights = torch.softmax(high_conf_concepts.max(dim=1)[0], dim=0)
    frame_concepts_weighted = torch.sum(
        high_conf_concepts * detection_weights.unsqueeze(1), dim=0
    )
    
    # Use max pooling as default (can be made configurable)
    frame_concepts = frame_concepts_max.cpu().numpy()
    
    return frame_concepts


def load_concepts_from_detections(detection_path):
    """
    Load concept representations from saved detection files.
    Compatible with both gen_dets and concept extraction outputs.
    
    Args:
        detection_path: Path to saved .pkl file
    
    Returns:
        concepts: Dictionary containing different concept representations
    """
    
    with open(detection_path, 'rb') as f:
        data = pickle.load(f)
    
    concepts = {
        'ego_concepts': data['ego'],  # Ego-vehicle actions
        'detection_concepts': None,   # Per-detection concepts
        #'frame_concepts': None,       # Frame-level concepts
        #'raw_concepts': None         # Raw per-anchor concepts
    }
    
    # Handle different data formats
    if 'frame_concepts' in data:
        concepts['frame_concepts'] = data['frame_concepts']
    
    if 'raw_concepts' in data:
        concepts['raw_concepts'] = data['raw_concepts']
    
    # Extract concepts from detection data
    main_data = data['main']
    if main_data.shape[0] > 0 and main_data.shape[1] > 5:  # Has concept logits
        concepts['detection_concepts'] = main_data[:, 4:]  # Everything after boxes
    
    return concepts


def create_gridlock_compatible_loader(concept_dir, sequence_length=8):
    """
    Create a concept loader that provides sequences compatible with GridLock,
    using the detection-based concept extraction format.
    
    Args:
        concept_dir: Directory containing extracted concepts
        sequence_length: Length of sequences to return
    
    Returns:
        Function that loads concept sequences for GridLock
    """
    
    def load_concept_sequence(videoname, start_frame, step_size=1, aggregation='frame_concepts'):
        """
        Load a sequence of concept representations for GridLock.
        
        Args:
            videoname: Video identifier
            start_frame: Starting frame number  
            step_size: Frame step size
            aggregation: Type of concepts to use ('frame_concepts', 'ego_concepts', 'combined')
        
        Returns:
            logits_per_image: [seq_len, num_concepts] - GridLock compatible format
        """
        
        sequence_concepts = []
        
        for i in range(sequence_length):
            frame_num = start_frame + i * step_size
            detection_path = os.path.join(concept_dir, videoname, f'{frame_num:05d}.pkl')
            
            try:
                concepts = load_concepts_from_detections(detection_path)
                
                if aggregation == 'frame_concepts' and concepts['frame_concepts'] is not None:
                    frame_repr = torch.from_numpy(concepts['frame_concepts'])
                elif aggregation == 'ego_concepts':
                    frame_repr = torch.from_numpy(concepts['ego_concepts'])
                elif aggregation == 'combined':
                    # Combine frame and ego concepts
                    frame_c = concepts['frame_concepts'] if concepts['frame_concepts'] is not None else np.array([])
                    ego_c = concepts['ego_concepts']
                    if len(frame_c) > 0:
                        combined = np.concatenate([frame_c, ego_c])
                    else:
                        combined = ego_c
                    frame_repr = torch.from_numpy(combined)
                else:
                    # Fallback to ego concepts
                    frame_repr = torch.from_numpy(concepts['ego_concepts'])
                
                sequence_concepts.append(frame_repr)
                
            except FileNotFoundError:
                logger.warning(f"Concepts not found for {videoname} frame {frame_num}")
                # Use last valid frame or zeros
                if sequence_concepts:
                    sequence_concepts.append(sequence_concepts[-1].clone())
                else:
                    # Create appropriate zero tensor (size will be determined from first valid frame)
                    sequence_concepts.append(torch.zeros(1))  # Placeholder
        
        if not sequence_concepts:
            raise ValueError(f"No valid concepts found for {videoname} starting at frame {start_frame}")
        
        # Handle placeholder zeros by using first valid frame size
        concept_size = max(c.numel() for c in sequence_concepts)
        for i, c in enumerate(sequence_concepts):
            if c.numel() == 1 and c.item() == 0:  # Placeholder zero
                sequence_concepts[i] = torch.zeros(concept_size)
        
        return torch.stack(sequence_concepts, dim=0)
    
    return load_concept_sequence


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


# GridLock Integration Helper
def replace_clip_in_gridlock(concept_loader, scenarios_tokens=None):
    """
    Function to replace CLIP forward pass in GridLock with concept extraction.
    
    Args:
        concept_loader: Function that loads concept sequences
        scenarios_tokens: Not used in our approach (kept for interface compatibility)
    
    Returns:
        Function that mimics CLIP's interface but uses extracted concepts
    """
    
    def concept_forward(img, video_info):
        """
        Replacement for CLIP forward pass in GridLock.
        
        Args:
            img: [batch_size, seq_len, h, w, c] - input images  
            video_info: Dictionary with video/frame information for concept loading
        
        Returns:
            logits_per_image: [batch_size * seq_len, num_concepts] - concept logits
            logits_per_text: None (we don't use text matching)
        """
        
        batch_size, seq_len = img.shape[:2]
        
        logits_list = []
        for b in range(batch_size):
            # Extract video information (you'll need to adapt this to your data structure)
            video_name = video_info['video_names'][b]
            start_frame = video_info['start_frames'][b]
            step_size = video_info.get('step_sizes', [1] * batch_size)[b]
            
            # Load concept sequence
            concepts = concept_loader(video_name, start_frame, step_size, aggregation='combined')
            logits_list.append(concepts)
        
        # Stack and reshape to match CLIP output format
        logits_per_image = torch.cat(logits_list, dim=0)  # [batch_size * seq_len, num_concepts]
        
        return logits_per_image, None  # No text logits needed
    
    return concept_forward