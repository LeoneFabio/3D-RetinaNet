"""
Concept Extraction Module for 3D-RetinaNet to GridLock Integration

This module modifies the existing gen_dets pipeline to extract concept representations
that can replace CLIP in the GridLock model.
"""

import os
import time
import torch
import numpy as np
import pickle
import torch.utils.data as data_utils
from data import custom_collate_comma
from modules import utils

logger = utils.get_logger(__name__)

def extract_concepts_for_gridlock(args, net, val_dataset, output_dir):
    """
    Extract concept representations from 3D-RetinaNet for GridLock integration.
    
    Args:
        args: Argument namespace
        net: Trained 3D-RetinaNet model
        val_dataset: Dataset (e.g., Comma2k19)
        output_dir: Directory to save concept representations
    
    Returns:
        Saves concept logits in GridLock-compatible format
    """
    
    net.eval()
    val_data_loader = data_utils.DataLoader(
        val_dataset, int(args.TEST_BATCH_SIZE), num_workers=args.NUM_WORKERS,
        shuffle=False, pin_memory=True, collate_fn=custom_collate_comma
    )
    
    # Load trained model weights
    '''for epoch in args.EVAL_EPOCHS:
        args.MODEL_PATH = args.SAVE_ROOT + 'model_{:06d}.pth'.format(epoch)
        net.load_state_dict(torch.load(args.MODEL_PATH))
        logger.info('Loaded model from %s' % args.MODEL_PATH)
        break  # Use the first (likely best) epoch'''
    epoch =  int(args.EVAL_EPOCHS)
    args.MODEL_PATH = args.SAVE_ROOT + 'model_{:06d}.pth'.format(epoch)
    net.load_state_dict(torch.load(args.MODEL_PATH))
    logger.info('Loaded model from %s' % args.MODEL_PATH)
    
    concept_save_dir = os.path.join(output_dir, "concepts")
    os.makedirs(concept_save_dir, exist_ok=True)
    
    activation = torch.nn.Sigmoid().cuda()
    
    with torch.no_grad():
        for val_itr, (images, gt_boxes, gt_targets, ego_labels, batch_counts, img_indexs, wh) in enumerate(val_data_loader):
            
            batch_size = images.size(0)
            images = images.cuda(0, non_blocking=True)
            
            # Forward pass - get raw outputs
            decoded_boxes, flat_conf, ego_preds = net(images)
            
            # Apply activation to get probabilities
            concept_probs = activation(flat_conf)  # [batch, seq_len, num_anchors, num_classes]
            ego_probs = activation(ego_preds)      # [batch, seq_len, num_ego_classes]
            
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
                
                # Extract concept logits for this sequence
                batch_concept_logits = extract_sequence_concepts(
                    concept_probs[b], ego_probs[b], args
                )
                
                # Save in GridLock format
                save_concepts_gridlock_format(
                    batch_concept_logits, videoname, frame_num, step_size, concept_save_dir
                )
            
            if val_itr % 10 == 0:
                logger.info(f'Processed {val_itr + 1} batches')
    
    logger.info(f'Concept extraction completed. Saved to {concept_save_dir}')


def extract_sequence_concepts(concept_probs, ego_probs, args):
    """
    Extract frame-level concept representations from detection outputs.
    
    Args:
        concept_probs: [seq_len, num_anchors, num_classes] - detection confidence scores
        ego_probs: [seq_len, num_ego_classes] - ego-vehicle action probabilities
        args: Arguments containing class information
    
    Returns:
        logits_per_image: [seq_len, total_concepts] - GridLock compatible format
    """
    
    seq_len = concept_probs.shape[0]
    
    # Method 1: Max pooling across anchors (strongest detection per concept per frame)
    # This mimics "what concepts are most confidently detected in this frame"
    frame_concepts_max = torch.max(concept_probs, dim=1)[0]  # [seq_len, num_classes]
    
    # Method 2: Mean pooling across anchors (average concept presence)
    frame_concepts_mean = torch.mean(concept_probs, dim=1)   # [seq_len, num_classes]
    
    # Method 3: Weighted sum by detection confidence (focus on high-confidence regions)
    # Get max confidence per anchor across all classes
    anchor_weights = torch.max(concept_probs, dim=2)[0]  # [seq_len, num_anchors]
    anchor_weights = torch.softmax(anchor_weights, dim=1)  # Normalize weights
    frame_concepts_weighted = torch.sum(
        concept_probs * anchor_weights.unsqueeze(2), dim=1
    )  # [seq_len, num_classes]
    
    # Choose aggregation method (can be made configurable)
    # Max pooling often works well for concept detection
    frame_concepts = frame_concepts_max
    
    # Combine with ego predictions
    # Ego actions are complementary to detected concepts
    combined_concepts = torch.cat([frame_concepts, ego_probs], dim=1)  # [seq_len, num_classes + num_ego_classes]
    
    return combined_concepts


def save_concepts_gridlock_format(concept_logits, videoname, start_frame, step_size, save_dir):
    """
    Save concept logits in a format compatible with GridLock.
    
    Args:
        concept_logits: [seq_len, total_concepts] - extracted concepts
        videoname: Video identifier
        start_frame: Starting frame number
        step_size: Frame step size
        save_dir: Directory to save concepts
    """
    
    seq_len = concept_logits.shape[0]
    
    video_save_dir = os.path.join(save_dir, videoname)
    os.makedirs(video_save_dir, exist_ok=True)
    
    for i in range(seq_len):
        frame_num = start_frame + i * step_size
        
        # Save individual frame concepts
        frame_concepts = {
            'logits_per_image': concept_logits[i].cpu().numpy(),
            'frame_num': frame_num,
            'video_name': videoname
        }
        
        save_path = os.path.join(video_save_dir, f'{frame_num:05d}_concepts.pkl')
        
        with open(save_path, 'wb') as f:
            pickle.dump(frame_concepts, f)


def load_concepts_for_gridlock(concept_dir, videoname, frame_num):
    """
    Load concept representations for GridLock inference.
    
    Args:
        concept_dir: Directory containing saved concepts
        videoname: Video identifier
        frame_num: Frame number
    
    Returns:
        logits_per_image: Concept logits for the frame
    """
    
    concept_path = os.path.join(concept_dir, videoname, f'{frame_num:05d}_concepts.pkl')
    
    if not os.path.exists(concept_path):
        raise FileNotFoundError(f"Concepts not found for {videoname} frame {frame_num}")
    
    with open(concept_path, 'rb') as f:
        concepts = pickle.load(f)
    
    return torch.from_numpy(concepts['logits_per_image'])


def create_gridlock_concept_loader(concept_dir, sequence_length=8):
    """
    Create a concept loader that provides sequences compatible with GridLock.
    
    Args:
        concept_dir: Directory containing extracted concepts
        sequence_length: Length of sequences to return
    
    Returns:
        Function that loads concept sequences
    """
    
    def load_concept_sequence(videoname, start_frame, step_size=1):
        """
        Load a sequence of concept representations.
        
        Returns:
            logits_per_image: [seq_len, num_concepts] - ready for GridLock
        """
        sequence_concepts = []
        
        for i in range(sequence_length):
            frame_num = start_frame + i * step_size
            try:
                frame_concepts = load_concepts_for_gridlock(concept_dir, videoname, frame_num)
                sequence_concepts.append(frame_concepts)
            except FileNotFoundError:
                # Handle missing frames with zeros or last valid frame
                if sequence_concepts:
                    sequence_concepts.append(sequence_concepts[-1])  # Repeat last frame
                else:
                    # Create zero tensor with appropriate size (need to determine from data)
                    # This is a fallback - in practice you'd want to handle this better
                    sequence_concepts.append(torch.zeros(100))  # Placeholder size
        
        return torch.stack(sequence_concepts, dim=0)
    
    return load_concept_sequence


# Integration function for GridLock model
def replace_clip_with_concepts(gridlock_model, concept_loader):
    """
    Helper function to integrate concept extraction into GridLock model.
    This replaces the CLIP forward pass with our concept extraction.
    
    Args:
        gridlock_model: The GridLock model instance
        concept_loader: Function that loads pre-extracted concepts
    
    Note: This is a conceptual function - the actual integration depends on 
    how GridLock is structured in your codebase.
    """
    
    def concept_forward(img, scenarios_tokens):
        """
        Replacement for CLIP forward pass in GridLock.
        
        Args:
            img: [batch_size, seq_len, h, w, c] - input images
            scenarios_tokens: Text scenarios (not used in our approach)
        
        Returns:
            logits_per_image: [batch_size * seq_len, num_concepts] - concept logits
            logits_per_text: None (we don't use text matching)
        """
        
        batch_size, seq_len = img.shape[:2]
        
        # In practice, you'd extract video/frame info from img tensor or metadata
        # For now, this is a placeholder showing the expected interface
        
        logits_list = []
        for b in range(batch_size):
            for s in range(seq_len):
                # Extract concepts for this frame
                # You'll need to map img[b,s] to actual video/frame identifiers
                concepts = concept_loader("video_name", frame_num=s)  # Placeholder
                logits_list.append(concepts)
        
        logits_per_image = torch.stack(logits_list, dim=0)
        
        return logits_per_image, None  # No text logits in our approach
    
    return concept_forward


# Modified main function integration
def add_concept_extraction_mode(args, val_dataset):
    """
    Add concept extraction mode to the existing main.py argument parser.
    
    This should be integrated into your main.py file.
    """
    
    if args.MODE == 'extract_concepts':
        from models.retinanet import build_retinanet
        '''from data import VideoDataset
        from torchvision import transforms
        import data.transforms as vtf
        
        
        # Setup dataset (same as gen_dets)
        val_transform = transforms.Compose([ 
            vtf.ResizeClip(args.MIN_SIZE, args.MAX_SIZE),
            vtf.ToTensorStack(),
            vtf.Normalize(mean=args.MEANS, std=args.STDS)
        ])
        
        args.SUBSETS = args.TEST_SUBSETS  # Use test subsets for concept extraction
        val_dataset = VideoDataset(args, train=False, transform=val_transform, 
                                 skip_step=args.SEQ_LEN, full_test=True)
        '''

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