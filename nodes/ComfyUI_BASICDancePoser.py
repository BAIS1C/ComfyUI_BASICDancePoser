import torch
import numpy as np
import librosa
import cv2
from typing import Dict, List, Tuple

class ComfyUI_BASICDancePoser:
    """
    Generate a DWPose-style stick figure dance video synchronized to audio beats.
    Outputs a video tensor and audio for use in a Wan VACE workflow.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "width": ("INT", {"default": 480, "min": 64, "max": 2048}),  # Wan VACE portrait
                "height": ("INT", {"default": 832, "min": 64, "max": 2048}),  # Wan VACE portrait
                "fps": ("INT", {"default": 30, "min": 12, "max": 60}),
                "dance_style": (["energetic", "smooth", "dramatic"], {"default": "energetic"}),
                "movement_intensity": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 2.0, "step": 0.1}),
                "beat_sensitivity": ("FLOAT", {"default": 0.6, "min": 0.1, "max": 1.0, "step": 0.1}),
            },
            "optional": {
                "background_color": (["black", "white", "gray"], {"default": "black"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("pose_video", "audio")
    FUNCTION = "generate_pose_video"
    CATEGORY = "BSC/Audio"

    def __init__(self):
        # DWPose keypoint connections (COCO 17 keypoints)
        self.pose_connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 11), (6, 12), (11, 12),  # Torso
            (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
        ]
        # Base neutral pose (normalized 0-1 coordinates)
        self.base_pose = [
            [0.5, 0.1], [0.48, 0.08], [0.52, 0.08], [0.46, 0.08], [0.54, 0.08],  # Head
            [0.4, 0.25], [0.6, 0.25], [0.35, 0.4], [0.65, 0.4], [0.3, 0.55], [0.7, 0.55],  # Arms
            [0.42, 0.5], [0.58, 0.5], [0.4, 0.7], [0.6, 0.7], [0.38, 0.9], [0.62, 0.9]  # Legs
        ]

    def generate_pose_video(self, audio, width, height, fps, dance_style, 
                           movement_intensity, beat_sensitivity, background_color="black"):
        """Generate DWPose-style stick figure video synced to audio beats"""
        
        # Extract waveform and sample rate
        waveform = audio["waveform"].cpu().numpy()
        sample_rate = audio["sample_rate"]
        
        # Handle waveform shape
        if waveform.ndim == 3 and waveform.shape[0] == 1 and waveform.shape[1] == 2:  # (1, 2, n_samples)
            waveform = waveform.squeeze(0)  # Remove batch dimension, get (2, n_samples)
        elif waveform.ndim == 2 and waveform.shape[0] == 2:  # Stereo (2, n_samples)
            pass  # Already in correct format
        elif waveform.ndim == 1:  # Mono (n_samples,)
            pass  # Already in correct format
        else:
            raise ValueError(f"Expected stereo (2, n_samples), batched stereo (1, 2, n_samples), or mono (n_samples,) waveform, got shape {waveform.shape}")
        
        # Convert stereo to mono if necessary
        if waveform.ndim == 2 and waveform.shape[0] == 2:
            waveform = np.mean(waveform, axis=0)  # Convert to mono (n_samples,)
        
        # Validate audio duration (3s to 3min)
        duration = len(waveform) / sample_rate
        if not (3 <= duration <= 180):
            raise ValueError(f"Audio duration must be between 3 and 180 seconds, got {duration:.2f} seconds")
        
        # Analyze audio for beats
        audio_features = self.analyze_audio_for_dance(waveform, sample_rate, fps)
        
        # Generate pose sequence
        pose_sequence = self.create_dance_sequence(
            audio_features, dance_style, movement_intensity, beat_sensitivity
        )
        
        # Render pose video
        pose_frames = self.render_pose_sequence(pose_sequence, width, height, background_color)
        
        return (pose_frames, audio)

    def analyze_audio_for_dance(self, waveform, sample_rate, fps):
        """Analyze audio for beat features"""
        
        # Ensure waveform is 1D
        if waveform.ndim != 1:
            raise ValueError(f"Waveform must be 1D, got shape {waveform.shape}")
        
        # Beat tracking
        tempo, beat_frames = librosa.beat.beat_track(
            y=waveform, sr=sample_rate, units='frames', hop_length=512
        )
        beat_times = librosa.frames_to_time(beat_frames, sr=sample_rate, hop_length=512)
        
        # Map to video timeline
        duration = len(waveform) / sample_rate
        total_frames = int(duration * fps)
        frame_times = np.linspace(0, duration, total_frames)
        
        # Create frame-by-frame beat strength
        beat_strength = np.zeros(total_frames)
        for beat_time in beat_times:
            frame_idx = int((beat_time / duration) * total_frames)
            if 0 <= frame_idx < total_frames:
                beat_strength[frame_idx] = 1.0
        
        return {
            "beat_strength": beat_strength,
            "total_frames": total_frames,
            "duration": duration
        }

    def create_dance_sequence(self, audio_features, dance_style, movement_intensity, beat_sensitivity):
        """Create sequence of DWPose keypoints synced to audio beats"""
        
        style_configs = {
            "energetic": {"arm_range": 0.4, "body_sway": 0.3, "leg_movement": 0.2},
            "smooth": {"arm_range": 0.2, "body_sway": 0.4, "leg_movement": 0.1},
            "dramatic": {"arm_range": 0.6, "body_sway": 0.2, "leg_movement": 0.3}
        }
        
        config = style_configs.get(dance_style, style_configs["energetic"])
        pose_sequence = []
        total_frames = audio_features["total_frames"]
        
        # Define multiple base poses for variation
        base_poses = [
            self.base_pose,  # Neutral
            [[0.5, 0.1], [0.48, 0.08], [0.52, 0.08], [0.46, 0.08], [0.54, 0.08],  # Head
             [0.38, 0.25], [0.62, 0.25], [0.32, 0.35], [0.68, 0.35], [0.28, 0.45], [0.72, 0.45],  # Arms
             [0.42, 0.5], [0.58, 0.5], [0.4, 0.7], [0.6, 0.7], [0.38, 0.9], [0.62, 0.9]],  # Legs
            [[0.5, 0.1], [0.48, 0.08], [0.52, 0.08], [0.46, 0.08], [0.54, 0.08],  # Head
             [0.4, 0.25], [0.6, 0.25], [0.35, 0.15], [0.65, 0.15], [0.3, 0.08], [0.7, 0.08],  # Arms up
             [0.42, 0.5], [0.58, 0.5], [0.4, 0.7], [0.6, 0.7], [0.38, 0.9], [0.62, 0.9]]  # Legs
        ]
        
        for frame_idx in range(total_frames):
            beat = audio_features["beat_strength"][frame_idx]
            
            # Select base pose based on beat
            pose_idx = 1 if beat > beat_sensitivity else 0
            if frame_idx % 100 == 0:  # Switch to arms-up pose periodically
                pose_idx = 2
            current_pose = base_poses[pose_idx].copy()
            
            # Apply movement based on beat
            arm_intensity = beat * config["arm_range"] * movement_intensity
            body_intensity = beat * config["body_sway"] * movement_intensity
            leg_intensity = beat * config["leg_movement"] * movement_intensity
            
            if beat > beat_sensitivity:
                arm_intensity *= 1.5
                leg_intensity *= 1.8
            
            frame_pose = self.apply_dance_movement(
                current_pose, arm_intensity, body_intensity, leg_intensity, frame_idx, total_frames
            )
            
            pose_sequence.append(frame_pose)
        
        return pose_sequence

    def apply_dance_movement(self, keypoints, arm_intensity, body_intensity, leg_intensity, frame_idx, total_frames):
        """Apply simple dance movements to base keypoints"""
        
        time_factor = (frame_idx / total_frames) * 2 * np.pi
        
        # Arm movements
        if arm_intensity > 0:
            keypoints[7][0] += arm_intensity * np.sin(time_factor * 2) * 0.3  # Left elbow
            keypoints[9][0] += arm_intensity * np.sin(time_factor * 3) * 0.4  # Left wrist
            keypoints[9][1] += arm_intensity * np.cos(time_factor * 2) * 0.3
            keypoints[8][0] -= arm_intensity * np.sin(time_factor * 2) * 0.3  # Right elbow
            keypoints[10][0] -= arm_intensity * np.sin(time_factor * 3) * 0.4  # Right wrist
            keypoints[10][1] += arm_intensity * np.cos(time_factor * 2) * 0.3
        
        # Body sway
        if body_intensity > 0:
            sway_x = body_intensity * np.sin(time_factor) * 0.1
            for idx in [5, 6, 11, 12]:  # Shoulders and hips
                keypoints[idx][0] += sway_x
        
        # Leg movements
        if leg_intensity > 0:
            keypoints[15][1] += leg_intensity * abs(np.sin(time_factor * 4)) * 0.1  # Left ankle
            keypoints[16][1] += leg_intensity * abs(np.sin(time_factor * 4)) * 0.1  # Right ankle
        
        # Keep keypoints in bounds
        for point in keypoints:
            point[0] = np.clip(point[0], 0.1, 0.9)
            point[1] = np.clip(point[1], 0.1, 0.9)
        
        return keypoints

    def render_pose_sequence(self, pose_sequence, width, height, background_color):
        """Render pose sequence to DWPose-style video frames"""
        
        bg_colors = {
            "black": (0, 0, 0),
            "white": (255, 255, 255), 
            "gray": (128, 128, 128)
        }
        bg_color = bg_colors.get(background_color, (0, 0, 0))
        
        frames = []
        for pose_keypoints in pose_sequence:
            frame = np.full((height, width, 3), bg_color, dtype=np.uint8)
            pixel_keypoints = [(int(x * width), int(y * height)) for x, y in pose_keypoints]
            
            # Draw connections (white lines)
            for connection in self.pose_connections:
                start_idx, end_idx = connection
                if start_idx < len(pixel_keypoints) and end_idx < len(pixel_keypoints):
                    cv2.line(frame, pixel_keypoints[start_idx], pixel_keypoints[end_idx], (255, 255, 255), 2)
            
            # Draw keypoints (green dots)
            for point in pixel_keypoints:
                cv2.circle(frame, point, 4, (0, 255, 0), -1)
            
            frame_tensor = torch.from_numpy(frame.astype(np.float32) / 255.0)
            frames.append(frame_tensor)
        
        return torch.stack(frames)

# Node registration
NODE_CLASS_MAPPINGS = {
    "ComfyUI_BASICDancePoser": ComfyUI_BASICDancePoser
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyUI_BASICDancePoser": "BASIC Dance Poser"
}