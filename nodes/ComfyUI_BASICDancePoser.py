import torch
import numpy as np
import librosa
import cv2
import random
import warnings

class ComfyUI_BASICDancePoser:
    """
    Enhanced DWPose-style stick figure dance poser with EQ-driven, beat-synced motion.
    Optimized for Wan VACE at 24 FPS with robust tempo handling and smooth animation.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "start_time": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3600.0, "step": 0.1}),
                "max_duration": ("FLOAT", {"default": 180.0, "min": 3.0, "max": 180.0, "step": 0.1}),
                "width": ("INT", {"default": 480, "min": 64, "max": 2048}),
                "height": ("INT", {"default": 832, "min": 64, "max": 2048}),
                "fps": ("INT", {"default": 24, "min": 12, "max": 60}),
                "dance_style": (["energetic", "smooth", "dramatic"], {"default": "energetic"}),
                "movement_intensity": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.1}),
                "smoothing": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05}),
                "leg_style": (["step", "shuffle", "march"], {"default": "step"}),
                "beat_sensitivity": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.1}),
                "tempo_factor": (["1x", "0.5x", "0.25x"], {"default": "1x"}),
            },
            "optional": {
                "background_color": (["black", "white", "gray"], {"default": "black"}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 99999}),
                "debug": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("pose_video", "audio")
    FUNCTION = "generate_pose_video"
    CATEGORY = "BSC/Audio"

    def __init__(self):
        self.pose_connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 11), (6, 12), (11, 12),  # Torso
            (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
        ]
        self.base_pose = [
            [0.5, 0.1], [0.48, 0.08], [0.52, 0.08], [0.46, 0.08], [0.54, 0.08],  # Head
            [0.4, 0.25], [0.6, 0.25], [0.35, 0.4], [0.65, 0.4], [0.3, 0.55], [0.7, 0.55],  # Arms
            [0.42, 0.5], [0.58, 0.5], [0.4, 0.7], [0.6, 0.7], [0.38, 0.9], [0.62, 0.9]  # Legs
        ]
        self.arm_poses = [
            [[0.35, 0.4], [0.65, 0.4], [0.3, 0.55], [0.7, 0.55]],  # Neutral
            [[0.35, 0.15], [0.65, 0.15], [0.3, 0.08], [0.7, 0.08]],  # Arms up
            [[0.25, 0.5], [0.75, 0.5], [0.22, 0.6], [0.78, 0.6]],  # Arms to side
            [[0.5, 0.3], [0.5, 0.3], [0.45, 0.45], [0.55, 0.45]],  # Cross-body
            [[0.32, 0.28], [0.68, 0.28], [0.28, 0.21], [0.72, 0.21]]  # Elbows bent
        ]

    def generate_pose_video(self, audio, start_time, max_duration, width, height, fps, dance_style,
                           movement_intensity, smoothing, leg_style, beat_sensitivity, tempo_factor,
                           background_color="black", seed=42, debug=False):
        random.seed(seed)
        np.random.seed(seed)
        waveform = audio["waveform"].cpu().numpy()
        sample_rate = audio["sample_rate"]

        # Handle waveform shape
        if waveform.ndim == 3 and waveform.shape[0] == 1 and waveform.shape[1] == 2:
            waveform = waveform.squeeze(0)
        elif waveform.ndim == 2 and waveform.shape[0] == 2:
            pass
        elif waveform.ndim == 1:
            pass
        else:
            raise ValueError(f"Expected stereo (2, n_samples), batched stereo (1, 2, n_samples), or mono (n_samples,) waveform, got shape {waveform.shape}")

        # Convert stereo to mono
        if waveform.ndim == 2 and waveform.shape[0] == 2:
            waveform = np.mean(waveform, axis=0)

        # Validate waveform length
        if len(waveform) < sample_rate * 3:
            raise ValueError(f"Waveform too short: {len(waveform)/sample_rate:.2f} seconds, minimum 3 seconds required")

        # Trim audio
        total_duration = len(waveform) / sample_rate
        if start_time < 0 or start_time >= total_duration:
            raise ValueError(f"start_time {start_time:.2f} seconds is invalid; must be between 0 and {total_duration:.2f} seconds")
        max_samples = int(min(max_duration, 180.0) * sample_rate)
        start_sample = int(start_time * sample_rate)
        end_sample = min(start_sample + max_samples, len(waveform))
        waveform = waveform[start_sample:end_sample]
        duration = (end_sample - start_sample) / sample_rate
        if duration < 3:
            raise ValueError(f"Trimmed audio duration {duration:.2f} seconds is too short; must be at least 3 seconds")

        trimmed_audio = {
            "waveform": torch.from_numpy(waveform).unsqueeze(0).to(audio["waveform"].device),
            "sample_rate": sample_rate
        }

        # Analyze audio
        features = self.analyze_audio_for_dance(waveform, sample_rate, fps)
        pose_seq = self.create_dance_sequence(features, dance_style, movement_intensity,
                                              smoothing, leg_style, beat_sensitivity, tempo_factor, debug)
        frames = self.render_pose_sequence(pose_seq, width, height, background_color, debug)
        return (frames, trimmed_audio)

    def analyze_audio_for_dance(self, waveform, sample_rate, fps):
        # Extract BPM with robust handling
        tempo = librosa.beat.tempo(y=waveform, sr=sample_rate)
        if isinstance(tempo, (int, float)):
            bpm = np.clip(float(tempo), 60, 200)
        else:
            bpm = np.clip(tempo[0], 60, 200)
        if not isinstance(tempo, (int, float, np.ndarray)) or len(waveform) == 0:
            raise ValueError("Failed to detect tempo from waveform, check audio input")
        if (isinstance(tempo, (list, np.ndarray)) and (tempo[0] < 60 or tempo[0] > 200)) or (isinstance(tempo, (int, float)) and (tempo < 60 or tempo > 200)):
            warnings.warn(f"BPM {tempo[0] if isinstance(tempo, (list, np.ndarray)) else tempo:.1f} outside 60-200 range, clamped to {bpm:.1f}")
        beat_interval = 60 / bpm
        frame_interval = int(beat_interval * fps)

        # Beat tracking
        _, beat_frames = librosa.beat.beat_track(y=waveform, sr=sample_rate, units='frames', hop_length=512)
        beat_times = librosa.frames_to_time(beat_frames, sr=sample_rate, hop_length=512)

        # EQ segmentation
        S = np.abs(librosa.stft(waveform, n_fft=2048, hop_length=int(sample_rate / fps)))
        freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=2048)
        low = (freqs >= 20) & (freqs < 200)
        mid = (freqs >= 200) & (freqs < 2000)
        high = (freqs >= 2000) & (freqs < 8000)
        low_energy = S[low, :].mean(axis=0)
        mid_energy = S[mid, :].mean(axis=0)
        high_energy = S[high, :].mean(axis=0)
        low_energy = low_energy / (low_energy.max() + 1e-6)
        mid_energy = mid_energy / (mid_energy.max() + 1e-6)
        high_energy = high_energy / (high_energy.max() + 1e-6)

        # Map to video frames
        duration = len(waveform) / sample_rate
        total_frames = int(duration * fps)
        beat_strength = np.zeros(total_frames)
        for bt in beat_times:
            idx = int((bt / duration) * total_frames)
            if 0 <= idx < total_frames:
                beat_strength[idx] = 1.0
        energy_len = S.shape[1]
        frame_idxs = np.linspace(0, energy_len - 1, total_frames).astype(np.int32)
        low_band = low_energy[frame_idxs]
        mid_band = mid_energy[frame_idxs]
        high_band = high_energy[frame_idxs]

        return {
            "beat_strength": beat_strength,
            "low_band": low_band,
            "mid_band": mid_band,
            "high_band": high_band,
            "total_frames": total_frames,
            "duration": duration,
            "frame_interval": frame_interval
        }

    def create_dance_sequence(self, features, dance_style, movement_intensity, smoothing, leg_style,
                              beat_sensitivity, tempo_factor, debug=False):
        config = {
            "energetic": {"arm": 0.5, "body": 0.3, "leg": 0.35},
            "smooth": {"arm": 0.2, "body": 0.4, "leg": 0.18},
            "dramatic": {"arm": 0.7, "body": 0.2, "leg": 0.5}
        }[dance_style]
        total_frames = features["total_frames"]
        frame_interval = features["frame_interval"]
        min_interval = max(4, int(frame_interval * 0.3))
        max_interval = min(60, int(frame_interval * 2))
        effective_interval = max(
            min_interval,
            min(max_interval, int(frame_interval * {"1x": 1.0, "0.5x": 0.5, "0.25x": 0.25}[tempo_factor]))
        )
        prev_pose = np.array(self.base_pose)
        pose_seq = []
        step_phase = 1
        prev_arm_pose = np.array(self.arm_poses[0])
        blend_frames = 5
        arm_indices = [7, 8, 9, 10]

        for i in range(total_frames):
            beat = features["beat_strength"][i]
            low = features["low_band"][i]
            mid = features["mid_band"][i]
            high = features["high_band"][i]

            # Quantized beat hit
            is_key_beat = (i % effective_interval < 5) and (beat > beat_sensitivity or low > 0.7)
            if is_key_beat:
                step_phase *= -1

            pose = np.array(self.base_pose)

            # --- Leg movement with smoothing ---
            leg_mag = config["leg"] * movement_intensity * (0.5 + 0.8 * low)
            if leg_style == "shuffle":
                target_leg = [
                    pose[15] + np.array([step_phase * leg_mag * 0.08, 0]),
                    pose[16] + np.array([-step_phase * leg_mag * 0.08, 0])
                ]
            elif leg_style == "march":
                target_leg = [
                    pose[15] + np.array([0, -step_phase * leg_mag * 0.10]),
                    pose[16] + np.array([0, step_phase * leg_mag * 0.10])
                ]
            else:  # step
                target_leg = [
                    pose[15] + np.array([0, step_phase * leg_mag * 0.09]),
                    pose[16] + np.array([0, -step_phase * leg_mag * 0.09])
                ]
            target_leg = np.array(target_leg)
            blend_idx = min(i % blend_frames, blend_frames - 1)
            alpha = blend_idx / (blend_frames - 1)
            pose[15:17] = (1 - alpha) * pose[15:17] + alpha * target_leg

            # --- Arm pose selection and interpolation ---
            arm_trig = (mid > 0.7 or high > 0.5) or (i % (effective_interval * 2) == 0)
            new_arm_pose = np.array(random.choice(self.arm_poses)) if arm_trig else np.array(self.arm_poses[0])
            for j, idx in enumerate(arm_indices):
                pose[idx] = (1 - alpha) * prev_arm_pose[j] + alpha * new_arm_pose[j]
            prev_arm_pose = new_arm_pose

            # Arm/Body movement
            arm_mag = config["arm"] * movement_intensity * (0.6 + 0.7 * mid + 0.3 * high) * 0.5
            body_mag = config["body"] * movement_intensity * (0.7 + 0.5 * mid) * 0.5
            pose[7][0] += arm_mag * np.sin(i * 0.15)
            pose[8][0] -= arm_mag * np.sin(i * 0.15)
            pose[9][1] += arm_mag * np.abs(np.sin(i * 0.10))
            pose[10][1] += arm_mag * np.abs(np.cos(i * 0.10))
            sway = body_mag * np.sin(i * 0.05)
            for idx in [5, 6, 11, 12]:
                pose[idx][0] += sway

            # Smoothing (skip first frame)
            pose_final = pose if i == 0 else smoothing * pose + (1.0 - smoothing) * prev_pose
            prev_pose = pose_final

            # Keep in bounds
            pose_final = np.clip(pose_final, 0.08, 0.92)
            pose_seq.append(pose_final.tolist())

            # Debug
            if debug and i % 24 == 0:
                try:
                    import matplotlib.pyplot as plt
                    plt.figure(figsize=(5, 10))
                    plt.plot([p[0] for p in pose_final], [p[1] for p in pose_final], 'g.-')
                    plt.title(f"Frame {i}, Beat: {beat:.2f}")
                    plt.gca().invert_yaxis()
                    plt.show()
                except ImportError:
                    print("Matplotlib not available for debug overlay")

        return pose_seq

    def render_pose_sequence(self, pose_sequence, width, height, background_color, debug=False):
        bg_colors = {
            "black": (0, 0, 0),
            "white": (255, 255, 255),
            "gray": (128, 128, 128)
        }
        bg_color = bg_colors.get(background_color, (0, 0, 0))

        frames = []
        for pose in pose_sequence:
            frame = np.full((height, width, 3), bg_color, dtype=np.uint8)
            keypoints = [(int(x * width), int(y * height)) for x, y in pose]
            for c in self.pose_connections:
                if c[0] < len(keypoints) and c[1] < len(keypoints):
                    cv2.line(frame, keypoints[c[0]], keypoints[c[1]], (255, 255, 255), 2)
            for pt in keypoints:
                cv2.circle(frame, pt, 4, (0, 255, 0), -1)
            frames.append(torch.from_numpy(frame.astype(np.float32) / 255.0))
        return torch.stack(frames)

# Node registration
NODE_CLASS_MAPPINGS = {"ComfyUI_BASICDancePoser": ComfyUI_BASICDancePoser}
NODE_DISPLAY_NAME_MAPPINGS = {"ComfyUI_BASICDancePoser": "BASIC Dance Poser"}
