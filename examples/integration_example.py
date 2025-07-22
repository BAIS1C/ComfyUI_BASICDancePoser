# Integration Example for DJ Beat-Sync Animation System
# This shows how to use the nodes in your ComfyUI workflow

"""
WORKFLOW SETUP:
1. Install the nodes in your ComfyUI custom_nodes directory
2. Install dependencies: librosa, numpy
3. Connect to your existing Wan VC workflow

BASIC WORKFLOW STRUCTURE:
Audio File → DJBeatSyncAnimation → DJPromptScheduler → Your Wan VC Pipeline
"""

# Example usage of the nodes in Python (for testing):

def test_dj_animation_system():
    """Test the DJ animation system with sample data"""
    
    # Initialize nodes
    beat_sync_node = DJBeatSyncAnimationNode()
    prompt_scheduler = DJPromptSchedulerNode()
    
    # Test parameters
    audio_path = "/path/to/your/music.wav"  # Replace with actual path
    video_fps = 24
    video_duration = 3.0
    character_name = "Kira"
    
    base_prompt = """A young woman named Kira energetically mixes music at a nightclub DJ booth. 
    She wears a vibrant blue crop top and high-waisted black pants, her long silver hair 
    styled in loose waves, sunglasses reflecting colorful club lights"""
    
    # Generate animation data
    animation_json, lip_sync, gesture_intensity, beat_triggers = beat_sync_node.generate_dj_animation(
        audio_path=audio_path,
        video_fps=video_fps,
        video_duration=video_duration,
        beat_sensitivity=0.6,
        energy_threshold=0.7,
        gesture_intensity=1.2
    )
    
    print("Generated Animation Data Summary:")
    print(f"Lip-sync tensor shape: {lip_sync.shape}")
    print(f"Gesture tensor shape: {gesture_intensity.shape}")
    print(f"Beat triggers shape: {beat_triggers.shape}")
    
    # Generate prompts for each frame
    total_frames = int(video_fps * video_duration)
    frame_prompts = []
    
    for frame in range(total_frames):
        positive_prompt, negative_prompt, strength, debug = prompt_scheduler.generate_frame_prompt(
            animation_data=animation_json,
            character_name=character_name,
            base_prompt=base_prompt,
            current_frame=frame,
            gesture_style="energetic",
            prompt_intensity=1.0
        )
        
        frame_prompts.append({
            "frame": frame,
            "positive": positive_prompt,
            "negative": negative_prompt,
            "strength": float(strength)
        })
        
        # Print first few frames for demo
        if frame < 3:
            print(f"\n--- Frame {frame} ---")
            print(f"Positive: {positive_prompt[:100]}...")
            print(f"Strength: {strength:.2f}")
    
    return frame_prompts, animation_json

# ComfyUI Integration Guide
"""
TO INTEGRATE WITH YOUR EXISTING WORKFLOW:

1. INSTALLATION:
   - Copy both node files to: ComfyUI/custom_nodes/
   - Install: pip install librosa numpy
   - Restart ComfyUI

2. NODE CONNECTIONS:
   
   [Load Audio] → [DJBeatSyncAnimation] → [DJPromptScheduler] → [Your Wan VC Nodes]
                                      ↓
                               [Additional Control Nodes]

3. DETAILED WORKFLOW:

   # Audio Analysis Stage
   Audio File Input → DJBeatSyncAnimation Node
   ├── Outputs: animation_data (JSON)
   ├── Outputs: lip_sync_params (tensor)
   ├── Outputs: gesture_intensity (tensor)  
   └── Outputs: beat_triggers (tensor)

   # Frame-by-Frame Processing
   For each video frame:
   animation_data + frame_index → DJPromptScheduler Node
   ├── Outputs: positive_prompt (dynamic text)
   ├── Outputs: negative_prompt (enhanced)
   ├── Outputs: gesture_strength (float)
   └── Outputs: debug_info (JSON)

   # Video Generation
   positive_prompt → Text Encoder → Wan VC Node
   reference_image → Image Encoder → Wan VC Node
   gesture_strength → ControlNet (if using pose control)

4. ADVANCED INTEGRATION OPTIONS:

   A. WITH POSE CONTROL:
      gesture_intensity → Pose Generator → ControlNet → Wan VC
   
   B. WITH LORA/STYLING:
      gesture_strength → LoRA Weight Scheduler → Wan VC
   
   C. WITH BATCH PROCESSING:
      Use ComfyUI's batch nodes to process entire sequences

5. PARAMETER RECOMMENDATIONS:

   For 3-second test clips:
   - video_fps: 24
   - beat_sensitivity: 0.5-0.8 (higher = more sensitive to beats)
   - energy_threshold: 0.6-0.8 (higher = only react to strong energy)
   - gesture_intensity: 0.8-1.2 (multiplier for movement strength)
   - gesture_style: "energetic" for club scenes

   For longer videos:
   - Increase overlap_frames in your Wan VC settings to 16-24
   - Use lower gesture_intensity (0.6-0.8) to avoid excessive movement
   - Consider "professional" style for more controlled gestures

6. TROUBLESHOOTING:

   If movements are too chaotic:
   - Decrease gesture_intensity (0.4-0.6)
   - Increase energy_threshold (0.8+)
   - Use "subtle" or "professional" style
   - Check your audio file for clipping/distortion

   If lip-sync isn't working:
   - Provide transcript_path for better vocal detection
   - Ensure audio has clear vocals
   - Check vocal_active values in debug_info

   If beats aren't detected properly:
   - Adjust beat_sensitivity (try 0.3-0.8 range)
   - Ensure audio is clear and has strong rhythm
   - Check tempo in animation_data JSON

7. EXAMPLE COMFYUI JSON WORKFLOW SNIPPET:

```json
{
  "1": {
    "class_type": "DJBeatSyncAnimation",
    "inputs": {
      "audio_path": "/path/to/music.wav",
      "video_fps": 24,
      "video_duration": 3.0,
      "beat_sensitivity": 0.6,
      "energy_threshold": 0.7,
      "gesture_intensity": 1.0
    }
  },
  "2": {
    "class_type": "DJPromptScheduler", 
    "inputs": {
      "animation_data": ["1", 0],
      "character_name": "Kira",
      "base_prompt": "A young woman named Kira DJing at a nightclub...",
      "current_frame": 0,
      "gesture_style": "energetic"
    }
  },
  "3": {
    "class_type": "WanVcTVd",
    "inputs": {
      "positive_prompt": ["2", 0],
      "negative_prompt": ["2", 1], 
      "reference_image": ["your_reference_node", 0],
      "gesture_strength": ["2", 2]
    }
  }
}
```

8. BATCH PROCESSING FOR FULL VIDEOS:

For videos longer than a few seconds, you'll want to process in chunks:

```python
def process_long_video(audio_path, total_duration, chunk_duration=5.0):
    chunks = []
    for start_time in range(0, int(total_duration), int(chunk_duration)):
        chunk_end = min(start_time + chunk_duration, total_duration)
        
        # Process each chunk
        chunk_animation = beat_sync_node.generate_dj_animation(
            audio_path=audio_path,
            video_fps=24,
            video_duration=chunk_end - start_time,
            # Add audio offset parameter for chunk processing
        )
        chunks.append(chunk_animation)
    
    return chunks
```

9. CUSTOM GESTURE EXPANSION:

To add new DJ gestures, extend the gesture_templates in DJPromptSchedulerNode:

```python
# Add to gesture_templates dictionary
"crossfader_scratch": "{char} aggressively working the crossfader with skilled scratching motion",
"headphone_cue": "{char} holding one headphone to ear while cueing next track",  
"crowd_interaction": "{char} pointing to the crowd and hyping them up",
"air_horn": "{char} triggering air horn with dramatic button press"
```

Then modify the determine_primary_action method to detect when to use them.

10. PERFORMANCE OPTIMIZATION:

- Cache audio analysis results for repeated processing
- Use lower hop_length (256) for more precise beat detection
- Process audio analysis once, reuse for multiple video generations
- Consider GPU acceleration for librosa if available

11. INTEGRATION WITH ECHOMIMIC:

To add the EchoMimic lip-sync functionality:

```python
# In DJBeatSyncAnimationNode, add this method:
def integrate_echomimic_lipsync(self, audio_path, transcript_path):
    # Load EchoMimic's phoneme extraction
    from echomimic import extract_phonemes, align_audio_text
    
    if transcript_path and os.path.exists(transcript_path):
        phonemes = extract_phonemes(audio_path, transcript_path)
        return phonemes
    else:
        # Fallback to audio-only vocal detection
        return self.detect_vocal_sections(audio_features)
```

This gives you both beat-driven body movement AND accurate lip-sync!
"""

if __name__ == "__main__":
    # Run the test
    try:
        frame_prompts, animation_data = test_dj_animation_system()
        print(f"\nSuccessfully generated {len(frame_prompts)} frame prompts!")
        
        # Save results for inspection
        import json
        with open("dj_animation_test.json", "w") as f:
            json.dump({
                "frame_prompts": frame_prompts,
                "animation_data": animation_data
            }, f, indent=2)
        
        print("Results saved to dj_animation_test.json")
        
    except Exception as e:
        print(f"Test failed: {e}")
        print("Make sure you have librosa installed and provide a valid audio path") gestures are too subtle:
   - Increase gesture_intensity
   - Lower energy_threshold
   - Use "party" gesture_style

   If