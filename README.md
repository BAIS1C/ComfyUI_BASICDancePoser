ComfyUI_BASICDancePoser
Enhanced DWPose-style stick figure dance poser for ComfyUI.
Turns any audio into a dancing pose “stickman” video, ideal for Wan2.1-VACE, reference-to-video, or video-to-video generation.
EQ-driven, beat-synced, customizable, and robust for long-form video and VRAM-limited setups.

Features
Audio-to-dance:
Analyzes music/voice for beats, rhythm, and energy, then animates a humanoid pose sequence.

ControlNet-style output:
Video is a series of stick figure frames—ready for VACE, ControlNet, or reference-driven video generation.

Highly customizable:
Choose dance style, leg movement mode, movement intensity, tempo sync (1x, 0.5x, 0.25x), background color, and smoothing.

Robust BPM/EQ detection:
Works on most music and voice tracks, even if not perfectly clean.

Debug mode:
Optional matplotlib visualization of poses every second (if matplotlib installed).

Native VRAM safety:
Optimized for 24 FPS, 480x832 (Wan2.1-VACE portrait), but scales up or down easily.

Node Inputs
Input	Type	Description
audio	AUDIO	Input audio tensor (mono or stereo, see ComfyUI spec)
start_time	FLOAT	Time offset (in seconds) to start dance in audio
max_duration	FLOAT	Max duration (seconds) for output video (up to 3 minutes)
width, height	INT	Output video resolution (portrait recommended)
fps	INT	Frames per second (default: 24)
dance_style	SELECT	“energetic”, “smooth”, “dramatic” movement archetype
movement_intensity	FLOAT	0.0 (none) to 1.0 (full) for all movement
smoothing	FLOAT	0.0 (jerky) to 1.0 (fluid, slow transitions)
leg_style	SELECT	“step”, “shuffle”, “march” for different footwork
beat_sensitivity	FLOAT	How much energy is required for a move (default 0.1)
tempo_factor	SELECT	“1x” = every beat, “0.5x” = every other beat, “0.25x” = slower
background_color	SELECT (opt)	“black”, “white”, “gray” (default: black)
seed	INT (opt)	Random seed (reproducible output)
debug	BOOLEAN (opt)	Show debug pose plots every second (if matplotlib available)

Outputs
pose_video: IMAGE (actually a stack of torch images, one per frame; ControlNet-ready stickman video)

audio: AUDIO (the original input audio, unmodified for downstream workflow compatibility)

Usage Example (ComfyUI)
Plug in any audio node or file to the audio input.

Configure parameters as needed (for music, energetic or smooth usually best).

Output “pose_video” to your preferred VACE/ControlNet or video composer.

Connect “audio” to any downstream audio-using node (VHS, video mux, etc).

(Optional): Enable “debug” to visualize poses.

