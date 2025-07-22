Changelog
v1.0: Initial Version
Basic pose stickman, “marching” on beats.

Hard-coded leg/arm logic, no EQ or style.

Mono audio only; no background color select.

Smoothing fixed at one value.

v2.0: Major Upgrade
Full EQ-driven movement:

Separate low/mid/high bands from audio used to trigger legs, arms, and body individually.

Multiple dance/leg styles:

Choose between step, shuffle, or march; arm poses sampled from a variety of types.

Movement intensity and smoothing fully user-controllable.

Tempo factor (1x, 0.5x, 0.25x) for different dance tempos.

Background color and random seed options.

Debug plotting (matplotlib-based) for quick pose sequence checks.

v2.1: Robustness and ComfyUI Compatibility Fixes
NumPy array math everywhere:
All pose math is now array-safe, never throws list * float errors.

Arm and leg blending:
Uses consistent interpolation for smooth, dance-like transitions.

Output audio is original input audio (not internally trimmed/reshaped audio, which avoids errors in VHS/composer nodes).

Stereo/mono audio detection:
Handles [N], [1, N], or [2, N] input shapes robustly, always passes through audio untouched.

Improved BPM handling:
Works with more librosa versions, handles array or scalar BPM.

Cleaner code/indentation for copy-paste into ComfyUI custom nodes folder.

Troubleshooting & Tips
Pose jumps at first frame?
Set your start_time to a downbeat or intro for smoother alignment.

Weird color or VRAM error?
Lower resolution, lower FPS, or switch background color to black.

Audio shape errors in downstream nodes?
Make sure you’re connecting the “audio” SOURCE output, which is untouched input to the VHSCOMBINE

Slow video preview?
Lower resolution, FPS, or blend_frames for faster generation.

Want even more pose variety?
Add new entries to self.arm_poses or modify pose/gesture logic as desired.

Author & Support