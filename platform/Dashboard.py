import streamlit as st
import cv2
import numpy as np
import tempfile

def sample_frames(video_file):
    # Load the video
    vidcap = cv2.VideoCapture(video_file.name)
    length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    duration = length / fps
    
    # Calculate the interval to sample frames
    interval = length // 15
    frames = []
    
    for i in range(15):
        frame_id = i * interval
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        success, image = vidcap.read()
        if success:
            # Convert the BGR image to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            frames.append(image)
    vidcap.release()
    return frames

def main():
    st.title("Video Frame Sampler")
    video_file = st.file_uploader("Upload a video (30s max)", type=["mp4", "avi", "mov"])
    
    if video_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(video_file.read())
            sampled_frames = sample_frames(tmp_file)
        
        st.write(f"Displaying {len(sampled_frames)} sampled frames:")
        for i, frame in enumerate(sampled_frames):
            st.image(frame, caption=f"Frame {i+1}", use_column_width=True)

if __name__ == "__main__":
    main()
