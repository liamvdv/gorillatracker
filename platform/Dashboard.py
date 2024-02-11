import streamlit as st
import cv2
import tempfile
from embedding_pipeline import get_tracking_and_embedding_data_for_video
from gorillatracker.utils.yolo_helpers import convert_from_yolo_format
from gorillatracker.utils.embedding_generator import read_embeddings_from_disk

import pandas as pd
import os


# Usage
# streamlit run platform/Dashboard.py


# Generate via python3 repl:
# $ python3
# from gorillatracker.utils.embedding_generator import generate_embeddings_from_run
# generate_embeddings_from_run("https://wandb.ai/gorillas/Embedding-SwinV2Large-CXL-Open/runs/a4t93htr/overview", "embeddings_a4t93htr.pkl")
embedding_file = "embeddings_a4t93htr.pkl"


def annotate_video_with_bboxes(df: pd.DataFrame, input_video_path: str):
    print(df.head())
    print(df.dtypes)
    # Generate output video path
    base, ext = os.path.splitext(input_video_path)
    output_video_path = f"{base}-annotated{ext}"

    # Open the video
    cap = cv2.VideoCapture(input_video_path)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # mp4v does not play with streamlit
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Get the current frame number
        current_frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1  # Frame IDs start at 0

        # Filter dataframe rows for the current frame
        frame_data = df[df["frame_id"] == current_frame_id]

        # Iterate over rows to draw bounding boxes
        for _, row in frame_data.iterrows():
            # TODO(liamvdv): read df['individual_id'] as tracking number.

            # Draw body bounding box
            print(row["body_bbox"], row["face_bbox"], type(row["body_bbox"]), type(row["face_bbox"]))
            # x, y, w, h = row["body_bbox"]  # Assuming the format is a string like "(x, y, x2, y2)"
            p1, p2 = convert_from_yolo_format(row["body_bbox"], frame_width, frame_height)
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2)  # Blue for body

            if pd.notna(row["face_bbox"]):
                # Draw face bounding box
                # x, y, w, h = row["face_bbox"]
                p1, p2 = convert_from_yolo_format(row["face_bbox"], frame_width, frame_height)
                cv2.rectangle(frame, p1, p2, (0, 0, 255), 2)  # Red for face

        # Write the modified frame
        out.write(frame)

    # Cleanup
    cap.release()
    out.release()

    # Return the path of the annotated video
    return output_video_path


@st.cache_resource
def load_known_embeddings(embedding_file: str) -> pd.DataFrame:
    return read_embeddings_from_disk(embedding_file)


# TODO(liamvdv): Use knn to add column 'label_string' to df
def reidentify(df: pd.DataFrame) -> pd.DataFrame:
    # known_df = load_known_embeddings(embedding_file)
    # print(known_df.head())
    # perform knn for every df row that has face_embedding, match to entry in known_df

    return df


@st.cache_data
def process_video(path: str):
    return get_tracking_and_embedding_data_for_video(path)


def main():
    st.title("GorillaTracker Re-ID")
    video = st.file_uploader("Upload a video", type=["mp4"])

    if video is not None:
        # with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        #     tmp_file.write(video.read())
        #     tmp_file.flush()
        #     tmp_file.seek(0)
        #     name = tmp_file.name
        # st.success("Video uploaded successfully")
        name = os.path.join("/tmp", video.name)  # for caching
        with open(name, "wb") as f:
            f.write(video.read())

        with st.expander("View Video"):
            st.video(video)

        df = process_video(name)  # cached
        st.success("Video processed successfully")

        df = reidentify(df)
        st.success("Re-identification successful")

        with st.expander("View Dataframe"):
            st.dataframe(df)

        annotated_video_fp = annotate_video_with_bboxes(df, name)
        st.write(annotated_video_fp)
        with open(annotated_video_fp, "rb") as f:
            st.video(f)

        with open(annotated_video_fp, "rb") as f:
            st.download_button("Download Annotated Video", data=f, file_name="annotated_video.mp4")


if __name__ == "__main__":
    main()
