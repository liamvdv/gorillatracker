import streamlit as st
import cv2
from embedding_pipeline import get_tracking_and_embedding_data_for_video
from gorillatracker.utils.yolo_helpers import convert_from_yolo_format
from gather_labels import LabelGatherer
import pandas as pd
import numpy as np
import os

# TODO(liamvdv): rename individual_id to tracking_id in the dataframe. This is the unique identifier for a yolo tracking. Then edit process_video

# Usage
# streamlit run platform/Dashboard.py


# Generate via python3 repl:
# $ python3
# from gorillatracker.utils.embedding_generator import generate_embeddings_from_run
# generate_embeddings_from_run("https://wandb.ai/gorillas/Embedding-SwinV2Large-CXL-Open/runs/a4t93htr/overview", "embeddings_a4t93htr.pkl")
model_from_run = "https://wandb.ai/gorillas/Embedding-SwinV2-CXL-Open/runs/69ok0oyl"
known_embeddings_data_dir = "/workspaces/gorillatracker/data/splits/ground_truth-cxl-face_images-openset-reid-val-0-test-0-mintraincount-3-seed-42-train-50-val-25-test-25"
known_embeddings_data_loader = "gorillatracker.datasets.cxl.CXLDataset"

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


def reidentify(df: pd.DataFrame) -> pd.DataFrame:
    """Adds the column 'label_string' to the dataframe, which contains the label of the individual in the frame."""
    lg = LabelGatherer(model_from_run=model_from_run, data_dir=known_embeddings_data_dir, dataset_class=known_embeddings_data_loader, use_cache=True)
    
    # iter over lg by tracking id, mean 'embedding column'. Then add gathered label to the df
    # check if any face_embeddings
    
    # filter out `face_embeddings` is NaN
    df_filtered = df.dropna(subset=['face_embeddings'])

    def mean_embeddings(group):
        # Assuming embeddings are numpy arrays, if they're lists, you might need to convert them first
        embeddings = np.array(group.tolist())
        return embeddings.mean(axis=0)

    # (tracking_id, mean_embedding)
    df_mean_embeddings = df_filtered.groupby('tracking_id')['face_embeddings'].agg(mean_embeddings).reset_index()

    for row in df_mean_embeddings.iterrows():
        tracking_id = row['tracking_id']
        embedding = row['face_embeddings']
        label = lg.get_label_for_embedding(embedding)
        df.loc[df['tracking_id'] == tracking_id, 'label_string'] = label
    
    return df


@st.cache_data
def process_video(path: str):
    df = get_tracking_and_embedding_data_for_video(path)
    # TODO(liamvdv): REMOVE THIS, fix in get_tracking_and_embedding_data_for_video
    df.rename(columns={'individual_id': 'tracking_id'}, inplace=True)
    return df


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