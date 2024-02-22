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

model_from_run = "https://wandb.ai/gorillas/Embedding-SwinV2-CXL-Open/runs/69ok0oyl"
known_embeddings_data_dir = "/workspaces/gorillatracker/data/splits/ground_truth-cxl-face_images-openset-reid-val-0-test-0-mintraincount-3-seed-42-train-50-val-25-test-25"
known_embeddings_data_loader = "gorillatracker.datasets.cxl.CXLDataset"


def get_last_seen_frame_id(df: pd.DataFrame, search_frame_id: int) -> int:
    """Returns the last frame_id where the individual was seen before the search_frame_id."""
    return df[df["frame_id"] < search_frame_id]["frame_id"].max()

def get_following_seen_frame_id(df: pd.DataFrame, search_frame_id: int) -> int:
    """Returns the first frame_id where the individual was seen after the search_frame_id."""
    return df[df["frame_id"] > search_frame_id]["frame_id"].min()

def get_intersection_over_face_area(body_bbox, face_bbox):
    x1, y1, w1, h1 = body_bbox
    x2, y2, w2, h2 = face_bbox
    
    # scale the values up by factor 1000 to avoid floating point errors
    x1, y1, w1, h1 = x1 * 1000, y1 * 1000, w1 * 1000, h1 * 1000
    x2, y2, w2, h2 = x2 * 1000, y2 * 1000, w2 * 1000, h2 * 1000
    
    width = min(x1 + w1/2, x2 + w2/2) - max(x1 - w1/2, x2 - w2/2)
    height = min(y1 + h1/2, y2 + h2/2) - max(y1 - h1/2, y2 - h2/2)
    
    return width * height / (w2 * h2)


HPI_ORANGE = (8, 97, 221)
HPI_RED = (58, 6, 177)

def rounded_rectangle(img, top_left, bottom_right, color,thickness, line_type=cv2.LINE_AA, radius=5):
    """
    Draws a rounded rectangle on the image.

    Parameters:
    - img: The image to draw on.
    - top_left: The top-left corner of the rectangle (before rounding) as a tuple (x, y).
    - bottom_right: The bottom-right corner of the rectangle (before rounding) as a tuple (x, y).
    - radius: The radius of the rounded corners.
    - color: Rectangle color as a tuple (B, G, R).
    - thickness: Thickness of the lines that make up the rectangle. Use -1 for filled.
    """
    # Ensure thickness is not greater than radius for aesthetics
    thickness = min(thickness, radius)

    # Calculate key points for straight lines
    points = {
        "top_left_inner": (top_left[0] + radius, top_left[1]),
        "top_right_inner": (bottom_right[0] - radius, top_left[1]),
        "bottom_left_inner": (top_left[0] + radius, bottom_right[1]),
        "bottom_right_inner": (bottom_right[0] - radius, bottom_right[1]),
    }

    # Draw straight lines
    cv2.line(img, points["top_left_inner"], points["top_right_inner"], color, thickness, line_type)
    cv2.line(img, (top_left[0], top_left[1] + radius), (top_left[0], bottom_right[1] - radius), color, thickness, line_type)
    cv2.line(img, (bottom_right[0], top_left[1] + radius), (bottom_right[0], bottom_right[1] - radius), color, thickness, line_type)
    cv2.line(img, points["bottom_left_inner"], points["bottom_right_inner"], color, thickness, line_type)
    
    # Draw rounded corners using ellipses
    cv2.ellipse(img, (top_left[0] + radius, top_left[1] + radius), (radius, radius), 180, 0, 90, color, thickness, line_type)
    cv2.ellipse(img, (bottom_right[0] - radius, top_left[1] + radius), (radius, radius), 270, 0, 90, color, thickness, line_type)
    cv2.ellipse(img, (top_left[0] + radius, bottom_right[1] - radius), (radius, radius), 90, 0, 90, color, thickness, line_type)
    cv2.ellipse(img, (bottom_right[0] - radius, bottom_right[1] - radius), (radius, radius), 0, 0, 90, color, thickness, line_type)

cv2.rounded_rectangle = rounded_rectangle


def annotate_video_with_bboxes(df: pd.DataFrame, input_video_path: str, fps_ratio: float = 1.0):
    assert fps_ratio >= 1.0, "fps_ratio must be between 0.0 and 1.0"
    if fps_ratio > 1.0:
        df["frame_id"] = (df["frame_id"] * fps_ratio).astype(int) 
    
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
    
    # only keep rows where the label_string is NaN or does correspond to the predominant label_string for the tracking_id
    df_per_tracking_id = df[["tracking_id", "label_string"]].groupby("tracking_id").aggregate(lambda x: x.mode())
    df = df[df.apply(lambda x: pd.isna(x["label_string"]) or x["label_string"] == df_per_tracking_id.loc[x["tracking_id"]]["label_string"], axis=1)]
    

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Get the current frame number
        current_frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1  # Frame IDs start at 0
        
        frame_data_idx_prev = get_last_seen_frame_id(df, current_frame_id)
        frame_data_idx_next = get_following_seen_frame_id(df, current_frame_id)
        
        frame_data_prev = df[df["frame_id"] == frame_data_idx_prev]
        frame_data_next = df[df["frame_id"] == frame_data_idx_next]
        
        # Inner join
        frame_data = pd.merge(frame_data_prev, frame_data_next, on="tracking_id", suffixes=("_prev", "_next"))
        
        # remove faces where the label_string is not the same for prev and next
        unequal_labels = frame_data["label_string_prev"] != frame_data["label_string_next"]
        frame_data.loc[unequal_labels, ["face_bbox_prev", "face_bbox_next"]] = pd.Series([np.nan, np.nan])
        
        # remove faces where intersection area / face area < 0.5 
        frame_data["face_bbox_intersection_area"] = frame_data.apply(lambda x: 0 if pd.isna(x["face_bbox_prev"]) else get_intersection_over_face_area(x["body_bbox_prev"], x["face_bbox_prev"]),  axis=1)
        
        too_less_intersection = frame_data["face_bbox_intersection_area"] < 0.75
        frame_data.loc[too_less_intersection, ["face_bbox_prev", "face_bbox_next"]] = pd.Series([np.nan, np.nan])

        # only keep the row where the bounding boxes are closest for a tracking_id
        frame_data["bbox_distance"] = frame_data.apply(lambda x: 0 if pd.isna(x["face_bbox_prev"]) or pd.isna(x["face_bbox_next"]) else np.linalg.norm(np.array(x["face_bbox_prev"][0:2]) - np.array(x["face_bbox_next"][0:2])), axis=1, result_type=None)
        frame_data = frame_data.sort_values("bbox_distance").groupby("tracking_id").head(1)
        
        
        ft = cv2.free
        
        # Iterate over rows to draw bounding boxes
        for _, row in frame_data.iterrows():
            # Draw body bounding box
            t = float((current_frame_id - frame_data_idx_prev) / (frame_data_idx_next - frame_data_idx_prev))
            body_bbox = [prev * (1 - t) + next * t for prev, next in zip(row["body_bbox_prev"], row["body_bbox_next"])]
            
            p1, p2 = convert_from_yolo_format(body_bbox, frame_width, frame_height)
            overlay = frame.copy()
            cv2.rounded_rectangle(overlay, p1, p2, HPI_RED, 4)
            alpha = 0.5
            cv2.addWeighted(overlay, 1 - alpha, frame, alpha, 0, frame)
            
            if pd.notna(row["label_string_prev"]):
                label = str(row["label_string_prev"])
                cv2.putText(frame, label, (p1[0], p1[1] - 10), cv2.FONT_HERSHEY_TRIPLEX, 2, HPI_RED, 2)

            if pd.notna(row["face_bbox_prev"]) and pd.notna(row["face_bbox_next"]):
                face_bbox = [prev * (1 - t) + next * t for prev, next in zip(row["face_bbox_prev"], row["face_bbox_next"])]
                p1, p2 = convert_from_yolo_format(face_bbox, frame_width, frame_height)
                cv2.rounded_rectangle(frame, p1, p2, HPI_ORANGE, 2)  # Red for face

                # Annotate face bounding box with label string
                # label = str(row["label_string_prev"])
                # cv2.putText(frame, label, (p1[0], p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

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
    # check if any face_embedding
    
    # filter out `face_embedding` is NaN
    df_filtered = df.dropna(subset=["face_embedding"])

    def mean_embeddings(group):
        # Assuming embeddings are numpy arrays, if they're lists, you might need to convert them first
        embeddings = np.array(group.tolist()).copy()
        
        embeddings = embeddings.reshape(embeddings.shape[0], -1)
        embeddings_mean_dist = np.linalg.norm(embeddings - embeddings.mean(axis=0), axis=1)
        # trim outliers (20% according to euclidean distance)
        mask = embeddings_mean_dist < np.percentile(embeddings_mean_dist, 80)
        
        embeddings = embeddings[mask] # only keep the closest 80% of the embeddings
        return embeddings.mean(axis=0).reshape(1, -1)

    # (tracking_id, mean_embedding)
    df_mean_embeddings = df_filtered.groupby("tracking_id")["face_embedding"].agg(mean_embeddings).reset_index()

    for _, row in df_mean_embeddings.iterrows():
        tracking_id = row["tracking_id"]
        embedding = row["face_embedding"]
        label = lg.get_label_for_embeddings(embedding)
        df.loc[df["tracking_id"] == tracking_id, "label_string"] = label
    
    return df


def reduce_frame_rate(input_video, output_video, target_fps):
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        # Only write every nth frame to the output video
        if frame_count % int(fps / target_fps) == 0:
            if out is None:
                out = cv2.VideoWriter(output_video, fourcc, target_fps, (frame_width, frame_height))
            out.write(frame)

    cap.release()
    if out is not None:
        out.release()
        return fps / target_fps
    return 1.0


@st.cache_data
def process_video(path: str, model_from_run: str = model_from_run):
    df = get_tracking_and_embedding_data_for_video(path, model_from_run=model_from_run)
    # TODO(liamvdv): REMOVE THIS, fix in get_tracking_and_embedding_data_for_video
    df.rename(columns={"individual_id": "tracking_id"}, inplace=True)
    return df


def main():
    st.title("GorillaTracker Re-ID")
    video = st.file_uploader("Upload a video", type=["mp4"])

    if video is not None:
        print(video.name, video.size, video.type)
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
        
        output_video_path = os.path.join("/tmp", video.name.split(".")[0] + "-reduced.mp4")
        fps_ratio = reduce_frame_rate(name, output_video_path, 10)
        
        df = process_video(output_video_path)  # cached
        st.success("Video processed successfully")

        df = reidentify(df)
        st.success("Re-identification successful")

        with st.expander("View Dataframe"):
            displayable = df.copy()
            displayable["face_embedding"] = displayable["face_embedding"].apply(lambda x: x.tolist() if pd.notna(x) else None)
            st.dataframe(displayable)

        annotated_video_fp = annotate_video_with_bboxes(df, name, fps_ratio)
        st.write(annotated_video_fp)
        
        # Why does this not work?
        # with open(annotated_video_fp, "rb") as f:
        #     st.video(f)

        with open(annotated_video_fp, "rb") as f:
            st.download_button("Download Annotated Video", data=f, file_name="annotated_video.mp4")


if __name__ == "__main__":
    main()
