import json
import os
import subprocess
from typing import Any, Dict, List, MutableSet, Tuple

import cv2

# TODO: properly type
Data = Dict[Any, Any]
# TODO: properly type
BBox = Dict[Any, Any]


class GorillaVideoTracker:
    def __init__(self, path: str, out_path: str = "", video_path: str = "", allowed_overlap: float = 0.25):
        """
        initialize tracking of Individuals on videos with bounding boxes in json
        parameter:
            path: path to json files
            out_path: if not set output will be saved in path directory
            video_path: path to directory where videos are stored; if not set it will try to get videos from path
            allowed_overlap: % of the area of bboxes which is allowed to overlap between two individuals, default is 0.25
        """
        self.path = path
        self.out_path = self.path if out_path == "" else out_path
        self.video_path = self.path if video_path == "" else video_path
        self.allowed_overlap = allowed_overlap

    def track_files(self, log: bool = True, logging_iteration: int = 5) -> None:
        """
        track individuals in path
        parameter:
            log: boolean; if progress should be logged to the terminal, default is True
            logging_iteration: int; how often progress should be logged, default is 5
        """
        files = os.listdir(self.path)
        file_count = len(files)

        assert all(file.endswith(".json") for file in files), "Error: not all files in path are json files"

        for idx, file in enumerate(files):
            if log is True and idx % logging_iteration == 0:
                print(f"tracking...{idx}/{file_count}", end="\r")
            file_path = os.path.join(self.path, file)
            self.track_file(file_path, log=False)

        if log is True:
            print(f"{file_count} files successfully tracked")

    def track_file(self, file_path: str, log: bool = True) -> None:
        """
        track individuals in file
        parameter:
            log: boolean; if progress should be logged to the terminal, default is True
        """
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        out_file = os.path.join(self.out_path, file_name + "_tracked.json")

        if log is True:
            print(f"tracking {file_name}.json", end="\r")

        data = self._read_from_json(file_path)
        data, id_count = self._track_ids(data)
        data = self._label_faces(data)
        negatives = self._get_negatives(data, id_count)
        self._write_to_json(out_file, data, negatives)

        if log is True:
            print(f"{file_name}.json successfully tracked and saved as {file_name}_tracked.json")

    def save_videos(self, max_video: int = 0, log: bool = True, compress: bool = True) -> None:
        """
        save videos with bounding boxes for all tracked files
        parameter:
            max_video: int, how many videos should be saved at maximum, 0 means no maximum, default is 0
            log: boolean; if progress should be logged to the terminal, default is True
            compress: boolean; if videofile should be compressed, default is True
        """
        tracked_files = [file for file in os.listdir(self.out_path) if file.endswith("_tracked.json")]
        max_video_idx = len(tracked_files) if max_video == 0 else min(max_video - 1, len(tracked_files))

        for idx, file in enumerate(tracked_files[: max_video_idx + 1]):
            video_name = os.path.basename(file)[:-13]  # -13 to remove "_tracked.json"
            video_file_path = os.path.join(self.video_path, video_name + ".mp4")

            if log is True:
                print(" " * 80, end="\r")  # clear line
                print(f"saving video {idx + 1}/{max_video_idx + 1}: {video_name}.mp4", end="\r")

            self.save_video(video_file_path, log=False, compress=compress)

        if log is True:
            print(" " * 80, end="\r")  # clear line
            print(f"{max_video_idx + 1} videos successfully saved to {self.out_path}")

    def save_video(self, video_path: str, log: bool = True, compress: bool = True) -> None:
        """
        save video with bounding boxes
        parameter:
            video_path: path to videofile e.g. /path/to/example.mp4
            log: boolean; if progress should be logged to the terminal, default is True
            compress: boolean; if videofile should be compressed, default is True
        """
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        json_path = os.path.join(self.out_path, video_name + "_tracked.json")
        video_out_path = os.path.join(self.out_path, video_name + "_tracked.mp4")

        assert os.path.exists(video_path), f"Error: {video_path} not found"
        assert os.path.exists(json_path), f"Error: {json_path} not found, try calling track() first"

        # log
        if log is True:
            print(f"saving video {video_name}.mp4 to {self.out_path}", end="\r")

        # process video
        self._process_video(video_path, json_path, video_out_path)

        # compress
        if compress is True:
            self._compress_video(video_out_path)

        # log
        if log is True:
            print(f"video {video_name}.mp4 successfully saved to {self.out_path}")

    def _process_video(self, video_path: str, json_path: str, video_out_path: str) -> None:
        """
        processes the video, drawing bboxes and labels and writing to outputfile
        parameter:
            video_path: path to video
            json_path: path to json file
            video_out_path: path where the video will be saved
        """
        # input video
        video = cv2.VideoCapture(video_path)
        # output video
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(video.get(cv2.CAP_PROP_FPS))
        box_color = (255, 0, 0)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(video_out_path, fourcc, fps, (width, height))
        # json
        json_data = self._read_from_json(json_path)
        # iterate over frames, draw bboxes and labels and write to outputfile
        for frame_number, bboxes in enumerate(json_data["labels"]):
            ret, frame = video.read()
            if not ret:
                break
            for bbox in bboxes:
                center_x, center_y, w, h = (
                    int(bbox["center_x"] * width),
                    int(bbox["center_y"] * height),
                    int(bbox["w"] * width),
                    int(bbox["h"] * height),
                )
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
                label = str(bbox["id"])
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
            out.write(frame)

        video.release()
        out.release()

    def _compress_video(self, video_path: str, resolution: str = "1280x720") -> None:
        """
        compresses video to resolution
        parameter:
            video_path: path to video
            resolution: resolution of output video, default is 1280x720
        """
        compressed_file_path = os.path.join(os.path.splitext(video_path)[0] + "_c.mp4")
        subprocess.call(
            f"ffmpeg -i {video_path} -s {resolution} -acodec copy -y {compressed_file_path}",
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        os.remove(video_path)
        os.rename(compressed_file_path, video_path)
        # vscode can't download/open the file without the next 2 lines
        open_for_vscode_bugfix = cv2.VideoCapture(video_path)
        open_for_vscode_bugfix.release()

    def _read_from_json(self, path: str) -> Data:
        """
        reads data from json file
        parameter:
            path: path to json file
        return value:
            data: data from json file
        """
        with open(path, "r") as file:
            data = json.load(file)
        return data

    def _write_to_json(self, path: str, data: Data, negatives: Any) -> None:
        """
        writes bboxes and IDs to json file
        parameter:
            path: path to json file
            data: "labels" part with bboxes including ids for each frame
            negatives: list of IDs for each tracked ID, which can be used as negatives in tripletloss
        """
        id_data = {"tracked_IDs": [{"id": i, "negatives": list(s)} for i, s in enumerate(negatives)]}
        data = {**id_data, **data}
        with open(path, "w") as file:
            json.dump(data, file, indent=4)

    def _bboxes_overlap(
        self, bbox1: BBox, bbox2: BBox, allowed_overlap: float = 0.25, width: int = 1920, height: int = 1080
    ) -> bool:
        """
        check if bboxes overlap
        parameter:
            bbox1/bbox2: bboxes to check for
            allowed_overlap: % of the area of bboxes which is allowed to overlap between two individuals for still returning false, default is 0.25
            width/height: width and height of video, default is 1920*1080
        return value:
            overlap: boolean; True if bboxes overlap
        """
        resize = 1 - allowed_overlap

        # bbox1
        w1 = int(bbox1["w"] * resize * width)
        h1 = int(bbox1["h"] * resize * height)
        x1 = int(bbox1["center_x"] * width - w1 / 2)
        y1 = int(bbox1["center_y"] * height - h1 / 2)
        # bbox2
        w2 = int(bbox2["w"] * resize * width)
        h2 = int(bbox2["h"] * resize * height)
        x2 = int(bbox2["center_x"] * width - w2 / 2)
        y2 = int(bbox2["center_y"] * height - h2 / 2)

        overlap = not (x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1)
        return overlap

    def _track_ids(self, data: Data, ttl: int = 15) -> Tuple[Data, int]:
        """
        track individuals and create IDs; also removes bboxes when they overlap too much
        parameter:
            data: json data with bboxes for each frame of a video
        return values:
            data: json data of the video including IDs in each bbox
            id_count: int; number of tracked individuals
            ttl: int; how many frames an individual can be not detected before it is removed, default is 15
        """
        overlap = 0.5  # how much overlap is nessesary between to frames to be considered the same individual
        body_class = 0
        width_scale = (
            0.1  # how much the width of a bbox can change in % between frames to be considered the same individual
        )
        height_scale = (
            0.05  # how much the height of a bbox can change in % between frames to be considered the same individual
        )

        id_count = -1
        openIDs: List[Dict[str, int]] = []

        # iterate over frames in video
        for frame_data in data["labels"]:
            # iterate over bounding boxes and delete colliding ones
            bboxes = [bbox for bbox in frame_data if bbox["class"] == body_class]
            colliding_bboxes = [
                bbox1
                for bbox1 in bboxes
                for bbox2 in bboxes
                if bbox1 != bbox2 and self._bboxes_overlap(bbox1, bbox2, self.allowed_overlap)
            ]
            for bbox in colliding_bboxes[:]:
                if bbox in frame_data:
                    frame_data.remove(bbox)

            # iterate over remaning bboxes and give IDs
            bboxes = [bbox for bbox in frame_data if bbox["class"] == body_class]
            for bbox in bboxes:
                # check if individual already detected
                for id in openIDs:
                    scale_match = ((1 - width_scale) * id["w"] <= bbox["w"] <= (1 + width_scale) * id["w"]) or (
                        (1 - height_scale) * id["h"] <= bbox["h"] <= (1 + height_scale) * id["h"]
                    )
                    if self._bboxes_overlap(bbox, id, allowed_overlap=overlap) and scale_match:
                        bbox["id"] = id["id"]
                        id.update(
                            center_x=bbox["center_x"], center_y=bbox["center_y"], w=bbox["w"], h=bbox["h"], ttl=ttl
                        )
                        break

                # new individual
                if "id" not in bbox:
                    id_count += 1
                    bbox["id"] = id_count
                    openIDs.append(
                        dict(
                            id=id_count,
                            center_x=bbox["center_x"],
                            center_y=bbox["center_y"],
                            w=bbox["w"],
                            h=bbox["h"],
                            ttl=ttl,
                        )
                    )

            # decrease ttl (and remove) not tracked IDs from list
            for id in openIDs:
                id["ttl"] -= 1
            openIDs = [openID for openID in openIDs if openID["ttl"] > 0]

        return data, id_count

    def _label_faces(self, data: Data) -> Data:
        """
        label the face bboxes according to the already labeled body bboxes
        parameter:
            data: json data with bboxes and body IDs for each frame of a video
        return value:
            data: json data of the video including IDs for the face bboxes
        """
        body_class = 0
        face_class = 1

        for frame_data in data["labels"]:
            body_bboxes = [bbox for bbox in frame_data if bbox["class"] == body_class]
            face_bboxes = [bbox for bbox in frame_data if bbox["class"] == face_class]
            for face_bbox in face_bboxes:
                if len(body_bboxes) == 0:
                    frame_data.remove(face_bbox)
                    continue
                for body_bbox in body_bboxes:
                    if self._bboxes_overlap(face_bbox, body_bbox, allowed_overlap=0):
                        face_bbox["id"] = body_bbox["id"]
                if "id" not in face_bbox:
                    frame_data.remove(face_bbox)

        return data

    def _get_negatives(self, data: Data, id_count: int) -> List[MutableSet[str]]:
        """
        creates a list of lists which stores IDs which are possible negatives, because they existed at the same time
        parameter:
            data: json data with bboxes and IDs for each frame of a video
        return value:
            negatives: list of lists; at negatives[ID] stores a list of IDs which are negatives for ID
        """
        body_class = 0
        negatives = [set() for i in range(id_count + 1)]

        for frame_data in data["labels"]:
            bboxes = [bbox for bbox in frame_data if bbox["class"] == body_class]
            frame_ids = set()
            for bbox in bboxes:
                frame_ids.add(bbox["id"])
            for id in frame_ids:
                for frame_id in frame_ids:
                    if frame_id != id:
                        negatives[id].add(frame_id)

        return negatives
