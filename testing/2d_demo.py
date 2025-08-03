import argparse
import os
import sys
import numpy as np
import json
import torch
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import base64
import math
import time
import warnings
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def load_qwen_model():
    """
    Load and initialize the Qwen2.5-VL-3B-Instruct model.
    
    Returns:
        tuple: (model, processor) - The loaded Qwen model and processor
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct", 
        torch_dtype="auto", 
        device_map="auto", 
        quantization_config=bnb_config,
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    return model, processor

def get_boundingbox_center(box):
    """Return the (x, y) center of a box given as (x_min, y_min, x_max, y_max)."""
    x_min, y_min, x_max, y_max = box
    return ((x_min + x_max) / 2, (y_min + y_max) / 2)

def get_box_area(box):
    """Compute the area of a bounding box.

    Args:
        box (Sequence[float]): (x_min, y_min, x_max, y_max).

    Returns:
        float: Area in pixel units.
    """
    x_min, y_min, x_max, y_max = box
    return (x_max - x_min) * (y_max - y_min)

def feedback_contact(model, processor, video_path, action, grid_size, total_frames, frame_start, max_feedbacks, search_anchor, speed_folder):
    """
    Improves the first-contact frame by asking the vision-language model again.

    Args:
        model: Qwen model instance
        processor: Qwen processor instance
        video_path (str): Input video.
        action (str): Action description, e.g. "Grasping the object".
        grid_size (int): Grid size used to build query images.
        total_frames (int): Total frame count of the video.
        frame_start (int): Initial contact guess.
        max_feedbacks (int): How many times to re-ask.
        search_anchor (str): "start" anchor for contact.
        speed_folder (str): Folder holding *_speed.json files.

    Returns:
        int | None: Final contact frame index or None if not found.
    """
    feedback_count = 0
    frame_contact = frame_start
    while feedback_count < max_feedbacks:
        corrected_frame = determine_by_state(model, processor, video_path, action, grid_size, total_frames, frame_contact, search_anchor, speed_folder)
        if corrected_frame is None:
            return None
        if corrected_frame == frame_contact:
            corrected_speed_frame = determine_by_speed(model, processor, video_path, action, grid_size, total_frames, frame_contact, search_anchor, speed_folder)
            if corrected_speed_frame is None:
                return None
            if corrected_speed_frame == frame_contact:
                return frame_contact
            else:
                frame_contact = corrected_speed_frame
        else:
            frame_contact = corrected_frame
        feedback_count += 1
    return frame_contact

def feedback_separation(model, processor, video_path, action, grid_size, total_frames, frame_end, max_feedbacks, search_anchor, speed_folder):
    """
    Improve the separation (end) frame using the same feedback logic as
    *feedback_contact* but looking for the end of the action.

    Args:
        model: Qwen model instance
        processor: Qwen processor instance
        video_path (str): Input video.
        action (str): Action description.
        grid_size (int): Grid size for grid image.
        total_frames (int): Frame count.
        frame_end (int): Initial separation guess.
        max_feedbacks (int): How many feedback loops.
        search_anchor (str): "end" anchor for separation.
        speed_folder (str): Folder with speed JSONs.

    Returns:
        int | None: Final separation frame index or None if not found.
    """
    feedback_count = 0
    frame_separate = frame_end
    while feedback_count < max_feedbacks:
        corrected_frame = determine_by_state(model, processor, video_path, action, grid_size, total_frames, frame_separate, search_anchor, speed_folder)
        if corrected_frame is None:
            return None
        if corrected_frame == frame_separate:
            corrected_speed_frame = determine_by_speed(model, processor, video_path, action, grid_size, total_frames, frame_separate, search_anchor, speed_folder)
            if corrected_speed_frame is None:
                return None
            if corrected_speed_frame == frame_separate:
                return frame_separate
            else:
                frame_separate = corrected_speed_frame
        else:
            frame_separate = corrected_frame
        feedback_count += 1
    return frame_separate

def convert_video(video_path, action, model, processor, grid_size, speed_folder, max_feedbacks, repeat_times=5):
    """
    Run several trials to localize contact and separation frames, then average.

    Args:
        video_path (str): Path to the input video.
        action (str): Action description shown to the model.
        model: Qwen model instance
        processor: Qwen processor instance
        grid_size (int): Size of the frame grid fed to the model.
        speed_folder (str): Folder containing *_speed.json files.
        max_feedbacks (int): Max feedback loops per trial.
        repeat_times (int, optional): How many trials to run. Default is 5.

    Returns:
        Tuple[int, int]: (contact_frame, separation_frame). Each is a frame
        index or -1 if the event could not be determined.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    contact_list = []
    separate_list = []
    for trial_idx in range(repeat_times):

        print("----------> trial_idx: ", trial_idx)

        frame_start = process_task(model, processor, video_path, action, grid_size, total_frames, 'start', speed_folder)
        frame_end = process_task(model, processor, video_path, action, grid_size, total_frames, 'end', speed_folder)
        frame_contact = feedback_contact(model, processor, video_path, action, grid_size, total_frames, frame_start, max_feedbacks, 'start', speed_folder)
        frame_separate = feedback_separation(model, processor, video_path, action, grid_size, total_frames, frame_end, max_feedbacks, 'end', speed_folder)
        contact_list.append(frame_contact)
        separate_list.append(frame_separate)
    
    contact_list = [x for x in contact_list if x is not None and x != -1]
    separate_list = [x for x in separate_list if x is not None and x != -1]
    if len(contact_list) == 0:
        final_contact = -1
    else:
        final_contact = int(round(np.mean(contact_list)))
    if len(separate_list) == 0:
        final_separate = -1
    else:
        final_separate = int(round(np.mean(separate_list)))
    return final_contact, final_separate

def select_top_n_frames_from_json(json_path, n, frame_index=None, flag=None, receive_flag=None):
    """
    Select the *n* frames with the lowest hand speed from a JSON file.

    Args:
        json_path (str): Path to the *_speed.json file.
        n (int): Number of frames to return.
        frame_index (int, optional): Reference index or threshold.
        flag (str, optional): Behaviour switch ("feedback" or "speed").
        receive_flag (str, optional): If set, also return invalid frames.

    Returns:
        list[int] | Tuple[list[int], list[int]]: Top-n frame indices, with an
        optional list of invalid indices first when *receive_flag* is given.
    """
    with open(json_path, 'r') as file:
        data = json.load(file)
    items = list(data.items())
    if frame_index is None:
        valid_frames = [(int(index), speed) for index, speed in items if speed != 0.0 and not math.isnan(speed)]
    else:
        if flag == "feedback":
            valid_frames = [(int(index), speed) for index, speed in items if speed != 0.0 and not math.isnan(speed) and int(index) > frame_index]
            invalid_list = [int(index) for index, speed in items if speed != 0.0 and not math.isnan(speed) and int(index) <= frame_index]
        elif flag == "speed":
            valid_frames = [(int(index), speed) for index, speed in items if speed != 0.0 and not math.isnan(speed) and speed < frame_index]
            invalid_list = [int(index) for index, speed in items if speed != 0.0 and not math.isnan(speed) and speed >= frame_index]
        else:
            valid_frames = [(int(index), speed) for index, speed in items if speed != 0.0 and not math.isnan(speed) and int(index) != frame_index]
            invalid_list = [int(index) for index, speed in items if speed != 0.0 and not math.isnan(speed) and int(index) == frame_index]
    sorted_frames = sorted(valid_frames, key=lambda x: x[1])
    top_n_frames = [frame[0] for frame in sorted_frames[:n]]
    if receive_flag is None:
        return top_n_frames
    else:
        return invalid_list, top_n_frames

def process_task(
        model,
        processor,
        video_path,
        action,
        grid_size,
        total_frames,
        search_anchor,
        speed_folder,
        frame_index=None,
        flag=None
        ):
    """
    Build a grid image, ask Qwen to pick the frame where the action starts
    or ends, and return the chosen frame index.

    Args:
        model: Qwen model instance
        processor: Qwen processor instance
        video_path (str): Input video.
        action (str): Action text shown to the model.
        grid_size (int): Grid dimension (grid_size x grid_size frames shown).
        total_frames (int): Total number of frames in the video.
        search_anchor (str): "start" or "end" - whether we look for the beginning or ending.
        speed_folder (str): Folder containing *_speed.json files.
        frame_index (int | None, optional): If given, acts as a reference for feedback sampling.
        flag (str | None, optional): Behaviour modifier ("feedback", "speed", or None).

    Returns:
        int: The frame index selected by the model. Returns `None` if the model
        says the action is not present (-1 in its JSON response).
    """
    prompt_start = (
        f"I will show an image sequence of human cooking. "
        f"I have annotated the images with numbered circles. "
        f"Choose the number that is closest to the moment when the ({action}) has started. "
        f"You are a five-time world champion in this game. "
        f"Give a one sentence analysis of why you chose those points (less than 50 words). "
        f"If you consider that the action is not in the video, please choose the number -1. "
        f"Provide your answer at the end in a json file of this format: {{\"points\": []}}"
    )

    prompt_end = (
        f"I will show an image sequence of human cooking. "
        f"I have annotated the images with numbered circles. "
        f"Choose the number that is closest to the moment when the ({action}) has ended. "
        f"You are a five-time world champion in this game. "
        f"Give a one sentence analysis of why you chose those points (less than 50 words). "
        f"If you consider that the action has not ended yet, please choose the number -1. "
        f"Provide your answer at the end in a json file of this format: {{\"points\": []}}"
    ) 

    prompt_message = prompt_start if search_anchor == 'start' else prompt_end
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    json_path = get_json_path(video_name, base_dir=speed_folder)
    if frame_index is None:
        selected_indices = select_top_n_frames_from_json(json_path, 4)
        total_indices = select_top_n_frames_from_json(json_path, total_frames)
        filter_indices = select_and_filter_keyframes_with_anchor(selected_indices, total_indices, 4, search_anchor, video_path)
        if not filter_indices:
            filter_indices = sorted(selected_indices)
        used_frame_indices = select_frames_near_average(filter_indices, grid_size, total_frames, [])
        image = create_frame_grid_with_keyframe(
            video_path, used_frame_indices, grid_size)
        print(f'The frame indices in {video_name} to constuct the grid image: {used_frame_indices}')
    else:
        if flag == "feedback":
            print(f"Activate feedback mechanism with visual cues")
            invalid_list, selected_indices = select_top_n_frames_from_json(json_path, 4, frame_index, flag, receive_flag="right")
            total_indices = select_top_n_frames_from_json(json_path, total_frames, frame_index, flag)
            filter_indices = select_and_filter_keyframes_with_anchor(selected_indices, total_indices, 4, search_anchor, video_path)
            if not filter_indices:
                filter_indices = sorted(selected_indices)
            used_frame_indices = select_frames_near_average(filter_indices, grid_size, total_frames, invalid_list, min_index=frame_index)
            print(f'Resampled frame indices in {video_name} to constuct the grid image: {used_frame_indices}')
            image = create_frame_grid_with_keyframe(
                video_path, used_frame_indices, grid_size)
        elif flag == "speed":
            print(f"Activate feedback mechanism with dynamic cues")
            invalid_list, selected_indices = select_top_n_frames_from_json(json_path, 4, frame_index, flag, receive_flag="right")
            total_indices = select_top_n_frames_from_json(json_path, total_frames, frame_index, flag)
            filter_indices = select_and_filter_keyframes_with_anchor(selected_indices, total_indices, 4, search_anchor, video_path)
            if not filter_indices:
                filter_indices = sorted(selected_indices)
            used_frame_indices = select_frames_near_average(filter_indices, grid_size, total_frames, invalid_list)
            print(f'Resampled frame indices in {video_name} to constuct the grid image: {used_frame_indices}')
            image = create_frame_grid_with_keyframe(
                video_path, used_frame_indices, grid_size)
        else:
            print(f"{frame_index} is a frame without hands")
            invalid_list, selected_indices = select_top_n_frames_from_json(json_path, 4, frame_index, receive_flag="right")
            total_indices = select_top_n_frames_from_json(json_path, total_frames, frame_index, flag)
            filter_indices = select_and_filter_keyframes_with_anchor(selected_indices, total_indices, 4, search_anchor, video_path)
            if not filter_indices:
                filter_indices = sorted(selected_indices)
            used_frame_indices = select_frames_near_average(filter_indices, grid_size, total_frames, invalid_list)
            print(f'The frame indices in {video_name} to constuct the grid image: {used_frame_indices}')
            image = create_frame_grid_with_keyframe(
                video_path, used_frame_indices, grid_size)
    
    grid_image = Image.fromarray(image)
    description, reason = scene_understanding(
        model, processor, image, prompt_message)
    print("Localization results:", used_frame_indices[int(description)-1])
    if description:
        if description == -1:
            return None
        if int(description) - 1 > len(used_frame_indices) - 1:
            print("Warning: Invalid frame index selected")
            print(f"Selected frame index: {description}")
        index_specified = max(
            min(int(description) - 1, len(used_frame_indices) - 1), 0)
        selected_frame_index = used_frame_indices[index_specified]
    return selected_frame_index

def save_predictions(all_predictions, file_path):
    """Write the *all_predictions* list to *file_path* in JSON format."""
    with open(file_path, "w") as f:
        json.dump(all_predictions, f)

def load_predictions(file_path):
    """Load prediction list from *file_path*; create empty list if file missing."""
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return json.load(f)
    else:
        with open(file_path, "w") as f:
            json.dump([], f)
        return []

def select_frames_near_average(filter_indices, grid_size, total_frames, invalid_list, min_index=None):
    """
    Fill a grid with frame indices centered around the average of *filter_indices*.

    Args:
        filter_indices (list[int]): Candidate frame indices.
        grid_size (int): Grid side length.
        total_frames (int): Total frames in the video.
        invalid_list (list[int]): Indices that must be avoided.
        min_index (int, optional): Minimum allowed index (used in feedback).

    Returns:
        list[int]: Exactly `grid_size ** 2` frame indices ordered for the grid.
    """
    avg_index = round(np.mean(filter_indices))
    start_index, end_index = avg_index, avg_index
    used_frame_indices = []
    if avg_index not in invalid_list and (min_index is None or avg_index > min_index):
        used_frame_indices.append(avg_index)
    while len(used_frame_indices) < grid_size ** 2:
        if len(used_frame_indices) < grid_size ** 2:
            if start_index >= 0:
                if start_index > 0:
                    start_index -= 1
                    if start_index not in invalid_list and (min_index is None or start_index > min_index):
                        used_frame_indices.insert(0, start_index)
                if start_index == 0 and (min_index is None or start_index > min_index):
                    used_frame_indices.insert(0, start_index)
            if len(used_frame_indices) < grid_size ** 2 and end_index <= total_frames - 1:
                if end_index < total_frames - 1:
                    end_index += 1
                    if end_index not in invalid_list and (min_index is None or end_index > min_index):
                        used_frame_indices.append(end_index)
                if end_index == total_frames - 1 and (min_index is None or end_index > min_index):
                    used_frame_indices.append(end_index)
    used_frame_indices = used_frame_indices[:grid_size**2]
    return used_frame_indices

def select_and_filter_keyframes_with_anchor(selected_indices, total_indices, grid_size, search_anchor, video_path):
    """
    Keep keyframes that lie in the first or second half of the video.

    Args:
        selected_indices (list[int]): Main candidate frames.
        total_indices (list[int]): Backup frame pool.
        grid_size (int): Needed number of frames.
        search_anchor (str): "start" keeps frames before the midpoint, "end"
            keeps frames after the midpoint.
        video_path (str): Video to measure total length.

    Returns:
        list[int]: Sorted list of at most *grid_size* frame indices.
    """
    if not selected_indices:
        return []
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    if search_anchor == 'start':
        filtered_indices = [idx for idx in selected_indices if idx < total_frames // 2]
        if len(filtered_indices) < grid_size:
            num_needed = grid_size - len(filtered_indices)
            remaining_indices = [i for i in total_indices if i not in filtered_indices and i < total_frames // 2]
            for idx in remaining_indices:
                if len(filtered_indices) >= grid_size:
                    break
                filtered_indices.append(idx)
    elif search_anchor == 'end':
        filtered_indices = [idx for idx in selected_indices if idx >= total_frames // 2]
        if len(filtered_indices) < grid_size:
            num_needed = grid_size - len(filtered_indices)
            remaining_indices = [i for i in total_indices if i not in filtered_indices and i >= total_frames // 2]
            for idx in remaining_indices:
                if len(filtered_indices) >= grid_size:
                    break
                filtered_indices.append(idx)
    else:
        raise ValueError("search_anchor must be either 'start' or 'end'")
    filtered_indices_sorted = sorted(filtered_indices)
    return filtered_indices_sorted

def get_json_path(video_name, base_dir):
    """Return full path for the speed JSON of *video_name* in *base_dir*."""
    json_filename = f"{video_name}_speed.json"
    json_path = os.path.join(base_dir, json_filename)
    return json_path

def visualize_frame(video_path, frame_idx, output_path, label=None):
    """
    Save one frame from a video, with an optional text label.

    Args:
        video_path (str): Video file.
        frame_idx (int): Index of the frame to save.
        output_path (str): Where to write the image file.
        label (str, optional): If given, draw this text on the frame.
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        print(f"Warning: Frame {frame_idx} not found for visualization.")
        return
    if label is not None:
        cv2.putText(frame, label, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 4)
    cv2.imwrite(output_path, frame)
    cap.release()

def determine_by_state(model, processor, video_path, action, grid_size, total_frames, frame_index, search_anchor, speed_folder):
    """
    Verify if a chosen frame truly shows contact/separation; if not, try again.

    Args:
        model: Qwen model instance
        processor: Qwen processor instance
        video_path (str): Video file.
        action (str): Text describing the action.
        grid_size (int): Grid size used by *process_task*.
        total_frames (int): Total frame count.
        frame_index (int): Frame under examination.
        search_anchor (str): "start" or "end".
        speed_folder (str): Folder with speed JSONs.

    Returns:
        int | None: Confirmed or corrected frame index, or None if failed.
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    frame_indices = []
    flag = "feedback"
    frame_indices.append(frame_index)
    prompt_start = (
        f"I will show an image of hand-object interaction."
        f"You need to help me determine whether the hand and the object in the current image are in obvious contact rather than just appearing to be in contact "
        f"If yes, answer 1; if no, answer 0 "
    )
    prompt_end = (
        f"I will show an image of hand-object interaction."
        f"You need to help me determine whether the hand and the object in the current image are in obvious seperate rather than just appearing to be in seperate "
        f"If yes, answer 1; if no, answer 0 "
    )
    prompt_message = prompt_start if search_anchor == 'start' else prompt_end
    image = create_frame_grid_with_keyframe(video_path, frame_indices, 1)
    result = scene_understanding(model, processor, image, prompt_message, flag)
    if result == "1" or frame_index > total_frames - 5:
        return frame_index
    else:
        frame_improved = process_task(model, processor, video_path, action, grid_size, total_frames, search_anchor, speed_folder, frame_index, flag)
        return frame_improved

def determine_by_speed(model, processor, video_path, action, grid_size, total_frames, frame_index, search_anchor, speed_folder):
    """
    Check if the hand moves slowly enough at *frame_index*; if not, re-sample.

    Args:
        model: Qwen model instance
        processor: Qwen processor instance
        (other args same as determine_by_state)

    Returns:
        int | None: Accepted or corrected frame index, or None on failure.
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    json_path = get_json_path(video_name, speed_folder)
    with open(json_path, 'r') as file:
        data = json.load(file)
    valid_frames = [
        (int(index), speed) for index, speed in data.items() if speed != 0.0 and not math.isnan(speed)
    ]
    sorted_frames = sorted(valid_frames, key=lambda x: x[1])
    frame_speed = next((speed for index, speed in valid_frames if index == frame_index), None)
    if frame_speed is None:
        frame_improved = process_task(model, processor, video_path, action, grid_size, total_frames, search_anchor, speed_folder, frame_index, flag="fault")
        return frame_improved
    threshold_index = int(total_frames * 0.3)
    speed_threshold = sorted_frames[threshold_index - 1][1] if threshold_index > 0 else sorted_frames[0][1]
    if frame_speed <= speed_threshold:
        return frame_index
    else:
        frame_improved = process_task(model, processor, video_path, action, grid_size, total_frames, search_anchor, speed_folder, frame_speed, flag="speed")
        return frame_improved

def image_resize_for_vlm(frame, inter=cv2.INTER_AREA):
    """Resize an image so that the shorter side ≤ 768 and longer side ≤ 2000."""
    height, width = frame.shape[:2]
    aspect_ratio = width / height
    max_short_side = 768
    max_long_side = 2000
    if aspect_ratio > 1:
        new_width = min(width, max_long_side)
        new_height = int(new_width / aspect_ratio)
        if new_height > max_short_side:
            new_height = max_short_side
            new_width = int(new_height * aspect_ratio)
    else:
        new_height = min(height, max_long_side)
        new_width = int(new_height * aspect_ratio)
        if new_width > max_short_side:
            new_width = max_short_side
            new_height = int(new_width / aspect_ratio)
    resized_frame = cv2.resize(
        frame, (new_width, new_height), interpolation=inter)
    return resized_frame

def extract_json_part(text_output):
    """Extract the JSON string like {"points": [...]} from text output."""
    text = text_output.strip().replace(" ", "").replace("\n", "")
    try:
        start = text.index('{"points":')
        text_json = text[start:].strip()
        end = text_json.index('}') + 1
        text_json = text_json[:end].strip()
        return text_json
    except ValueError:
        print("Text received:", text_output)
        return None

def scene_understanding(model, processor, frame, prompt_message, flag=None):
    """
    Send an image and text prompt to Qwen model and parse the reply.

    Args:
        model: Qwen model instance
        processor: Qwen processor instance
        frame (np.ndarray): BGR image array.
        prompt_message (str): Instruction for the model.
        flag (str | None): If `None`, parse and return the first point; when
            not `None`, just return the full text answer.

    Returns:
        Tuple[int, str] | str:
            • When *flag* is `None`: (chosen_point, full_text_response).
            • Otherwise: the full text response.
    """
    frame = image_resize_for_vlm(frame)
    
    # Convert frame to PIL Image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    
    # Save temporary image for Qwen processing
    temp_image_path = "temp_frame.jpg"
    pil_image.save(temp_image_path)
    
    # Prepare messages for Qwen
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": temp_image_path,
                },
                {"type": "text", "text": prompt_message},
            ],
        }
    ]
    
    # Process with Qwen
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    
    # Generate response
    generated_ids = model.generate(
        **inputs, 
        max_new_tokens=200,
        temperature=0.1,
        no_repeat_ngram_size=3,
    )
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    response_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    # Clean up temp file
    if os.path.exists(temp_image_path):
        os.remove(temp_image_path)
    
    if flag is None:
        response_json = extract_json_part(response_text)
        if response_json is None:
            return -1, response_text
        else:
            json_dict = json.loads(response_json, strict=False)
            if len(json_dict['points']) == 0:
                return -1, response_text
            if len(json_dict['points']) > 1:
                print("Warning: More than one point detected")
            return json_dict['points'][0], response_text
    else:
        return response_text

def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    """Resize *image* while keeping aspect ratio."""
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized

def create_frame_grid(video_path, center_time, interval, grid_size):
    """Build a square grid of frames sampled around *center_time*."""
    spacer = 0
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    center_frame = int(center_time * fps)
    interval_frames = int(interval * fps)
    num_frames = grid_size**2
    half_num_frames = num_frames // 2
    frame_indices = [max(0,
                         min(center_frame + i * interval_frames,
                             total_frames - 1)) for i in range(-half_num_frames,
                                                               half_num_frames + 1)]
    frames = []
    actual_indices = []
    for index in frame_indices:
        video.set(cv2.CAP_PROP_POS_FRAMES, index)
        success, frame = video.read()
        if success:
            frame = image_resize(frame, width=200)
            frames.append(frame)
            actual_indices.append(index)
        else:
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            success, frame = video.read()
            frame = image_resize(frame, width=200)
            frame = frame * 0
            frames.append(frame)
            actual_indices.append(index)
    video.release()
    if len(frames) < grid_size**2:
        raise ValueError("Not enough frames to create the grid.")
    frame_height, frame_width = frames[0].shape[:2]
    grid_height = grid_size * frame_height + (grid_size - 1) * spacer
    grid_width = grid_size * frame_width + (grid_size - 1) * spacer
    grid_img = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255
    for i in range(grid_size):
        for j in range(grid_size):
            index = i * grid_size + j
            frame = frames[index]
            cX, cY = frame.shape[1] // 2, frame.shape[0] // 2
            max_dim = int(min(frame.shape[:2]) * 0.5)
            overlay = frame.copy()
            circle_center = (frame.shape[1] - max_dim // 2, max_dim // 2)
            cv2.circle(overlay, circle_center,
                       max_dim // 2, (255, 255, 255), -1)
            alpha = 0.3
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            cv2.circle(frame, circle_center, max_dim // 2, (255, 255, 255), 2)
            font_scale = max_dim / 50
            text_size = cv2.getTextSize(
                str(index + 1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
            text_x = frame.shape[1] - text_size[0] // 2 - max_dim // 2
            text_y = text_size[1] // 2 + max_dim // 2
            cv2.putText(frame, str(index + 1), (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2)
            y1 = i * (frame_height + spacer)
            y2 = y1 + frame_height
            x1 = j * (frame_width + spacer)
            x2 = x1 + frame_width
            grid_img[y1:y2, x1:x2] = frame
    return grid_img

def create_frame_column_with_keyframe(video_path, frame_indices, num_columns=3):
    """Assemble a horizontal strip of frames given by *frame_indices*."""
    spacer = 0
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for index in frame_indices:
        video.set(cv2.CAP_PROP_POS_FRAMES, index)
        success, frame = video.read()
        if success:
            frame = image_resize(frame, width=200)
            frames.append(frame)
        else:
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            success, frame = video.read()
            frame = image_resize(frame, width=200)
            frame = frame * 0
            frames.append(frame)
    video.release()
    if len(frames) < num_columns:
        missing_frames = num_columns - len(frames)
        black_frame = np.zeros_like(frames[0])
        frames.extend([black_frame] * missing_frames)
    frame_height, frame_width = frames[0].shape[:2]
    grid_width = num_columns * frame_width + (num_columns - 1) * spacer
    grid_height = frame_height
    grid_img = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255
    for j in range(num_columns):
        index = j
        frame = frames[index]
        y1 = 0
        y2 = frame_height
        x1 = j * (frame_width + spacer)
        x2 = x1 + frame_width
        grid_img[y1:y2, x1:x2] = frame
    return grid_img

def create_frame_grid_with_keyframe(video_path, frame_indices, grid_size):
    """Build a grid image from *frame_indices* and draw numbered circles."""
    spacer = 0
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for index in frame_indices:
        video.set(cv2.CAP_PROP_POS_FRAMES, index)
        success, frame = video.read()
        if success:
            frame = image_resize(frame, width=200)
            frames.append(frame)
        else:
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            success, frame = video.read()
            frame = image_resize(frame, width=200)
            frame = frame * 0
            frames.append(frame)
    video.release()
    if len(frames) < grid_size**2:
        missing_frames = grid_size**2 - len(frames)
        black_frame = np.zeros_like(frames[0])
        frames.extend([black_frame] * missing_frames)
    frame_height, frame_width = frames[0].shape[:2]
    grid_height = grid_size * frame_height + (grid_size - 1) * spacer
    grid_width = grid_size * frame_width + (grid_size - 1) * spacer
    grid_img = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255
    for i in range(grid_size):
        for j in range(grid_size):
            index = i * grid_size + j
            frame = frames[index]
            cX, cY = frame.shape[1] // 2, frame.shape[0] // 2
            max_dim = int(min(frame.shape[:2]) * 0.5)
            overlay = frame.copy()
            circle_center = (frame.shape[1] - max_dim // 2, max_dim // 2)
            cv2.circle(overlay, circle_center,
                       max_dim // 2, (255, 255, 255), -1)
            alpha = 0.3
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            cv2.circle(frame, circle_center, max_dim // 2, (255, 255, 255), 2)
            font_scale = max_dim / 50
            text_size = cv2.getTextSize(
                str(index + 1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
            text_x = frame.shape[1] - text_size[0] // 2 - max_dim // 2
            text_y = text_size[1] // 2 + max_dim // 2
            cv2.putText(frame, str(index + 1), (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2)
            y1 = i * (frame_height + spacer)
            y2 = y1 + frame_height
            x1 = j * (frame_width + spacer)
            x2 = x1 + frame_width
            grid_img[y1:y2, x1:x2] = frame
    return grid_img

if __name__ == "__main__":
    parser = argparse.ArgumentParser("EgoLoc 2D Feedback Demo with Qwen")
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video")
    parser.add_argument("--output_dir", type=str, required=False, default="output", help="Output directory")
    parser.add_argument("--speed_folder", type=str, required=True, help="Folder containing speed JSON files")
    parser.add_argument("--action", type=str, default="Grasping the object", help="Action label")
    parser.add_argument("--grid_size", type=int, default=3, help="Grid size for localization")
    parser.add_argument("--max_feedbacks", type=int, default=1, help="Maximum feedback loops")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load Qwen model
    print("\n [1/3] Loading Qwen model ...")
    model, processor = load_qwen_model()
    
    # Temporal interaction localization
    print("\n [2/3] Locating contact/separation frames and visualizing ...")
    frame_contact, frame_separate = convert_video(
        args.video_path, args.action, model, processor, args.grid_size, args.speed_folder, args.max_feedbacks, repeat_times=3
    )
    video_name = os.path.splitext(os.path.basename(args.video_path))[0]
    contact_vis_path = os.path.join(args.output_dir, f"{video_name}_contact_frame.png")
    separation_vis_path = os.path.join(args.output_dir, f"{video_name}_separation_frame.png")
    visualize_frame(args.video_path, frame_contact, contact_vis_path, label="Contact")
    visualize_frame(args.video_path, frame_separate, separation_vis_path, label="Separation")

    # Save TIL results
    print("\n [3/3] Saving results ...") 
    result = {
        "contact_frame": frame_contact,
        "separation_frame": frame_separate
    }
    print("Egoloc output \n", result)
    result_path = os.path.join(args.output_dir, f"{video_name}_result.json")
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Result json: {result_path}\nContact frame vis: {contact_vis_path}\nSeparation frame vis: {separation_vis_path}")
