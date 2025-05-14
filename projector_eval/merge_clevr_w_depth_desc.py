import json
import os

from PIL import Image
import pandas as pd
from datasets import Dataset, Image
import os

def load_clevr_scenes(split="train", data_path="/vast/yw4142/datasets/CLEVR_CoGenT_v1.0"):
    """
    Args:
        split (str): One of ['train', 'val'].
        data_path (str): Path to CLEVR dataset root.

    Returns:
        scenes (list): List of scene dictionaries.
    """
    assert split in ["trainA", "valA", "valB", "testA", "testB"], "Invalid split. Expected one of ['trainA', 'valA', 'valB', 'testA', 'testB']."
    
    scene_file = os.path.join(data_path, "scenes", f"CLEVR_{split}_scenes.json")
    
    with open(scene_file, "r") as f:
        scene_data = json.load(f)
    
    scenes = scene_data.get("scenes", [])
    return scenes

def print_scene_example(scene):
    keys_to_check = ['image_index', 'objects', 'relationships', 'image_filename', 'split', 'directions']
    
    for key in keys_to_check:
        print(f"\n🔑 {key}:")
        value = scene.get(key)
        
        if isinstance(value, list):
            for i, item in enumerate(value[:3]):  # Print up to 3 elements
                print(f"  [{i}] {item}")
        elif isinstance(value, dict):
            for k, v in list(value.items())[:3]:  # Print up to 3 items
                print(f"  {k}: {v}")
        else:
            print(value)

def describe_clevr_scene_structured(scene):
    objects = scene["objects"]
    relations = scene["relationships"]
    filename = scene["image_filename"]

    # Concise object descriptions with IDs
    obj_descriptions = [
        f"{i}: {obj['size']} {obj['color']} {obj['material']} {obj['shape']}"
        for i, obj in enumerate(objects)
    ]
    obj_line = "Objects: " + "; ".join(obj_descriptions) + "."

    # All relations as triples
    rel_triples = []
    for direction in ['left', 'right', 'front', 'behind']:
        for i, related in enumerate(relations.get(direction, [[]]*len(objects))):
            for j in related:
                rel_triples.append(f"{i} {direction} {j}")
    rel_line = "Relations: " + ", ".join(rel_triples) + "."

    return f"Scene:\n{obj_line}\n{rel_line}"

def describe_clevr_scene_2_direction(scene):
    objects = scene["objects"]
    filename = scene["image_filename"]

    # Build labeled object descriptions
    obj_descs = [
        (i, obj, f"{i}: {obj['size']} {obj['color']} {obj['material']} {obj['shape']}")
        for i, obj in enumerate(objects)
    ]

    # Sort by x (left-to-right)
    left_to_right = sorted(obj_descs, key=lambda x: x[1]["3d_coords"][0])
    left_line = "Left to right: " + "; ".join([desc for _, _, desc in left_to_right]) + "."

    # Sort by y (front-to-back)
    front_to_back = sorted(obj_descs, key=lambda x: x[1]["3d_coords"][1])
    front_line = "Front to back: " + "; ".join([desc for _, _, desc in front_to_back]) + "."

    return f"Scene:\n{left_line}\n{front_line}"

def describe_clevr_scene_nlp(scene):
    objects = scene["objects"]
    filename = scene["image_filename"]

    # Build object descriptions
    obj_descs = [
        (i, obj, f"a {obj['size']} {obj['color']} {obj['material']} {obj['shape']}")
        for i, obj in enumerate(objects)
    ]

    # Sort by x for left-to-right
    left_to_right = sorted(obj_descs, key=lambda x: x[1]["3d_coords"][0])
    left_sentences = [
        f"{left_to_right[i][2]} is to the left of {left_to_right[i+1][2]}"
        for i in range(len(left_to_right) - 1)
    ]

    # Sort by y for front-to-back
    front_to_back = sorted(obj_descs, key=lambda x: x[1]["3d_coords"][1])
    front_sentences = [
        f"{front_to_back[i][2]} is in front of {front_to_back[i+1][2]}"
        for i in range(len(front_to_back) - 1)
    ]

    intro = f"The image contains {len(objects)} objects."
    return f"{intro} " + ". ".join(left_sentences + front_sentences) + ". "

def load_clevr_questions(split="valB", data_path="/vast/yw4142/datasets/CLEVR_CoGenT_v1.0"):
    q_file = os.path.join(data_path, "questions", f"CLEVR_{split}_questions.json")
    with open(q_file, "r") as f:
        questions_data = json.load(f)
    return questions_data.get("questions", [])

def build_hf_rows(train_scenes, questions):
    q_by_index = {q['image_index']: q for q in questions}
    valid_answers = set(map(str, range(11))) | {"yes", "no"}
    rows = []

    for scene in train_scenes:
        idx = scene['image_index']
        image_path = os.path.join(image_base, scene['image_filename'])
        if not os.path.exists(image_path):
            continue

        q_entry = q_by_index.get(idx)
        if q_entry is None:
            continue

        answer = q_entry.get("answer", "").lower()
        if answer not in valid_answers:
            continue

        row = {
            "image": image_path,  # Use the image path instead of loading the image
            "question": q_entry["question"],
            "answer": answer,
            "description_structured": describe_clevr_scene_structured(scene),
            "description_ordered": describe_clevr_scene_2_direction(scene),
            "description_nlp": describe_clevr_scene_nlp(scene),
        }
        rows.append(row)

    return rows

def upload_dataset(train_scenes, questions, dataset_name):
    hf_rows = build_hf_rows(train_scenes, questions)
    df = pd.DataFrame(hf_rows)
    
    # Define the dataset with the Image feature
    dataset = Dataset.from_pandas(df)
    dataset = dataset.cast_column("image", Image())

    # Upload to Hugging Face Hub
    dataset.push_to_hub(dataset_name, private=True)

# Main execution
image_base = "/vast/yw4142/datasets/CLEVR_CoGenT_v1.0/images/valB"
train_scenes = load_clevr_scenes("valB")
questions = load_clevr_questions("valB")

upload_dataset(train_scenes, questions, "Yvonne511/Clevr_CoGenT_ValB_w_depth_prompt")

# print(scenes[0].keys())
# # print(describe_clevr_scene(train_scenes[1]))

# print("NLP Description")
# print(describe_clevr_scene_nlp(scenes[0]))
# print("Structured Description")
# print(describe_clevr_scene_structured(scenes[0]))
# print("Ordered Description")
# print(describe_clevr_scene_2_direction(scenes[0]))

