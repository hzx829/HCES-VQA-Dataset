import sys
import openai
from openai import OpenAI
import cv2
#use openai to connect oneapi
client = OpenAI(api_key="sk-ryAHHlMGV5o8tWEP9887005720A6421dBcC38cFb4939AdDa", base_url="http://47.251.49.91/v1")

dataset_path = './dataset_v1/'
# prompt
common_question_map = {
    "is_person_show": "Is the whole head of the worker can be clearly seen?",
    "wearing_helmets": "Are all the workers in this image wearing safety helmets?[Sensitive]",
    "wearing_clothes": "Are all the workers wearing long-sleeve work clothes without rolling up the sleeves?",
    "form_filled": "If a form appears in the image, are all required fields filled, and are there any alterations?",
    "not_smoking": "Are all the workers in the image not smoking?[Sensitive]",
    "safety_line": "Is the work area enclosed by a safety line(red)?",
}

high_position_question_map = {
    # "is_high_position": "Is any workers climbing or at high position?",
    "is_high_position": "Does this image show Working at Heights(more than 3 meters high)?",
    "not_carry_heavy": "Is the worker(climbing or at high position) carrying nothing heavy?",
    "wearing_harnesses": "Is the worker(climbing or at high position) wearing safety harnesses?",
    "hook_attached": "Is there at least one safety hook on the worker(climbing or at high position) attached to solid objects?[Sensitive]",
}

fire_question_map = {
    "is_hot_work": "Does this image show a hot work?",
    # "flammable_debris": "Is there any flammable debris on the work surface?",
    "firefighting": "Are there firefighting equipment available on the work surface?",
    "wearing_gloves": "Is the operator wearing gloves(white) properly?",
    # "acetylene_cylinder": "Is there any acetylene cylinder (brown, single) in this image?",
    # "oxygen_cylinder": "Is there any oxygen cylinder (blue, single) in this image?",
    "worker_located_properly": "Is the operator located more than 10 meters away from both the oxygen cylinder (blue, single) and the acetylene cylinder (brown, single)?[Sensitive]",
    "cylinder_located_properly": "Is the distance between the oxygen cylinder (blue, single) and the acetylene cylinder (brown, single) more than 5 meters?[Sensitive]",
    "acetylene_cylinder_stand_up": "Does the acetylene cylinder (brown, single) stand up?"
}

confined_space_question_map = {
    "is_confined_space_work": "Does this image show a confined space work/entry?",
    "supervisor": "Is there any supervisor in this image?",
    "wearing_rope": "Did the operator secure and fasten the safety ropes tied to his body?[Sensitive]",
    "holding_rope": "Is the supervisor holding the safety ropes?",
    "is_entrance_clear": "Is the entrance to the confined space free of clutter and debris?"
}

lifting_question_map = {
    "is_lifting_operations": "Does this image show a lifting operations?",
    "is_cabin_empty": "Is there anyone inside the car cabin during hoisting?",
    "is_object_secured": "Is the heavy object securely bound, fastened, and balanced before hoisting?",
    "is_under_hook_clear": "Is the area under the crane hook or hoisted object clear of people and loose objects?",
    "is_straight_lift": "Is the heavy object being hoisted straight and balanced without any diagonal pulling?[Sensitive]",
    "is_safety_rope_used": "Are safety ropes used at both ends to control the balance of the hoisted object?",
    "is_proper_anchor_used": "Are proper anchor points used instead of pipelines, pipe racks, electric poles, or mechanical equipment?",
    "is_no_suspended_loads": "Are there no hoisted objects, cages, tools, or slings left suspended in the air during work stoppages and breaks?",
    "is_safe_distance_maintained": "Is a safe distance maintained from the hoisted object during positioning, using stretchers, poles, or hooks to assist?",
    "is_hook_properly_used": "Is the hoisting hook not directly wrapped around the heavy object for lifting?",
    # "is_caution_area_clear": "Are non-working personnel prohibited from entering the work caution area?"
}

dig_question_map = {
    "is_dig_work": "Does this image show a Excavation and Trenching",
    "is_material_storage_safe": "Are the materials and excavated soil stored at least 1 meter away from the edge of pits, trenches, wells, "
                                "and ditches, and is the height of the soil piles not greater than 1.5 meters?",
    "is_resting_in_trenches_prohibited": "Are workers not resting inside or near pits, trenches, wells, or ditches?"
}

road_closure_question_map = {
    "is_road_closure_operations": "Does this image show a road closure operations",
    "is_barrier": "Is there any cones and barrier tape?",
}

electrical_question_map = {
    "is_electrical_work": "Does this image show a Temporary Electrical Work?",
    "are_wires_not_on_branches": "Are the electrical wires not hanging on tree branches?",
    "are_wires_hung": "Are the wires hung more than 1.5 meters above the ground?"
}

# 盲板:OCR识别后,先假定盲板编号为352/353
blanking_question_map = {
    "is_blanking_work": "Does this image show a Line Blanking or Blinding?",
    "is_number_show": "Is there any number on this image?",
}

system_prompt = ('You are an AI specialized in generating prompts for a Safety Engineering AI Agent. Your primary '
                 'responsibility is to interpret images to detect safety violations in industrial settings. Here’s '
                 'how you should operate:\n\nGuidelines:\n1. Analyze the Image: The scene in the image depicts one or '
                 'more of the eight high-risk operations. Thoroughly examine each provided'
                 'image to identify potential safety hazards or violations. Pay close attention to the elements '
                 'within the image that might indicate unsafe practices or non-compliance with safety '
                 'regulations.\n\n2. Answer Questions: Based on your image analysis, answer these questions one by one. '
                 'You should answer true or false if you are confident about your answer. If you not sure '
                 'about your answer(because of critical parts are not visible, obscured), answer with null.'
                 'However, if the question description has [Sensitive] Tag, you can answer true or false even you are not that sure'
                 ' and tend to find violation in the image.'
                 'The questions format is: wearing_helmets: Are all the workers in this image wearing safety helmets?. '
                 'Output the JSON format with key: "wearing_helmets"(True/False/None)')

def generate_safety_prompt(question_map):
    base_prompt = ""
    for key, question in question_map.items():
        base_prompt += f"\n{key}:{question}"
    return base_prompt


import vertexai
from vertexai.generative_models import GenerativeModel, Image, HarmCategory, HarmBlockThreshold

# Initialize Vertex AI
vertexai.init(project='koppieos', location='us-central1')
# Load the model
# from vertexai. import HarmBlockThreshold, HarmCategory
# gpt-4o #gemini-flash-1.5
import base64

import PIL.Image


def detect_violation_frame(img_path, model='gpt-4o'):
    detect_prompt = (generate_safety_prompt(common_question_map) +
                     generate_safety_prompt(high_position_question_map) +
                     generate_safety_prompt(fire_question_map) +
                     generate_safety_prompt(confined_space_question_map) +
                     generate_safety_prompt(lifting_question_map) +
                     generate_safety_prompt(dig_question_map) +
                     generate_safety_prompt(road_closure_question_map) +
                     generate_safety_prompt(electrical_question_map) +
                     generate_safety_prompt(blanking_question_map)
                     )

    print("path:", dataset_path + img_path)
    frame = cv2.imread(dataset_path + img_path)
    ret, buffer = cv2.imencode('.jpeg', frame)

    if not ret:
        print("Could not encode frame to JPEG")

    base64_string = base64.b64encode(buffer).decode('utf-8')
    print("deocode:", dataset_path + img_path)

    if model.find('gemini') != -1:
        print("gemini")
        # sample_file = genai.upload_file(path=dataset_path+img_path,
        #                     display_name=dataset_path+img_path)
        # # file = genai.get_file(name=sample_file.name)
        # print(f"Uploaded file '{sample_file.display_name}' as: {sample_file.uri}")
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
            # HarmCategory.HARM_CATEGORY_DEROGATORY: HarmBlockThreshold.BLOCK_NONE,
            # HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE
            # HarmCategory.HARM_CATEGORY_DANGEROUS: HarmBlockThreshold.BLOCK_NONE,
            # HarmCategory.HARM_CATEGORY_VIOLENCE: HarmBlockThreshold.BLOCK_NONE
        }
        # Choose a Gemini API model.
        if model == 'gemini-flash-1.5':
            model = GenerativeModel(model_name="gemini-1.5-flash",
                                          generation_config={"response_mime_type": "application/json"},
                                          safety_settings=safety_settings)
        elif model == 'gemini-pro-1.5':
            model = GenerativeModel(model_name="gemini-1.5-pro",
                                          generation_config={"response_mime_type": "application/json"},
                                          safety_settings=safety_settings)

        img = Image.load_from_file(dataset_path + img_path)
        # Prompt the model with text and the previously uploaded image.
        print(model.count_tokens([system_prompt, detect_prompt, img]))
        response = model.generate_content([system_prompt, detect_prompt, img])
        print(response.usage_metadata)
        return response.text
    else:
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": system_prompt
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": 'data:image/jpeg;base64,{}'.format(base64_string),
                            "detail": "high"
                        },
                    },
                    {
                        "type": "text",
                        "text": detect_prompt,
                    }
                ],
            }
        ]

        try:
            response = client.chat.completions.create(
                model=model,
                response_format={"type": "json_object"},
                messages=messages,
                temperature=0,
                max_tokens=4095,
                top_p=1
            )
        except openai.BadRequestError as e:
            print("error")

        return response.choices[0].message.content
    print("return:", dataset_path + img_path)

if __name__ == "__main__":
    import os
    import json

    if len(sys.argv) != 3:
        print("Usage: python example.py <param1> <param2>")
        sys.exit(1)

    param1 = int(sys.argv[1])
    param2 = int(sys.argv[2])

    # different model test
    model_list = ['gemini-flash-1.5', 'gemini-pro-1.5']
    # model_list = [ 'gpt-4o'] #
    i = 0
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(".png"):
                i += 1
                jsonr = {"gt": {}, "llm_label": {}}
                model_index = 0  # 新增一个索引来追踪model_list中的位置
                while model_index < len(model_list):
                    model = model_list[model_index]
                    try:
                        if i > param1 and i < param2:
                            print(i)
                            ret = detect_violation_frame(file, model)
                            jsonr["gt"] = json.loads(ret)
                            jsonr["llm_label"][model] = json.loads(ret)
                            jsonrr = json.dumps(jsonr)
                            with open('./dataset_v1/json_gemini/' + file + '.json', 'w') as f2:
                                f2.write(jsonrr)
                        model_index += 1  # 只有当无错误发生时，才移动到下一个模型
                    except Exception as e:
                        print(f"Error processing file {file} with model {model}: {e}")
                        continue  # 发生错误时重新试验当前模型和当前文件

