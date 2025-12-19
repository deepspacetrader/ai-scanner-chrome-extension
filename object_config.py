# object_config.py

# Categorized COCO 80 classes
# Colors: 
#   - Humans: Greens
#   - Vehicles: Oranges/Reds
#   - Animals: Purples/Pinks
#   - Accessories/Food: Yellows/Cyan
#   - Furniture/Electronics: Browns/Blues

OBJECT_CONFIG = {
    # --- Humans ---
    "person": {
        "color": "#00FF00",
        "task": "<DETAILED_CAPTION>",
        "prompt": "Identify this person.",
        "llm_query": "Identify this person. Provide name or physical description.",
        "category": "Humans",
        "is_analyzable": True
    },
    
    # --- Vehicles ---
    "bicycle": {
        "color": "#FF6B00",
        "task": "<VQA>",
        "prompt": "What type of bicycle is this?",
        "llm_query": "Identify the type and notable features of this bicycle.",
        "category": "Vehicles",
        "is_analyzable": True
    },
    "car": {
        "color": "#FF6B00",
        "task": "<VQA>",
        "prompt": "What is the make and model of this car?",
        "llm_query": "Identify manufacturer, model, and estimated year.",
        "category": "Vehicles",
        "is_analyzable": True
    },
    "motorcycle": {
        "color": "#FF6B00",
        "task": "<VQA>",
        "prompt": "What make and model is this motorcycle?",
        "llm_query": "Identify manufacturer and model of this motorcycle.",
        "category": "Vehicles",
        "is_analyzable": True
    },
    "airplane": {
        "color": "#FF6B00",
        "task": "<VQA>",
        "prompt": "What kind of aircraft is this?",
        "llm_query": "Identify the aircraft model and airline if visible.",
        "category": "Vehicles",
        "is_analyzable": True
    },
    "bus": {
        "color": "#FF6B00",
        "task": "<VQA>",
        "prompt": "What type of bus is this?",
        "llm_query": "Identify the bus type and any company branding.",
        "category": "Vehicles",
        "is_analyzable": True
    },
    "train": {
        "color": "#FF6B00",
        "task": "<VQA>",
        "prompt": "What type of train or locomotive is this?",
        "llm_query": "Identify the train type and operator.",
        "category": "Vehicles",
        "is_analyzable": True
    },
    "truck": {
        "color": "#FF6B00",
        "task": "<VQA>",
        "prompt": "What is the make and model of this truck?",
        "llm_query": "Identify the truck manufacturer and configuration.",
        "category": "Vehicles",
        "is_analyzable": True
    },
    "boat": {
        "color": "#FF6B00",
        "task": "<VQA>",
        "prompt": "What kind of boat or ship is this?",
        "llm_query": "Identify the vessel type and name if visible.",
        "category": "Vehicles",
        "is_analyzable": True
    },

    # --- Animals ---
    "bird": {
        "color": "#FF00FF",
        "task": "<VQA>",
        "prompt": "What species of bird is this?",
        "llm_query": "Identify the bird species and notable plumage details.",
        "category": "Animals",
        "is_analyzable": True
    },
    "cat": {
        "color": "#FF00FF",
        "task": "<VQA>",
        "prompt": "What breed is this cat?",
        "llm_query": "Identify the cat breed and coat patterns.",
        "category": "Animals",
        "is_analyzable": True
    },
    "dog": {
        "color": "#FF00FF",
        "task": "<VQA>",
        "prompt": "What breed is this dog?",
        "llm_query": "Identify the dog breed and size.",
        "category": "Animals",
        "is_analyzable": True
    },
    "horse": {
        "color": "#FF00FF",
        "task": "<VQA>",
        "prompt": "What breed of horse is this?",
        "llm_query": "Identify the horse breed and color.",
        "category": "Animals",
        "is_analyzable": True
    },
    "sheep": {
        "color": "#FF00FF",
        "task": "<DETAILED_CAPTION>",
        "prompt": "Describe this sheep.",
        "llm_query": "Describe the sheep's condition and environment.",
        "category": "Animals",
        "is_analyzable": False
    },
    "cow": {
        "color": "#FF00FF",
        "task": "<VQA>",
        "prompt": "What breed of cattle is this?",
        "llm_query": "Identify the cattle breed.",
        "category": "Animals",
        "is_analyzable": False
    },
    "elephant": {
        "color": "#FF00FF",
        "task": "<VQA>",
        "prompt": "Is this an African or Asian elephant?",
        "llm_query": "Identify the elephant species.",
        "category": "Animals",
        "is_analyzable": True
    },
    "bear": {
        "color": "#FF00FF",
        "task": "<VQA>",
        "prompt": "What kind of bear is this?",
        "llm_query": "Identify the bear species (Grizzly, Black, Polar, etc).",
        "category": "Animals",
        "is_analyzable": True
    },
    "zebra": {
        "color": "#FF00FF",
        "task": "<DETAILED_CAPTION>",
        "prompt": "Describe this zebra.",
        "llm_query": "Describe the zebra's appearance.",
        "category": "Animals",
        "is_analyzable": False
    },
    "giraffe": {
        "color": "#FF00FF",
        "task": "<DETAILED_CAPTION>",
        "prompt": "Describe this giraffe.",
        "llm_query": "Describe the giraffe's appearance.",
        "category": "Animals",
        "is_analyzable": False
    },

    # --- Outdoor/Infrastructure ---
    "traffic light": {"color": "#FF0000", "task": "<VQA>", "prompt": "What color is the traffic light?", "llm_query": "Identify traffic light status.", "category": "Outdoors", "is_analyzable": False},
    "fire hydrant": {"color": "#FF0000", "task": "<DETAILED_CAPTION>", "prompt": "Describe the hydrant.", "llm_query": "Describe fire hydrant.", "category": "Outdoors", "is_analyzable": False},
    "stop sign": {"color": "#FF0000", "task": "<DETAILED_CAPTION>", "prompt": "Identify the sign.", "llm_query": "Confirm stop sign status.", "category": "Outdoors", "is_analyzable": False},
    "parking meter": {"color": "#AAAAAA", "task": "<DETAILED_CAPTION>", "prompt": "Describe the meter.", "llm_query": "Describe parking meter.", "category": "Outdoors", "is_analyzable": False},
    "bench": {"color": "#8B4513", "task": "<DETAILED_CAPTION>", "prompt": "Describe the bench.", "llm_query": "Describe bench style.", "category": "Outdoors", "is_analyzable": False},

    # --- Accessories ---
    "backpack": {"color": "#FFFF00", "task": "<VQA>", "prompt": "What brand is this backpack?", "llm_query": "Identify backpack brand/type.", "category": "Accessories", "is_analyzable": True},
    "umbrella": {"color": "#00FFFF", "task": "<DETAILED_CAPTION>", "prompt": "Describe the umbrella.", "llm_query": "Describe umbrella.", "category": "Accessories", "is_analyzable": False},
    "handbag": {"color": "#FFFF00", "task": "<VQA>", "prompt": "What brand is this handbag?", "llm_query": "Identify handbag brand.", "category": "Accessories", "is_analyzable": True},
    "tie": {"color": "#FFFF00", "task": "<DETAILED_CAPTION>", "prompt": "Describe the tie.", "llm_query": "Describe tie pattern.", "category": "Accessories", "is_analyzable": False},
    "suitcase": {"color": "#FFFF00", "task": "<VQA>", "prompt": "What brand is this suitcase?", "llm_query": "Identify suitcase brand.", "category": "Accessories", "is_analyzable": True},

    # --- Sports ---
    "frisbee": {"color": "#00FFFF", "task": "<DETAILED_CAPTION>", "prompt": "Describe the frisbee.", "llm_query": "Describe frisbee.", "category": "Sports", "is_analyzable": False},
    "skis": {"color": "#00FFFF", "task": "<VQA>", "prompt": "What brand are these skis?", "llm_query": "Identify skis brand.", "category": "Sports", "is_analyzable": True},
    "snowboard": {"color": "#00FFFF", "task": "<VQA>", "prompt": "What brand is this snowboard?", "llm_query": "Identify snowboard brand.", "category": "Sports", "is_analyzable": True},
    "sports ball": {"color": "#00FFFF", "task": "<VQA>", "prompt": "What kind of ball is this?", "llm_query": "Identify the sport for this ball.", "category": "Sports", "is_analyzable": True},
    "kite": {"color": "#00FFFF", "task": "<DETAILED_CAPTION>", "prompt": "Describe the kite.", "llm_query": "Describe kite.", "category": "Sports", "is_analyzable": False},
    "baseball bat": {"color": "#00FFFF", "task": "<DETAILED_CAPTION>", "prompt": "Describe the bat.", "llm_query": "Describe bat.", "category": "Sports", "is_analyzable": False},
    "baseball glove": {"color": "#00FFFF", "task": "<DETAILED_CAPTION>", "prompt": "Describe the glove.", "llm_query": "Describe glove.", "category": "Sports", "is_analyzable": False},
    "skateboard": {"color": "#00FFFF", "task": "<VQA>", "prompt": "What is on the graphic of this skateboard?", "llm_query": "Describe skateboard graphic.", "category": "Sports", "is_analyzable": True},
    "surfboard": {"color": "#00FFFF", "task": "<VQA>", "prompt": "What brand is this surfboard?", "llm_query": "Identify surfboard brand.", "category": "Sports", "is_analyzable": True},
    "tennis racket": {"color": "#00FFFF", "task": "<VQA>", "prompt": "What brand is this racket?", "llm_query": "Identify racket brand.", "category": "Sports", "is_analyzable": True},

    # --- Household/Kitchen ---
    "bottle": {"color": "#FF4444", "task": "<VQA>", "prompt": "What is in this bottle?", "llm_query": "Identify bottle content/brand.", "category": "Household", "is_analyzable": True},
    "wine glass": {"color": "#FF4444", "task": "<DETAILED_CAPTION>", "prompt": "Describe the glass.", "llm_query": "Describe wine glass.", "category": "Household", "is_analyzable": False},
    "cup": {"color": "#FF4444", "task": "<VQA>", "prompt": "What brand or logo is on this cup?", "llm_query": "Identify cup branding.", "category": "Household", "is_analyzable": True},
    "fork": {"color": "#FF4444", "task": "<DETAILED_CAPTION>", "prompt": "Describe the fork.", "llm_query": "Describe fork.", "category": "Household", "is_analyzable": False},
    "knife": {"color": "#FF4444", "task": "<DETAILED_CAPTION>", "prompt": "Describe the knife.", "llm_query": "Describe knife.", "category": "Household", "is_analyzable": False},
    "spoon": {"color": "#FF4444", "task": "<DETAILED_CAPTION>", "prompt": "Describe the spoon.", "llm_query": "Describe spoon.", "category": "Household", "is_analyzable": False},
    "bowl": {"color": "#FF4444", "task": "<VQA>", "prompt": "What is in this bowl?", "llm_query": "Identify bowl contents.", "category": "Household", "is_analyzable": True},

    # --- Food ---
    "banana": {"color": "#CCFF00", "task": "<DETAILED_CAPTION>", "prompt": "Describe the banana.", "llm_query": "Describe banana ripeness.", "category": "Food", "is_analyzable": False},
    "apple": {"color": "#CCFF00", "task": "<VQA>", "prompt": "What type of apple is this?", "llm_query": "Identify apple variety.", "category": "Food", "is_analyzable": True},
    "sandwich": {"color": "#CCFF00", "task": "<VQA>", "prompt": "What kind of sandwich is this?", "llm_query": "Identify sandwich type/ingredients.", "category": "Food", "is_analyzable": True},
    "orange": {"color": "#CCFF00", "task": "<VQA>", "prompt": "Is this an orange or a tangerine?", "llm_query": "Identify citrus type.", "category": "Food", "is_analyzable": False},
    "broccoli": {"color": "#CCFF00", "task": "<DETAILED_CAPTION>", "prompt": "Describe the broccoli.", "llm_query": "Describe broccoli.", "category": "Food", "is_analyzable": False},
    "carrot": {"color": "#CCFF00", "task": "<DETAILED_CAPTION>", "prompt": "Describe the carrot.", "llm_query": "Describe carrot.", "category": "Food", "is_analyzable": False},
    "hot dog": {"color": "#CCFF00", "task": "<VQA>", "prompt": "What toppings are on this hot dog?", "llm_query": "Describe hot dog toppings.", "category": "Food", "is_analyzable": True},
    "pizza": {"color": "#CCFF00", "task": "<VQA>", "prompt": "What toppings are on this pizza?", "llm_query": "Identify pizza toppings.", "category": "Food", "is_analyzable": True},
    "donut": {"color": "#CCFF00", "task": "<VQA>", "prompt": "What kind of donut is this?", "llm_query": "Identify donut type.", "category": "Food", "is_analyzable": True},
    "cake": {"color": "#CCFF00", "task": "<VQA>", "prompt": "What kind of cake is this?", "llm_query": "Identify cake type.", "category": "Food", "is_analyzable": True},

    # --- Furniture ---
    "chair": {"color": "#8B4513", "task": "<DETAILED_CAPTION>", "prompt": "Describe the chair.", "llm_query": "Describe chair style.", "category": "Household", "is_analyzable": True},
    "couch": {"color": "#8B4513", "task": "<DETAILED_CAPTION>", "prompt": "Describe the couch.", "llm_query": "Describe couch style.", "category": "Household", "is_analyzable": True},
    "potted plant": {"color": "#228B22", "task": "<VQA>", "prompt": "What species of plant is this?", "llm_query": "Identify plant species.", "category": "Outdoors", "is_analyzable": True},
    "bed": {"color": "#8B4513", "task": "<DETAILED_CAPTION>", "prompt": "Describe the bed.", "llm_query": "Describe bed type.", "category": "Household", "is_analyzable": False},
    "dining table": {"color": "#8B4513", "task": "<DETAILED_CAPTION>", "prompt": "Describe the table.", "llm_query": "Describe table.", "category": "Household", "is_analyzable": False},
    "toilet": {"color": "#FFFFFF", "task": "<DETAILED_CAPTION>", "prompt": "Describe the toilet.", "llm_query": "Describe toilet.", "category": "Household", "is_analyzable": False},

    # --- Electronics ---
    "tv": {"color": "#0080FF", "task": "<VQA>", "prompt": "What is showing on this TV?", "llm_query": "Describe TV content.", "category": "Electronics", "is_analyzable": True},
    "laptop": {"color": "#0080FF", "task": "<VQA>", "prompt": "What brand of laptop is this?", "llm_query": "Identify laptop brand.", "category": "Electronics", "is_analyzable": True},
    "mouse": {"color": "#0080FF", "task": "<VQA>", "prompt": "What brand of mouse is this?", "llm_query": "Identify mouse brand.", "category": "Electronics", "is_analyzable": True},
    "remote": {"color": "#0080FF", "task": "<DETAILED_CAPTION>", "prompt": "Describe the remote.", "llm_query": "Describe remote.", "category": "Electronics", "is_analyzable": False},
    "keyboard": {"color": "#0080FF", "task": "<VQA>", "prompt": "What brand of keyboard is this?", "llm_query": "Identify keyboard brand.", "category": "Electronics", "is_analyzable": True},
    "cell phone": {"color": "#0080FF", "task": "<VQA>", "prompt": "What model of phone is this?", "llm_query": "Identify phone model.", "category": "Electronics", "is_analyzable": True},
    "microwave": {"color": "#0080FF", "task": "<VQA>", "prompt": "What brand of microwave is this?", "llm_query": "Identify microwave brand.", "category": "Electronics", "is_analyzable": False},
    "oven": {"color": "#0080FF", "task": "<VQA>", "prompt": "What brand of oven is this?", "llm_query": "Identify oven brand.", "category": "Electronics", "is_analyzable": False},
    "toaster": {"color": "#0080FF", "task": "<VQA>", "prompt": "What brand of toaster is this?", "llm_query": "Identify toaster brand.", "category": "Electronics", "is_analyzable": False},
    "sink": {"color": "#0080FF", "task": "<DETAILED_CAPTION>", "prompt": "Describe the sink.", "llm_query": "Describe sink.", "category": "Household", "is_analyzable": False},
    "refrigerator": {"color": "#0080FF", "task": "<VQA>", "prompt": "What brand of refrigerator is this?", "llm_query": "Identify refrigerator brand.", "category": "Household", "is_analyzable": False},

    # --- Misc ---
    "book": {"color": "#A52A2A", "task": "<VQA>", "prompt": "What is the title of this book?", "llm_query": "Identify book title and author.", "category": "Accessories", "is_analyzable": True},
    "clock": {"color": "#A52A2A", "task": "<VQA>", "prompt": "What time is it on this clock?", "llm_query": "Identify time on clock.", "category": "Household", "is_analyzable": True},
    "vase": {"color": "#A52A2A", "task": "<DETAILED_CAPTION>", "prompt": "Describe the vase.", "llm_query": "Describe vase style.", "category": "Household", "is_analyzable": False},
    "scissors": {"color": "#A52A2A", "task": "<DETAILED_CAPTION>", "prompt": "Describe the scissors.", "llm_query": "Describe scissors.", "category": "Household", "is_analyzable": False},
    "teddy bear": {"color": "#A52A2A", "task": "<DETAILED_CAPTION>", "prompt": "Describe the teddy bear.", "llm_query": "Describe teddy bear.", "category": "Accessories", "is_analyzable": False},
    "hair drier": {"color": "#A52A2A", "task": "<VQA>", "prompt": "What brand is this hair drier?", "llm_query": "Identify hair drier brand.", "category": "Electronics", "is_analyzable": False},
    "toothbrush": {"color": "#A52A2A", "task": "<DETAILED_CAPTION>", "prompt": "Describe the toothbrush.", "llm_query": "Describe toothbrush.", "category": "Household", "is_analyzable": False},
}
