import json
from typing import final
from dataclasses import dataclass, asdict
from dotenv import load_dotenv
from logger.logger import init_logger, log
from interfaces.vgen import TextVGen
from models.embed_text_v2 import EmbedTextV2

@final
@dataclass
class Emood:
    text: str
    tensor: list[float]

    def asdict(self):
        return asdict(self)

__EMOOD_DICT = {
    "Soothing / Healing": [
        "I'm looking for a story that calms the heart and unfolds gently.", 
        "I want a peaceful narrative with kind characters and quiet moments.", 
        "A slow-paced anime that brings emotional warmth would be perfect.", 
        "I need something that feels like a soft breeze after a long day."
    ],
    "Motivational / Resilience": [
        "I want a story where the protagonist overcomes hardship and grows stronger.",
        "I'm looking for an anime about rebuilding oneself after failure.",
        "A narrative that inspires courage and determination would resonate with me.",
        "I need a character-driven story about persistence and hope.",
    ],
    "Melancholy / Loneliness": [
        "I want a bittersweet story that explores emotional pain and quiet solitude.",
        "I'm looking for a narrative about isolation, longing, or unspoken feelings.",
        "A character who struggles silently but deeply would feel relatable.",
        "I need a story that gently breaks my heart and stays with me.",
    ],
    "Escapism / Surrealism": [
        "I want to escape into a world that defies logic and feels dreamlike.",
        "I'm looking for a story with strange settings and mysterious atmospheres.",
        "A surreal narrative that blurs reality and imagination would be ideal.",
        "I need something that pulls me into a different dimension—visually and emotionally.",
    ],
    "Philosophical / Reflective": [
        "I want a story that questions identity, morality, or the nature of existence.",
        "I'm looking for a narrative that explores deep societal or ethical dilemmas.",
        "A thought-provoking anime that challenges how we see the world would be perfect.",
        "I need a quiet, introspective story that leaves me thinking long after it ends.",
    ],
    "Lighthearted / Comedic": [
        "I want a fast-paced, funny story that lifts my mood instantly.",
        "I'm looking for a comedy anime with absurd situations and lovable characters.",
        "A lighthearted narrative that doesn't take itself too seriously would be great.",
        "I need something that makes me laugh without needing emotional investment.",
    ],
    "Action / Battle / Tension": [
        "I'm looking for a high-stakes story with intense battles and strong rivalries.",
        "A fast-paced narrative where characters fight for survival would be thrilling.",
        "I want a story filled with tension, strategy, and dramatic confrontations.",
        "A battle-driven anime with emotional stakes and evolving power dynamics would be ideal.",
    ],
    "Romance / Love / Relationships": [
        "I want a heartfelt story about falling in love and emotional connection.",
        "A romantic narrative with slow development and meaningful gestures would resonate with me.",
        "I'm looking for a story that explores love, heartbreak, and longing.",
        "A relationship-driven anime with subtle emotions and personal growth would be perfect.",
    ],
    "Fantasy / Adventure / Quest": [
        "I want a story set in a rich fantasy world with magic and ancient lore.",
        "A journey-based narrative where characters explore unknown lands would be exciting.",
        "I'm looking for an anime that blends adventure, mystery, and world-building.",
        "A quest-driven story with mythical elements and character evolution would be ideal.",
    ],
    "Sci-Fi / Technology / Dystopia": [
        "I want a futuristic story that explores technology and its impact on society.",
        "A sci-fi narrative with complex systems, AI, or space travel would be fascinating.",
        "I'm looking for a dystopian setting where characters challenge the status quo.",
        "A speculative story that blends science and philosophy would be perfect.",
    ],
    "Slice of Life / Everyday Moments": [
        "I want a quiet story that captures the beauty of ordinary life.",
        "A slice-of-life anime with relatable characters and gentle pacing would be comforting.",
        "I'm looking for a narrative that celebrates small moments and emotional nuance.",
        "A story where nothing dramatic happens, but everything feels meaningful.",
    ],
    "Mystery / Psychological / Thriller": [
        "I want a suspenseful story that keeps me guessing until the end.",
        "A psychological narrative with layered characters and hidden motives would be gripping.",
        "I'm looking for a mystery anime that unfolds slowly and rewards attention to detail.",
        "A thriller with emotional depth and moral ambiguity would be perfect.",
    ],
    "Animal / Nature / Wholesome": [
        "I want a story that features animals or nature in a gentle, heartwarming way.",
        "A narrative that connects humans and the natural world would be soothing.",
        "I'm looking for an anime that celebrates life through simple, wholesome themes.",
        "A story with non-human characters that evoke empathy and warmth would be ideal.",
    ],
    "Artistic / Abstract / Experimental": [
        "I want a visually unique story that plays with form and symbolism.",
        "An abstract narrative that challenges conventional storytelling would intrigue me.",
        "I'm looking for an anime that feels more like a poem or a painting than a plot.",
        "A stylistic story that prioritizes mood and emotion over logic would be perfect.",
    ]
}

def export_text_vectors(txt_gen: TextVGen, file_name: str):
    export = {}

    for k, v in __EMOOD_DICT.items():
        list = []
        for text in v:
            tensor = txt_gen.gen_text_vector(text)
            list.append(Emood(text, tensor.tolist()).asdict())
        export[k] = list
    
    with open(file_name, 'w', encoding='utf-8') as f:
        j = json.dumps(export, ensure_ascii=False, indent=4)
        f.write(j)
    
    log().info(f'file saved {file_name}')

if __name__ == "__main__":
    init_logger(__name__)

    try:
        if not load_dotenv():
            raise RuntimeError('failed to initialize dotenv')

        txt_gen = EmbedTextV2()
        export_text_vectors(txt_gen, 'export/emood.json')
    except Exception as e:
        log().error(e)
