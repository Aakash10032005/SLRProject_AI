import torch
import torch.nn as nn
from typing import Tuple

# Letters A-Z (indices 0-25), Numbers 0-9 (indices 26-35)
_LETTER_LABELS = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
_NUMBER_LABELS = [str(i) for i in range(10)]

# WLASL top-500 common words (indices 36-535)
_WLASL_WORDS = [
    "hello", "thank you", "please", "sorry", "yes", "no", "help", "water",
    "food", "eat", "drink", "more", "stop", "go", "come", "look", "want",
    "need", "like", "love", "good", "bad", "big", "small", "hot", "cold",
    "happy", "sad", "sick", "tired", "name", "where", "what", "who", "when",
    "how", "why", "home", "school", "work", "hospital", "doctor", "medicine",
    "pain", "bathroom", "sleep", "wake", "morning", "night", "today", "tomorrow",
    "yesterday", "time", "money", "phone", "computer", "book", "car", "bus",
    "walk", "run", "sit", "stand", "open", "close", "give", "take", "buy",
    "sell", "read", "write", "speak", "hear", "see", "understand", "know",
    "think", "feel", "remember", "forget", "learn", "teach", "play", "work",
    "rest", "wait", "try", "finish", "start", "again", "always", "never",
    "sometimes", "now", "later", "before", "after", "here", "there", "inside",
    "outside", "up", "down", "left", "right", "fast", "slow", "easy", "hard",
    "new", "old", "young", "beautiful", "family", "mother", "father", "sister",
    "brother", "friend", "baby", "child", "man", "woman", "people", "animal",
    "dog", "cat", "bird", "fish", "tree", "flower", "sun", "moon", "rain",
    "snow", "wind", "fire", "water_noun", "earth", "sky", "city", "country",
    "language", "sign", "deaf", "hearing", "interpreter", "communicate",
    "meeting", "class", "student", "teacher", "hospital_noun", "police",
    "emergency", "danger", "safe", "lost", "found", "broken", "fixed",
    "clean", "dirty", "full", "empty", "heavy", "light", "loud", "quiet",
    "near", "far", "same", "different", "true", "false", "possible", "impossible",
    "important", "interesting", "boring", "funny", "serious", "angry", "scared",
    "surprised", "confused", "excited", "bored", "hungry", "thirsty", "cold_adj",
    "warm", "ready", "busy", "free", "alone", "together", "first", "last",
    "next", "previous", "number", "color", "red", "blue", "green", "yellow",
    "white", "black", "orange", "purple", "pink", "brown", "gray", "dark",
    "bright", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
    "sunday", "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december", "year", "month",
    "week", "hour", "minute", "second", "birthday", "holiday", "party",
    "game", "sport", "music", "movie", "show", "story", "news", "weather",
    "temperature", "address", "city_noun", "state", "country_noun", "world",
    "america", "india", "english", "hindi", "tamil", "telugu", "malayalam",
    "marathi", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "hundred", "thousand", "million", "half", "quarter",
    "percent", "plus", "minus", "equal", "question", "answer", "idea",
    "problem", "solution", "reason", "example", "information", "message",
    "letter_noun", "word", "sentence", "paragraph", "page", "chapter",
    "picture", "video", "audio", "internet", "email", "text", "call",
    "meeting_noun", "appointment", "schedule", "plan", "goal", "dream",
    "hope", "wish", "promise", "secret", "surprise_noun", "gift", "award",
    "prize", "winner", "loser", "team", "group", "leader", "member",
    "volunteer", "job", "career", "business", "company", "office", "store",
    "restaurant", "hotel", "airport", "station", "park", "beach", "mountain",
    "river", "lake", "ocean", "forest", "desert", "island", "bridge",
    "road", "street", "building", "house", "room", "kitchen", "bedroom",
    "bathroom_noun", "garden", "door", "window", "floor", "ceiling", "wall",
    "table", "chair", "bed", "sofa", "lamp", "mirror", "clock", "calendar",
    "bag", "box", "bottle", "cup", "plate", "spoon", "fork", "knife",
    "shirt", "pants", "shoes", "hat", "glasses", "watch", "ring", "key",
    "umbrella", "camera", "television", "radio", "newspaper", "magazine",
    "pencil", "pen", "paper", "scissors", "tape", "glue", "paint", "brush",
    "ball", "toy", "doll", "puzzle", "card", "coin", "bill", "ticket",
    "map", "flag", "sign_noun", "symbol", "shape", "circle", "square",
    "triangle", "line", "point", "area", "size", "weight", "height",
    "distance", "speed", "direction", "position", "level", "degree",
    "type", "kind", "style", "quality", "value", "price", "cost", "pay",
    "save", "spend", "earn", "lose", "win", "choose", "decide", "agree",
    "disagree", "accept", "refuse", "allow", "prevent", "cause", "effect",
    "change", "improve", "increase", "decrease", "add", "remove", "move",
    "stay", "return", "arrive", "leave", "enter", "exit", "pass", "fail",
    "succeed", "continue", "repeat", "practice", "prepare", "check", "test",
    "measure", "count", "compare", "explain", "describe", "show_verb",
    "demonstrate", "perform", "create", "build", "design", "develop",
    "manage", "control", "support", "protect", "share", "connect", "join",
    "separate", "combine", "divide", "multiply", "calculate", "analyze",
    "research", "discover", "invent", "imagine", "believe", "doubt",
    "trust", "respect", "care", "worry", "enjoy", "suffer", "recover",
    "survive", "celebrate", "mourn", "pray", "meditate", "exercise",
    "relax", "travel", "visit", "explore", "adventure", "experience",
    "memory", "history", "culture", "tradition", "religion", "science",
    "technology", "nature", "environment", "health", "education", "art",
    "music_noun", "dance", "theater", "literature", "philosophy", "politics",
    "economy", "society", "community", "government", "law", "right",
    "responsibility", "freedom", "peace", "war", "conflict", "cooperation",
]

# Pad to exactly 500 words if needed
while len(_WLASL_WORDS) < 500:
    _WLASL_WORDS.append(f"word_{len(_WLASL_WORDS)}")

INDEX_TO_LABEL = _LETTER_LABELS + _NUMBER_LABELS + _WLASL_WORDS[:500]


class ClassifierHead(nn.Module):
    """Classification head mapping HSTFe features to sign labels."""

    def __init__(self, input_dim: int = 512, num_classes: int = 536, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: [B, 512] -> (logits [B, 536], probs [B, 536])"""
        logits = self.net(x)
        probs = torch.softmax(logits, dim=-1)
        return logits, probs

    def predict(self, probs: torch.Tensor) -> Tuple[str, float]:
        """Returns top-1 label and confidence."""
        top_idx = int(probs.argmax(dim=-1).item())
        confidence = float(probs[0, top_idx].item()) if probs.dim() > 1 else float(probs[top_idx].item())
        label = INDEX_TO_LABEL[top_idx] if top_idx < len(INDEX_TO_LABEL) else f"class_{top_idx}"
        return label, confidence
