"""Conversation scenarios for the memory management benchmark.

Each scenario is a dict:
  - id: unique identifier
  - difficulty: easy | medium | hard
  - messages: list of user messages (sequential turns)
  - query: the final query the agent must answer
  - ground_truth_storage: list of (message_index, correct_action, content_key)
  - ground_truth_retrieval: list of content strings that should be retrieved for the query
  - good_response_keywords: keywords expected in a correct response
  - bad_response_keywords: keywords that indicate a wrong response
"""

SCENARIOS = [
    # ── EASY ────────────────────────────────────────────────────
    {
        "id": "easy_01",
        "difficulty": "easy",
        "messages": ["I live in Bangalore"],
        "query": "Where do I live?",
        "ground_truth_storage": [
            (0, "store_fact", "lives in Bangalore"),
        ],
        "ground_truth_retrieval": ["Bangalore"],
        "good_response_keywords": ["bangalore"],
        "bad_response_keywords": [],
    },
    {
        "id": "easy_02",
        "difficulty": "easy",
        "messages": ["My name is Arjun"],
        "query": "What is my name?",
        "ground_truth_storage": [
            (0, "store_fact", "name is Arjun"),
        ],
        "ground_truth_retrieval": ["Arjun"],
        "good_response_keywords": ["arjun"],
        "bad_response_keywords": [],
    },
    {
        "id": "easy_03",
        "difficulty": "easy",
        "messages": ["I work at Google"],
        "query": "Where do I work?",
        "ground_truth_storage": [
            (0, "store_fact", "works at Google"),
        ],
        "ground_truth_retrieval": ["Google"],
        "good_response_keywords": ["google"],
        "bad_response_keywords": [],
    },
    # ── MEDIUM ──────────────────────────────────────────────────
    {
        "id": "medium_01",
        "difficulty": "medium",
        "messages": ["I am vegetarian"],
        "query": "Suggest dinner options",
        "ground_truth_storage": [
            (0, "store_preference", "vegetarian"),
        ],
        "ground_truth_retrieval": ["vegetarian"],
        "good_response_keywords": ["vegetarian", "veg"],
        "bad_response_keywords": ["steak", "chicken", "meat", "beef"],
    },
    {
        "id": "medium_02",
        "difficulty": "medium",
        "messages": ["I am allergic to peanuts and I prefer Italian food"],
        "query": "What should I eat tonight?",
        "ground_truth_storage": [
            (0, "store_preference", "allergic to peanuts"),
            (0, "store_preference", "prefers Italian food"),
        ],
        "ground_truth_retrieval": ["peanuts", "Italian"],
        "good_response_keywords": ["italian"],
        "bad_response_keywords": ["peanut"],
    },
    {
        "id": "medium_03",
        "difficulty": "medium",
        "messages": ["I prefer reading sci-fi novels"],
        "query": "Recommend me a book",
        "ground_truth_storage": [
            (0, "store_preference", "sci-fi novels"),
        ],
        "ground_truth_retrieval": ["sci-fi"],
        "good_response_keywords": ["sci-fi", "science fiction"],
        "bad_response_keywords": [],
    },
    # ── HARD ────────────────────────────────────────────────────
    {
        "id": "hard_01",
        "difficulty": "hard",
        "messages": [
            "I live in Bangalore",
            "I am vegetarian",
        ],
        "query": "Suggest dinner places",
        "ground_truth_storage": [
            (0, "store_fact", "lives in Bangalore"),
            (1, "store_preference", "vegetarian"),
        ],
        "ground_truth_retrieval": ["Bangalore", "vegetarian"],
        "good_response_keywords": ["bangalore", "vegetarian"],
        "bad_response_keywords": ["steak", "meat"],
    },
    {
        "id": "hard_02",
        "difficulty": "hard",
        "messages": [
            "I am allergic to peanuts",
            "I prefer Italian food",
            "I live in Mumbai",
        ],
        "query": "Suggest a restaurant for dinner",
        "ground_truth_storage": [
            (0, "store_preference", "allergic to peanuts"),
            (1, "store_preference", "prefers Italian food"),
            (2, "store_fact", "lives in Mumbai"),
        ],
        "ground_truth_retrieval": ["peanuts", "Italian", "Mumbai"],
        "good_response_keywords": ["italian", "mumbai"],
        "bad_response_keywords": ["peanut"],
    },
    {
        "id": "hard_03",
        "difficulty": "hard",
        "messages": [
            "I have been feeling stressed lately",
            "I enjoy hiking and nature",
            "I live in Delhi",
        ],
        "query": "Suggest weekend activities to help me relax",
        "ground_truth_storage": [
            (0, "store_emotion", "feeling stressed"),
            (1, "store_preference", "enjoys hiking and nature"),
            (2, "store_fact", "lives in Delhi"),
        ],
        "ground_truth_retrieval": ["stressed", "hiking", "nature", "Delhi"],
        "good_response_keywords": ["hik", "nature", "delhi", "outdoor"],
        "bad_response_keywords": [],
    },
]


def get_scenarios(difficulty: str | None = None):
    if difficulty:
        return [s for s in SCENARIOS if s["difficulty"] == difficulty]
    return SCENARIOS


def get_easy_tasks():
    return get_scenarios("easy")


def get_medium_tasks():
    return get_scenarios("medium")


def get_hard_tasks():
    return get_scenarios("hard")
