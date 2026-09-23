PROBLEMS = [
    {
        "title": "Two Sum",
        "difficulty": "Easy",
        "prompt": "Given an array of integers `nums` and an integer `target`, return indices of the two numbers that add up to `target`.",
        "examples": "Input: nums=[2,7,11,15], target=9 → [0,1]",
    },
    {
        "title": "Valid Parentheses",
        "difficulty": "Easy",
        "prompt": "Given a string containing just the characters '()[]{}', determine if the input string is valid.",
        "examples": "'()[]{}' → True; '(]' → False",
    },
    {
        "title": "Longest Substring Without Repeating Characters",
        "difficulty": "Medium",
        "prompt": "Given a string s, find the length of the longest substring without repeating characters.",
        "examples": "'abcabcbb' → 3 ('abc')",
    },
    {
        "title": "Group Anagrams",
        "difficulty": "Medium",
        "prompt": "Given an array of strings, group anagrams together.",
        "examples": "['eat','tea','tan','ate','nat','bat'] → [['eat','tea','ate'],['tan','nat'],['bat']]",
    },
    {
        "title": "Sliding Window Rate Limiter",
        "difficulty": "Medium",
        "prompt": "Design a rate limiter that allows at most N requests per rolling T-second window. Implement `allow(timestamp)`.",
        "examples": "N=3, T=60; allow(1)→True; allow(2)→True; allow(3)→True; allow(4)→False",
    },
    {
        "title": "LRU Cache",
        "difficulty": "Hard",
        "prompt": "Design an LRU cache supporting get(key) and put(key,value) in O(1) time.",
        "examples": "capacity=2; put(1,1); put(2,2); get(1)→1; put(3,3) evicts key 2",
    },
]
