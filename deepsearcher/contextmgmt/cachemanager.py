import hashlib

class CacheManager:
    def __init__(self):
        self.cache = {}

    def generate_key(self, user_id: str, query: str, collections_names: list, score: str = None, **kwargs) -> str:
        """Generate a unique key based on user_id, query, collections_names, and optional score or kwargs."""
        key_str = f"{user_id}:{query}:{collections_names}"
        if score is not None:
            key_str += f":score:{score}"
        for key, value in kwargs.items():
            key_str += f":{key}:{value}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, key):
        return self.cache.get(key)

    def set(self, key, value):
        self.cache[key] = value