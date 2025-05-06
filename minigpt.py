import json
import os
import random
from collections import deque

class Node:
    def __init__(self, keyword, response, children=None):
        self.keyword = keyword
        self.response = response
        self.children = children or []

class MiniGPT:
    def __init__(self):
        self.knowledge_base = self._load_knowledge_base()
        self.root = self._build_response_tree()
        
    def _load_knowledge_base(self):
        """Load knowledge base from JSON file"""
        kb_path = os.path.join(os.path.dirname(__file__), 'knowledge_base.json')
        
        if os.path.exists(kb_path):
            with open(kb_path, 'r') as f:
                return json.load(f)
        else:
            raise FileNotFoundError(f"Knowledge base file not found at {kb_path}")
    
    def _build_response_tree(self):
        """Build a tree structure from the knowledge base for efficient searching"""
        root = Node("ROOT", "")
        
        for category, data in self.knowledge_base.items():
            category_node = Node(category, random.choice(data["responses"]))
            for keyword in data["keywords"]:
                keyword_node = Node(keyword, random.choice(data["responses"]))
                category_node.children.append(keyword_node)
            root.children.append(category_node)
            
        return root
    
    def _bfs_search(self, query):
        """Breadth-First Search for finding the best response"""
        if not query:
            return self.knowledge_base["fallback"]["responses"][0]
            
        query_words = set(query.lower().split())
        queue = deque([self.root])
        best_match = None
        best_score = 0
        
        while queue:
            node = queue.popleft()
            
            # Calculate match score (number of matching keywords)
            score = sum(1 for word in query_words if word in node.keyword.lower() or node.keyword.lower() in word)
            
            if score > best_score and node.response:
                best_score = score
                best_match = node
            
            for child in node.children:
                queue.append(child)
        
        if best_match and best_score > 0:
            return best_match.response
        return random.choice(self.knowledge_base["fallback"]["responses"])
    
    def _dfs_search(self, query):
        """Depth-First Search for finding the best response"""
        if not query:
            return self.knowledge_base["fallback"]["responses"][0]
            
        query_words = set(query.lower().split())
        stack = [self.root]
        best_match = None
        best_score = 0
        
        while stack:
            node = stack.pop()
            
            # Calculate match score (number of matching keywords)
            score = sum(1 for word in query_words if word in node.keyword.lower() or node.keyword.lower() in word)
            
            if score > best_score and node.response:
                best_score = score
                best_match = node
            
            # Add children in reverse order to prioritize the first child in DFS
            for child in reversed(node.children):
                stack.append(child)
        
        if best_match and best_score > 0:
            return best_match.response
        return random.choice(self.knowledge_base["fallback"]["responses"])
    
    def get_response(self, query):
        """Get the best response using both BFS and DFS, and choose the better one"""
        # Use both search algorithms and determine which gives a better result
        bfs_response = self._bfs_search(query)
        dfs_response = self._dfs_search(query)
        
        # Simple heuristic: choose the longer response as it might be more informative
        if len(dfs_response) > len(bfs_response):
            return dfs_response
        return bfs_response
