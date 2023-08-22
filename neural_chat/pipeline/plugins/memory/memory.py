"""The memory fuction for saving chat history or loading chat history."""


class Memory:
    def __init__(self):
        self.chat_memory = []
    
    def clear(self):
        self.chat_memory.clear()
        
    
    def get(self):
        context = ""
        for history in self.chat_memory:
            if context:
                context += history
            else:
                context += f"\n{history}"
        return context
    
    def add(self, query, answer):
        turn = len(self.chat_memory)
        cur_conversation = "[Round {}]\nUser Query: {}\nAI Answer:{}".format(turn+1, query, answer)
        self.chat_memory.add(cur_conversation)
        
        
class Buffer_Memory:
    def __init__(self, buffer_size=3):
        self.chat_memory = []
        self.buffer_size = buffer_size
        self.turn = 0

    def clear(self):
        self.chat_memory.clear()
        self.turn = 0

    def get(self):
        context = ""
        for history in self.chat_memory:
            if context:
                context += history
            else:
                context += f"\n{history}"
        return context

    def add(self, query, answer):
        length = len(self.chat_memory)
        if length > self.buffer_size:
            self.chat_memory = self.chat_memory[1:]
        cur_conversation = "[Round {}]\nUser Query: {}\nAI Answer:{}".format(self.turn+1, query, answer)
        self.chat_memory.add(cur_conversation)
