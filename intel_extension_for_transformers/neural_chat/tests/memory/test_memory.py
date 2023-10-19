
m intel_extension_for_transformers.neural_chat.pipeline.plugins.memory.memory import Memory, Buffer_Memory
import unittest

class TestMemory(unittest.TestCase):
    def setUp(self):
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()
    
    def test_memory(self):
        query ='hello'
        answer = "Hello! It's nice to meet you. Is there something I can help you with or would you like to chat?"
        memory = Memory()
        memory.add(query, answer)
        context = memory.get()
        text = "User Query: hello"
        self.assertIn(text, context)

    def test_buffer_memory(self):
        query ='hello'
        answer = "Hello! It's nice to meet you. Is there something I can help you with or would you like to chat?"
        buffer_memory = Buffer_Memory()
        buffer_memory.add(query, answer)
        context = buffer_memory.get()
        text = "User Query: hello"
        self.assertIn(text, context)

if __name__ == "__main__":
    unittest.main()
