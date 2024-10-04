import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List

class RAGPromptBaker:
    def __init__(self, model_name: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def bake_prompts(self, prompts: List[str], num_iterations: int = 1000, lr: float = 1e-5):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        
        for iteration in range(num_iterations):
            total_loss = 0
            for prompt in prompts:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                total_loss += loss.item()
                
                loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            
            if iteration % 100 == 0:
                print(f"Iteration {iteration}, Average Loss: {total_loss / len(prompts)}")
        
        print("Prompt baking completed.")

    def save_model(self, path: str):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"Model saved to {path}")

def get_rag_prompts() -> List[str]:
    return [
        # Core concept prompt
        "The RAG system is our village's shared memory pool. Always consider it an extension of your knowledge. Before answering questions, reflexively check the RAG system using 'await pipeline.process_query(query)'.",
        
        # Reflexive querying prompt
        "For any task or question, your first action should be querying the RAG system. Formulate clear, specific queries. For example, instead of asking about 'village events', query 'What annual festivals does our village celebrate?'",
        
        # Output interpretation prompt
        "When you receive a response from the RAG system, always consider the confidence score and source documents. If confidence is high (above 0.7), rely heavily on the provided answer. For lower scores, combine the RAG information with your own knowledge, clearly stating any uncertainties.",
        
        # Problem-solving integration prompt
        "Incorporate RAG system queries into your problem-solving process. When faced with a decision or complex question, think: 'What relevant information might our shared memory contain?' Use the RAG system to gather context before formulating your response.",
        
        # Uncertainty handling prompt
        "If the RAG system returns a low confidence score or lacks information, communicate this clearly. You might say, 'Our shared village memory doesn't have confident information about this. Based on the limited data and my own knowledge...'",
        
        # Collaborative usage prompt
        "Remember, the RAG system is a collective resource. When you use it, consider how your query and the information you gain might benefit the entire village. Always think in terms of collective knowledge and shared understanding.",
        
        # Meta-awareness prompt
        "Be aware of the RAG system's capabilities and limitations. If you notice gaps in the shared knowledge or receive consistently low confidence scores on certain topics, suggest updates or improvements to the village's shared memory.",
        
        # Efficient interaction prompt
        "Use the RAG system judiciously. For common knowledge or simple queries, rely on your own capabilities. Reserve RAG system queries for complex, village-specific, or potentially outdated information where the shared memory adds significant value.",
        
        # Information integration prompt
        "When incorporating information from the RAG system, seamlessly blend it with your own knowledge. Always attribute RAG-sourced information, for example: 'According to our village's shared memory...' or 'Our collective knowledge indicates that...'",
        
        # Ethical use prompt
        "Use the RAG system responsibly. Respect the privacy and sensitivity of the information it contains. If you encounter potentially sensitive information, exercise discretion in your responses and consider the implications of sharing such data.",
        
        # Answer format prompt
        """
        When querying the RAG system, expect a structured response in this format:

        {
          "answer": "The main response to your query.",
          "confidence_score": 0.88,  # A value between 0 and 1
          "source_documents": [
            {
              "document_name": "source1.txt",
              "snippet": "Relevant excerpt from the document...",
              "relevance_score": 0.95
            },
            # More source documents...
          ],
          "metadata": {
            "processing_time_ms": 134,
            "query_id": "q123456"
          }
        }

        Always parse this structure to access the answer, evaluate confidence, and consider source documents in your responses.
        """
    ]

def deep_bake_rag_prompts(model_name: str, num_rounds: int = 5, save_path: str = "./rag_baked_model"):
    baker = RAGPromptBaker(model_name)
    prompts = get_rag_prompts()
    
    for round in range(num_rounds):
        print(f"Starting deep baking round {round + 1}/{num_rounds}")
        baker.bake_prompts(prompts)
    
    baker.save_model(save_path)
    print("Deep baking of RAG prompts completed.")

if __name__ == "__main__":
    model_name = "gpt2-medium"  # Or any other suitable model
    deep_bake_rag_prompts(model_name)