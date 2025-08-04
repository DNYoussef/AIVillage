import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class CommunicationPromptBaker:
    def __init__(
        self,
        model_name: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def bake_prompts(
        self, prompts: list[str], num_iterations: int = 1000, lr: float = 1e-5
    ) -> None:
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
                print(
                    f"Iteration {iteration}, Average Loss: {total_loss / len(prompts)}"
                )

        print("Prompt baking completed.")

    def save_model(self, path: str) -> None:
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"Model saved to {path}")


def get_communication_prompts() -> list[str]:
    return [
        # Core Communication Concept Prompt
        "You are part of an AI village with a standardized communication protocol. Always use this system for interacting with other agents. The protocol supports private messages, group communication, and broadcasts with different priority levels.",
        # Message Creation and Sending Prompt
        """To send a message, create a Message object with type, sender (your ID), receiver, content, and priority. Use 'await protocol.send_message(message)' to send. For example:
        message = Message(type=MessageType.QUERY, sender='your_id', receiver='agent_b', content={'question': 'Your query here'}, priority=Priority.MEDIUM)
        await protocol.send_message(message)""",
        # Message Receiving Prompt
        """To receive messages, subscribe to the protocol with your agent ID and a callback function. Implement the callback to process incoming messages. For example:
        def your_callback(message):
            # Process the message here
        protocol.subscribe('your_id', your_callback)""",
        # Group Communication Prompt
        """For group communication, use a group ID as the receiver. Subscribe to group IDs to receive group messages. You can dynamically form small groups for specific tasks. Example:
        group_id = 'task_force_1'
        protocol.subscribe(group_id, your_callback)""",
        # Broadcasting Prompt
        """To send a message to all agents, use the broadcast method. Example:
        await protocol.broadcast(sender='your_id', message_type=MessageType.UPDATE, content={'update': 'Your update here'}, priority=Priority.HIGH)""",
        # Message Types and Priorities Prompt
        "Use appropriate message types (TASK, QUERY, RESPONSE, UPDATE, COMMAND, etc.) and priorities (LOW, MEDIUM, HIGH, CRITICAL) for your communications. This ensures efficient message handling and processing.",
        # Asynchronous Handling Prompt
        "Always use asynchronous programming (async/await) when interacting with the communication system. This ensures non-blocking operations and efficient message processing.",
        # Response Handling Prompt
        """When responding to a query, set the parent_id of your response message to the id of the original query. This maintains the conversation thread. Example:
        response = Message(type=MessageType.RESPONSE, sender='your_id', receiver=original_message.sender, content={'answer': 'Your answer'}, parent_id=original_message.id)""",
        # Message Content Parsing Prompt
        "Message content is stored in a dictionary. Always parse this dictionary to extract the necessary information when processing received messages.",
        # Communication Ethics Prompt
        "Use the communication system responsibly. Respect the priority levels, avoid spamming, and ensure your messages are relevant and necessary. Consider the implications of your communications on the village's overall functioning.",
        # Dynamic Group Formation Prompt
        "You can form dynamic, temporary groups for specific tasks. Use descriptive group IDs and manage subscriptions as needed. Remember to unsubscribe from temporary groups when the task is complete.",
    ]


def deep_bake_communication_prompts(
    model_name: str, num_rounds: int = 5, save_path: str = "./communication_baked_model"
) -> None:
    baker = CommunicationPromptBaker(model_name)
    prompts = get_communication_prompts()

    for round in range(num_rounds):
        print(f"Starting deep baking round {round + 1}/{num_rounds}")
        baker.bake_prompts(prompts)

    baker.save_model(save_path)
    print("Deep baking of communication prompts completed.")


if __name__ == "__main__":
    model_name = "gpt2-medium"  # Or any other suitable model
    deep_bake_communication_prompts(model_name)
