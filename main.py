from datasets import load_dataset
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer

# Load multiple datasets
daily_dialog = load_dataset('datasets/JaQuAD')
# Add other datasets here as needed
# e.g., another_dataset = load_dataset('another_dataset_name')

# Extract conversations from the datasets
conversations = []

# Process daily_dialog dataset
for dialog in daily_dialog['train']:
    conversations.extend(dialog['dialog'])

# Process additional datasets similarly
# for dialog in another_dataset['train']:
#     conversations.extend(dialog['dialog'])

# Create a new instance of a ChatBot
chatbot = ChatBot("JapaneseBot")

# Create a trainer for the chatbot
trainer = ListTrainer(chatbot)

# Train the chatbot with the extracted conversations
for conversation in conversations:
    for i in range(len(conversation) - 1):
        trainer.train([conversation[i], conversation[i + 1]])

print("JapaneseBot is ready to chat! Type 'exit' to end the conversation.")

while True:
    try:
        user_input = input("あなた: ")
        if user_input.lower() == "exit":
            print("JapaneseBot: さようなら！")
            break
        response = chatbot.get_response(user_input)
        print(f"JapaneseBot: {response}")
    except (KeyboardInterrupt, EOFError, SystemExit):
        print("\nJapaneseBot: さようなら！")
        break