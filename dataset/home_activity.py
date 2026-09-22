from transformers import pipeline
import pandas as pd
import os

os.makedirs("dataset", exist_ok=True)

print("Loading model...")

generator = pipeline(
    "text-generation",
    model="distilgpt2"
)

print("Model loaded successfully!")

prompts = [

    {
        "prompt_id": "P06",
        "technique": "Zero-Shot",
        "prompt": "Explain the difference between mean and median to a beginner with a simple example."
    },

    {
        "prompt_id": "P07",
        "technique": "Zero-Shot",
        "prompt": "What are some general ways to maintain a healthy daily routine?"
    },

    {
        "prompt_id": "P08",
        "technique": "Zero-Shot",
        "prompt": "Explain the difference between saving and investing for a beginner."
    },

    {
        "prompt_id": "P09",
        "technique": "Zero-Shot",
        "prompt": "A customer received a damaged product. Write a polite response explaining the next steps."
    },

    {
        "prompt_id": "P10",
        "technique": "Zero-Shot",
        "prompt": "Explain what a Python function is to a beginner with a simple example."
    },

    {
        "prompt_id": "P11",
        "technique": "One-Shot",
        "prompt": """Example: Mean is the average of numbers, while median is the middle value after arranging them.
Now explain the difference between mean and median to a beginner with a simple example."""
    },

    {
        "prompt_id": "P12",
        "technique": "One-Shot",
        "prompt": """Example: A healthy routine can include regular sleep, balanced meals, exercise, and enough water.
Now suggest a simple healthy daily routine for a student."""
    },

    {
        "prompt_id": "P13",
        "technique": "One-Shot",
        "prompt": """Example: Saving means keeping money for future needs, while investing means putting money into something that may grow in value.
Now explain the difference between saving and investing for a beginner."""
    },

    {
        "prompt_id": "P14",
        "technique": "One-Shot",
        "prompt": """Example: We are sorry that you received a damaged product. Please share your order details and photos of the damage so we can help with the next steps.
Now write a polite customer-support response to a customer who received a damaged product."""
    },

    {
        "prompt_id": "P15",
        "technique": "One-Shot",
        "prompt": """Example: A Python function is a reusable block of code that performs a specific task.
Now explain what a Python function is to a beginner with a simple example."""
    },

    {
        "prompt_id": "P16",
        "technique": "Few-Shot",
        "prompt": """Example 1: Numbers 2, 4, 6. Mean: 4. Median: 4.
Example 2: Numbers 1, 3, 5. Mean: 3. Median: 3.
Now explain the difference between mean and median to a beginner with a simple example."""
    },

    {
        "prompt_id": "P17",
        "technique": "Few-Shot",
        "prompt": """Example 1: Healthy routine: sleep 8 hours, eat balanced meals, drink enough water, and exercise regularly.
Example 2: Healthy routine: maintain regular sleep, eat nutritious food, stay active, and take breaks.
Now suggest a simple healthy daily routine for a student."""
    },

    {
        "prompt_id": "P18",
        "technique": "Few-Shot",
        "prompt": """Example 1: Saving keeps money safe for future needs. Investing puts money into assets that may grow in value.
Example 2: Saving is useful for short-term goals. Investing is generally used for longer-term growth.
Now explain the difference between saving and investing for a beginner."""
    },

    {
        "prompt_id": "P19",
        "technique": "Few-Shot",
        "prompt": """Example 1: Customer received a damaged phone. Response: Apologize, ask for order details and photos, and explain the replacement process.
Example 2: Customer received a damaged book. Response: Apologize, ask for order details and photos, and explain the return or replacement process.
Now write a polite response to a customer who received a damaged product."""
    },

    {
        "prompt_id": "P20",
        "technique": "Few-Shot",
        "prompt": """Example 1: A Python function is reusable code that performs a specific task.
Example 2: A function can accept inputs and return an output.
Now explain what a Python function is to a beginner with a simple example."""
    },

    {
        "prompt_id": "P21",
        "technique": "Role Prompting",
        "prompt": "You are a Teacher. Explain the difference between mean and median to a beginner with a simple example."
    },

    {
        "prompt_id": "P22",
        "technique": "Role Prompting",
        "prompt": "You are a Doctor. Give general advice on maintaining a healthy daily routine for a student."
    },

    {
        "prompt_id": "P23",
        "technique": "Role Prompting",
        "prompt": "You are a Finance Advisor. Explain the difference between saving and investing for a beginner."
    },

    {
        "prompt_id": "P24",
        "technique": "Role Prompting",
        "prompt": "You are a Customer Support Manager. Write a polite response to a customer who received a damaged product and explain the next steps."
    },

    {
        "prompt_id": "P25",
        "technique": "Role Prompting",
        "prompt": "You are an AI Engineer. Explain what a Python function is to a beginner with a simple example."
    },

    {
        "prompt_id": "P26",
        "technique": "Chain-of-Thought",
        "prompt": """A student has 12 notebooks. They buy 5 more notebooks and give 3 notebooks to a friend.
How many notebooks do they have now?
Explain the steps."""
    },

    {
        "prompt_id": "P27",
        "technique": "Chain-of-Thought",
        "prompt": """A shop has 50 pens. It sells 18 pens in the morning and 12 pens in the evening.
How many pens are left?
Explain the steps."""
    },

    {
        "prompt_id": "P28",
        "technique": "Chain-of-Thought",
        "prompt": """A student studies for 2 hours on Monday, 3 hours on Tuesday, and 4 hours on Wednesday.
How many hours did the student study in total?
Explain the steps."""
    }

]

prompts_df = pd.DataFrame(prompts)

prompts_df.to_csv(
    "dataset/home_activity_prompts.csv",
    index=False
)

print("Home activity prompts saved successfully!")

responses = []

for item in prompts:

    print("\nGenerating response for:", item["prompt_id"], "-", item["technique"])

    result = generator(
        item["prompt"],
        max_new_tokens=60,
        temperature=0.7,
        do_sample=True
    )

    generated_text = result[0]["generated_text"]

    response = generated_text[len(item["prompt"]):].strip()

    responses.append({
        "prompt_id": item["prompt_id"],
        "technique": item["technique"],
        "prompt": item["prompt"],
        "generated_response": response
    })

print("\nAll home activity responses generated!")
responses_df = pd.DataFrame(responses)

responses_df.to_csv(
    "dataset/home_activity_responses.csv",
    index=False
)

print("Home activity responses saved successfully!")
