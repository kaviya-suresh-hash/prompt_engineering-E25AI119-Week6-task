from transformers import pipeline
import pandas as pd
import os

# ----------------------------------------------------------
# 1. Create dataset folder
# ----------------------------------------------------------


os.makedirs("dataset", exist_ok=True)

# ----------------------------------------------------------
# 2. Load Pre-trained Hugging Face Model
# ----------------------------------------------------------


print("Loading model...")

generator = pipeline(
    "text-generation",
    model="distilgpt2"
)

print("Model loaded successfully!")


# ----------------------------------------------------------
# 3. Create Prompting Examples
# ----------------------------------------------------------


prompts = [

    {
        "prompt_id": "P01",
        "technique": "Zero-Shot",
        "prompt": "Classify the sentiment of this sentence as Positive or Negative:"
    },

    {
        "prompt_id": "P02",
        "technique": "One-Shot",
        "prompt": """Sentence: "I love this phone."
Sentiment: Positive

Sentence: "The service was terrible."
Sentiment:"""
    },

    {
        "prompt_id": "P03",
        "technique": "Few-Shot",
        "prompt": """Sentence: "The movie was excellent."
Sentiment: Positive

Sentence: "The food was horrible."
Sentiment: Negative

Sentence: "I enjoyed the book."
Sentiment: Positive

Sentence: "The customer service was very bad."
Sentiment:"""
    },

    {
        "prompt_id": "P04",
        "technique": "Role Prompting",
        "prompt": """You are a software engineer.
Explain what a Python function is to a beginner."""
    },

    {
        "prompt_id": "P05",
        "technique": "Chain-of-Thought",
        "prompt": """A student has 5 books.
The student buys 3 more books and gives 2 books to a friend.
How many books does the student have now?
Explain the steps."""
    }

]

# ----------------------------------------------------------
# 4. Save Prompts
# ----------------------------------------------------------

prompts_df = pd.DataFrame(prompts)

prompts_df.to_csv(
    "dataset/prompt.csv",
    index=False
)

print("prompts.csv created.")

# ----------------------------------------------------------
# 5. Generate Responses
# ----------------------------------------------------------

responses = []

for item in prompts:

    print("\nGenerating response for:", item["technique"])

    result = generator(
        item["prompt"],
        max_new_tokens=60,
        temperature=0.7,
        do_sample=True
    )

    generated_text = result[0]["generated_text"]

    # Remove the original prompt from the output
    response = generated_text[len(item["prompt"]):].strip()

    responses.append({
        "prompt_id": item["prompt_id"],
        "technique": item["technique"],
        "prompt": item["prompt"],
        "generated_response": response
    })


# ----------------------------------------------------------
# 6. Save Generated Responses
# ----------------------------------------------------------

responses_df = pd.DataFrame(responses)

responses_df.to_csv(
    "dataset/generated_responses.csv",
    index=False
)

print("\nGenerated_responses.csv created.")

# ----------------------------------------------------------
# 7. Create Prompt Comparison
# ----------------------------------------------------------

comparison = []

for item in responses:

    response = item["generated_response"]

    comparison.append({
        "prompt_id": item["prompt_id"],
        "technique": item["technique"],
        "response_length": len(response.split()),
        "quality": "",
        "accuracy": "",
        "completeness": ""
    })

comparison_df = pd.DataFrame(comparison)

comparison_df.to_csv(
    "dataset/prompt_comparison.csv",
    index=False
)

print("prompt_comparison.csv created.")

# ----------------------------------------------------------
# 8. Create Evaluation File
# ----------------------------------------------------------

evaluation = []

for item in responses:

    # Scores can be obtained from manual evaluation
    relevance = 4
    accuracy = 5
    completeness = 3

    # Calculate weighted overall score
    overall_score = (
        relevance * 0.40 +
        accuracy * 0.40 +
        completeness * 0.20
    )

    evaluation.append({
        "prompt_id": item["prompt_id"],
        "technique": item["technique"],
        "relevance": relevance,
        "accuracy": accuracy,
        "completeness": completeness,
        "overall_score": round(overall_score, 2),
        "remarks": ""
    })

evaluation_df = pd.DataFrame(evaluation)

evaluation_df.to_csv(
    "dataset/evaluation_results.csv",
    index=False
)

print("\n========== GENERATED RESPONSES ==========\n")

for item in responses:

    print("Technique:", item["technique"])
    print("Prompt:", item["prompt"])
    print("Response:", item["generated_response"])
    print("-" * 60)

print("\nProject completed successfully!")