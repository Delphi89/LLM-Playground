# uses Llama (installed locally) with deepval in order to evaluate responses to 20 questions given in a .csv file

import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from rouge_score import rouge_scorer

# Custom Rouge Metric to evaluate generated responses
class RougeMetric(BaseMetric):
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)

    def measure(self, test_case: LLMTestCase):
        prediction = test_case.actual_output
        target = test_case.expected_output

        # Calculate the ROUGE score
        self.score = self.scorer.score(target, prediction)['rouge1'].fmeasure
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase):
        return self.measure(test_case)

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "Rouge Metric"

# Initialize model and pipeline
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
pipeline = pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

# Load CSV file without headers, treating the first two columns as question and answer
data = pd.read_csv("LLM-test.csv", header=None, delimiter=";")

# Initialize metric and list for storing individual scores
metric = RougeMetric()
scores = []

# Iterate through each question-answer pair in the CSV
for idx, row in data.iterrows():
    question = row[0]  # First column as question
    expected_output = row[1]  # Second column as answer
    answer_source = row[2]  # Third column will not be used
    
    # Set up message prompt for the model
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": question}
    ]

    # Prepare the prompt
    prompt = pipeline.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    # Generate response from the model
    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    outputs = pipeline(
        prompt,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )

    # Extract the generated answer
    print("")
    print(f"Question {idx + 1}: {question}")
    llm_answer = outputs[0]["generated_text"][len(prompt):].strip()
    print(f"Generated Answer for question {idx + 1}: {llm_answer}")

    # Create test case and calculate the metric score
    test_case = LLMTestCase(input=prompt, actual_output=llm_answer, expected_output=expected_output)
    score = metric.measure(test_case)
    scores.append(score)
    print(f"Score for question {idx + 1}: {score}")

# Calculate and display the average score
average_score = sum(scores) / len(scores) if scores else 0
print(f"Average Score: {average_score}")
