from transformers import pipeline
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Step 1: Run text generation pipeline with the desired prompt
prompt = "Are patients with schizophrenia crazy?"
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
result = pipe(f"<s>[INST] {prompt} [/INST]")  # Generate text based on the prompt
generated_text = result[0]['generated_text']
print("Generated Text:\n", generated_text)

# Step 2: Define the reference text(s) for BLEU score calculation
# This should be an accurate reference text or texts that resemble the ideal output for the prompt.
reference_text = """
Due to incorrect representation of the disease in media & books, there is a myth that schizophrenics are violent. The truth is most schizophrenics are docile and keep to themselves. The schizophrenics who have demonstrated bursts of violence are either in an acute stage of psychosis or are abusing an addictive substance.
"""

# Convert reference text to a list of tokenized words
references = [reference_text.split()]

# Step 3: Tokenize the generated text for BLEU score calculation
generated_text_tokens = generated_text.split()

# Step 4: Calculate BLEU Score
smooth_fn = SmoothingFunction().method4  # Smoothing function to handle short texts and avoid zero scores
bleu_score = sentence_bleu(references, generated_text_tokens, smoothing_function=smooth_fn)

print(f"BLEU Score: {bleu_score:.4f}")
