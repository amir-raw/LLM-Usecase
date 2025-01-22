
# Load the dataset
das = ds
#dataset_name = "mlabonne/guanaco-llama2-1k"
#dataset = load_dataset(dataset_name, split='train')

# Prepare the input texts
input_texts = []
for item in das:
    instruction = item.get('instruction', '')
    input_text = instruction.strip()
    if 'input' in item and item['input']:
        input_text += '\n' + item['input'].strip()
    response = item.get('output', '')
    # Combine instruction and response for perplexity computation
    full_text = input_text + '\n' + response.strip()
    input_texts.append(full_text)

# Join all input texts into a single string with separator tokens
full_text = '\n\n'.join(input_texts)

# Tokenize the full text
encodings = tokenizer(full_text, return_tensors='pt')


from tqdm import tqdm

# Get model's max position embeddings
max_length = model.config.max_position_embeddings
stride = 512
seq_len = encodings.input_ids.size(1)

nlls = []
prev_end_loc = 0

for begin_loc in tqdm(range(0, seq_len, stride)):
    end_loc = min(begin_loc + max_length, seq_len)
    trg_len = end_loc - begin_loc  # Length of this segment

    input_ids = encodings.input_ids[:, begin_loc:end_loc]
    target_ids = input_ids.clone()

    # Mask tokens before the current segment
    if begin_loc > 0:
        target_ids[:, :begin_loc] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)
        neg_log_likelihood = outputs.loss * trg_len

    nlls.append(neg_log_likelihood)

    if end_loc == seq_len:
        break

# Compute the mean negative log-likelihood
total_nll = torch.stack(nlls).sum()
ppl = torch.exp(total_nll / seq_len)

print(f"Perplexity: {ppl.item():.2f}")
