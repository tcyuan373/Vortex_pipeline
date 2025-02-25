from transformers import BartForSequenceClassification, BartTokenizer
import torch
import csv

def textcheck(batch_premise, run_times):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-mnli')
    model = BartForSequenceClassification.from_pretrained('facebook/bart-large-mnli').to(device)

    hypothesis = 'harmful.'

    # Tokenize all premises as a batch
    inputs = tokenizer(batch_premise, [hypothesis] * len(batch_premise), 
                       return_tensors='pt', padding=True, truncation=True).to(device)
    
    model_start_event = torch.cuda.Event(enable_timing=True)
    model_end_event = torch.cuda.Event(enable_timing=True)

    # time before running model
    model_start_event.record()

    # Forward pass
    with torch.no_grad():
        result = model(**inputs)

    # time after running model
    model_end_event.record()
    torch.cuda.synchronize()
    run_times.append((model_start_event.elapsed_time(model_end_event)) * 1e6)
    
    logits = result.logits  # result[0] is now deprecated, use result.logits instead

    # Take only entailment and contradiction logits
    entail_contradiction_logits = logits[:, [0, 2]]
    probs = entail_contradiction_logits.softmax(dim=1)
    true_probs = probs[:, 1] * 100  # Probability of entailment

    for i, prob in enumerate(true_probs.tolist()):
        print(f'Premise {i+1}: Probability that the label is true: {prob:.2f}%')

    return true_probs.tolist()


if __name__ == "__main__":
    # Example batch
    premise = 'A new model offers an explanation for how the Galilean satellites formed around the solar system’s largest world.'
    batch_premise = [premise, premise, premise]
    
    run_times = []
    for i in range(1000):
        textcheck(batch_premise, run_times)

    runtimes_file = 'text_check_batch_runtime.csv'

    with open(runtimes_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(run_times)