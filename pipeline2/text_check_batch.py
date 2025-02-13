from transformers import BartForSequenceClassification, BartTokenizer
import torch

def textcheck(batch_premise):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-mnli')
    model = BartForSequenceClassification.from_pretrained('facebook/bart-large-mnli').to(device)

    hypothesis = 'harmful.'

    # Tokenize all premises as a batch
    inputs = tokenizer(batch_premise, [hypothesis] * len(batch_premise), 
                       return_tensors='pt', padding=True, truncation=True).to(device)
    
    # Forward pass
    with torch.no_grad():
        result = model(**inputs)
    
    logits = result.logits  # result[0] is now deprecated, use result.logits instead

    # Take only entailment and contradiction logits
    entail_contradiction_logits = logits[:, [0, 2]]
    probs = entail_contradiction_logits.softmax(dim=1)
    true_probs = probs[:, 1] * 100  # Probability of entailment

    for i, prob in enumerate(true_probs.tolist()):
        print(f'Premise {i+1}: Probability that the label is true: {prob:.2f}%')

    return true_probs.tolist()

# Example batch
premise = 'A new model offers an explanation for how the Galilean satellites formed around the solar system’s largest world.'
batch_premise = [premise, premise, premise]
textcheck(batch_premise)