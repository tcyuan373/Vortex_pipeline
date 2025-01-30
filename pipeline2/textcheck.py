from transformers import BartForSequenceClassification, BartTokenizer, pipeline
import time
import torch

start = time.perf_counter()
device = torch.device('cuda')
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-mnli', device_map = device)
model = BartForSequenceClassification.from_pretrained('facebook/bart-large-mnli').to('cuda')

# pose sequence as a NLI premise and label (politics) as a hypothesis
# premise = 'I hate Asians!'
premise = 'A new model offers an explanation for how the Galilean satellites formed around the solar system’s largest world.'
hypothesis = 'harmful.'

# run through model pre-trained on MNLI
input_ids = tokenizer.encode(premise, hypothesis, return_tensors='pt').to(device)
logits = model(input_ids)[0]

# we throw away "neutral" (dim 1) and take the probability of
# "entailment" (2) as the probability of the label being true 
entail_contradiction_logits = logits[:,[0,2]]
probs = entail_contradiction_logits.softmax(dim=1)
true_prob = probs[:,1].item() * 100
print(f'Probability that the label is true: {true_prob:0.2f}%')

end = time.perf_counter()

print(f'the model runtime: {(end - start):0.5f}s')

# start = time.perf_counter()

# classifier = pipeline("zero-shot-classification",
#                       model="facebook/bart-large-mnli")

# sent = 'A new model offers an explanation for how the Galilean satellites formed around the solar system’s largest world. Konstantin Batygin did not set out to solve one of the solar system’s most puzzling mysteries when he went for a run up a hill in Nice, France. Dr. Batygin, a Caltech researcher, best known for his contributions to the search for the solar system’s missing “Planet Nine,” spotted a beer bottle. At a steep, 20 degree grade, he wondered why it wasn’t rolling down the hill. He realized there was a breeze at his back holding the bottle in place. Then he had a thought that would only pop into the mind of a theoretical astrophysicist: “Oh! This is how Europa formed.” Europa is one of Jupiter’s four large Galilean moons. And in a paper published Monday in the Astrophysical Journal, Dr. Batygin and a co-author, Alessandro Morbidelli, a planetary scientist at the Côte d’Azur Observatory in France, present a theory explaining how some moons form around gas giants like Jupiter and Saturn, suggesting that millimeter-sized grains of hail produced during the solar system’s formation became trapped around these massive worlds, taking shape one at a time into the potentially habitable moons we know today.'
# labels = ['space & cosmos', 'scientific discovery', 'microbiology', 'robots', 'archeology']
# scores = classifier(sent, labels, multi_label=True)['scores']
# print(f'scores:{scores}')

# end = time.perf_counter()

# print(f'second model runtime: {(end-start):0.5f}s')



# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import torch


# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# model_name = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSequenceClassification.from_pretrained(model_name)

# premise = "Angela Merkel ist eine Politikerin in Deutschland und Vorsitzende der CDU"
# hypothesis = "Emmanuel Macron is the President of France"

# input = tokenizer(premise, hypothesis, truncation=True, return_tensors="pt")
# print(input)
# output = model(input["input_ids"].to(device))  # device = "cuda:0" or "cpu"
# prediction = torch.softmax(output["logits"][0], -1).tolist()
# label_names = ["entailment", "neutral", "contradiction"]
# prediction = {name: round(float(pred) * 100, 1) for pred, name in zip(prediction, label_names)}
# print(prediction)