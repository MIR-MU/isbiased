import torch.cuda
from datasets import load_metric, load_dataset
from tqdm import tqdm
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

model_path = "/Users/xstefan3/PycharmProjects/isbiased/models/biased/distances/bert-base-cased"
split = "validation"

# dataset = load_dataset("adversarial_qa", "adversarialQA")[split].select(range(100))
dataset = load_dataset("squad")[split].select(range(100))

device = "cuda" if torch.cuda.is_available() else "cpu"

predictions = []
references = []

model = AutoModelForQuestionAnswering.from_pretrained(model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)

for sample in tqdm(dataset):
    inputs = tokenizer(sample["question"], text_pair=sample["context"], return_tensors="pt", truncation=True)
    outputs = model(**inputs.to(device))
    prediction = tokenizer.decode(inputs["input_ids"][0, outputs.start_logits.argmax(-1)[0]:
                                                         outputs.end_logits.argmax(-1)[0]])
    predictions.append({"id": sample["id"], "prediction_text": prediction.strip()})
    references.append({"id": sample["id"], "answers": sample["answers"]})

metric = load_metric("squad")
evaluations = metric.compute(predictions=predictions, references=references)
print(evaluations)
print()
