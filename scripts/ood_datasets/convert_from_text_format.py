import json

from datasets import load_dataset


def dump_sample(id, context, question, answers):
    out_sample = {"id": id,
                  "context": context,
                  "question": question,
                  "answers": {
                      "text": answers,
                      "answer_start": [context.find(a) for a in answers]
                  }}

    return out_sample


d = load_dataset("lucadiliello/newsqa", split="validation").select(range(2000))
out = d.map(lambda row: dump_sample(row["key"], row["context"], row["question"], row["answers"]))
for col_name in out.column_names:
    if col_name not in ["id", "context", "question", "answers"]:
        out = out.remove_columns([col_name])


dataset_as_list = out.to_dict()
dataset_as_list = [dict(zip(dataset_as_list, t)) for t in zip(*dataset_as_list.values())]

json.dump(dataset_as_list, open("news_qa_dev_2k_subset.json", "w"))


d = load_dataset("lucadiliello/searchqa", split="validation").select(range(2000))
out = d.map(lambda row: dump_sample(row["key"], row["context"], row["question"], row["answers"]))
for col_name in out.column_names:
    if col_name not in ["id", "context", "question", "answers"]:
        out = out.remove_columns([col_name])


dataset_as_list = out.to_dict()
dataset_as_list = [dict(zip(dataset_as_list, t)) for t in zip(*dataset_as_list.values())]

json.dump(dataset_as_list, open("search_qa_dev_2k_subset.json", "w"))

print()
