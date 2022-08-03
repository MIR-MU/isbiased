import nltk
import pandas as pd

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm
from nltk import tokenize
import spacy
import re

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


# TODO: maybe divide this into Heuristics = subclasses sharing the same interface?
class ComputeHeuristics:
    # TODO: whole class docstring
    def __init__(self, dataset: pd.DataFrame, tfidf_dataset: pd.DataFrame):
        self.data = dataset
        self.tfidf_dataset = tfidf_dataset

    def count_similar_words_in_question_and_context(self):
        """Function for similar words heuristic computation
        This function tokenize the question and the context into words
        Sets of words are created from both of them
        Intersection between sets is computed

        Args:
            # TODO: not the arg
            data (Pandas Dataframe): dataset for which you want to compute

        Returns:
            # TODO: return annotation
            int: single number for similar words
        """
        tokenizer = nltk.RegexpTokenizer(r"\w+")

        similar_words = []

        for i in range(len(self.data)):
            context1 = nltk.word_tokenize(self.data['context'][i])
            question1 = nltk.word_tokenize(self.data['question'][i])
            context_new = [word for word in context1 if word.isalnum()]
            question_new = [word for word in question1 if word.isalnum()]
            similar_words.append(len(set(context_new).intersection(set(question_new))))

        return similar_words

    # code from this web site https://www.codegrepper.com/code-examples/python/find+index+of+sublist+in+list+python
    def find_sub_list(self, sl, l):
        # TODO: types
        # TODO: public method?
        """Function for finding the index of sublist in list

        Args:
            sl (list): list created from answer text
            l (list): list created from context

        Returns:
            int: index of the sublist
        """
        results = []
        sll = len(sl)
        if sll <= 0:
            return results

        for ind in (i for i, e in enumerate(l) if e == sl[0]):
            if l[ind:ind + sll] == sl:
                results.append((ind, ind + sll - 1))

        return results

    def count_lowest_position_of_word_from_question_in_context(self):
        """Function for the word distance heuristic
        Computes the distance of word from the question from the answer text in the context

        Args:
            data (Pandas Dataframe): dataset

        Returns:
            int, str: distance of the closest word and the word  # TODO: nice, but also use return type annotation
        """
        tokenizer = RegexpTokenizer(r'\w+')
        distances = []
        words = []

        for i in range(len(self.data)):
            indexes_of_words = []
            context_list = tokenizer.tokenize(self.data['context'][i])
            question_list = tokenizer.tokenize(self.data['question'][i])
            answer_text = tokenizer.tokenize(self.data['answers'][i]['text'][0])
            answer_start = self.data['answers'][i]['answer_start']

            indexes_of_words = self.find_sub_list(answer_text, context_list)

            if len(indexes_of_words) > 0:
                answer_index = indexes_of_words[0][0]
            else:
                distances.append(-1)
                words.append('None')
                continue

            filtered_words = [word for word in question_list if word not in stopwords.words('english')]

            list_indexes = {}

            for word in filtered_words:
                if word in context_list:
                    for j in range(len(context_list)):
                        if word == context_list[j]:
                            list_indexes[abs(j - answer_index)] = context_list[j]

            sort_orders = sorted(list_indexes.items(), key=lambda x: x[0], reverse=False)

            if len(sort_orders) == 0:
                distances.append(-1)
                words.append('None')
            else:
                distances.append(sort_orders[0][0])
                words.append(sort_orders[0][1])

        return distances, words

    def identify_in_which_sentence_answer_is(self):
        """Function for the k-th sentence heuristic
        Computes in which sentence the answer is

        Args:
            data (Pandas Dataframe): dataset

        Returns:
            int: number representing the index of the sentence  # TODO: nice, but also use return type annotation
        """
        sentence_indexes = []

        for i in range(len(self.data)):
            context1 = tokenize.sent_tokenize(self.data['context'][i])
            answer = self.data['answers'][i]['text'][0]
            nth = 0
            for sentence in context1:
                if answer in sentence:
                    break
                nth += 1

            sentence_indexes.append(nth)

        return sentence_indexes

    def compute_similarity_between_context_and_question(self):
        """Function for cosine similarity heuristic
        Computes the cosine similarity from TF-IDF representation (trained on train SQuAD) for context and question

        Args:
            data (Pandas Dataframe): dataset

        Returns:
            decimal: number representing the cosine similarity  # TODO: nice, but also use return type annotation
        """
        vectorizer = TfidfVectorizer(stop_words='english', use_idf=True)
        model = vectorizer.fit(self.tfidf_dataset['context'])  # fit on the train dataset

        similarities = []

        for i in tqdm(range(len(self.data))):
            context1 = vectorizer.transform([self.data['context'][i]])
            question1 = vectorizer.transform([self.data['question'][i]])
            similarities.append(cosine_similarity(context1, question1)[0][0])

        return similarities

    def average_answer_length(self):
        """Function for the answer length heuristic
        Computes the average answer length in number of words

        Args:
            data (Pandas Dataframe): dataset

        Returns:
            decimal: number representing the awerage length  # TODO: nice, but also use return type annotation
        """
        answers_text = []
        for i in range(len(self.data)):
            answers_text.append(self.data['answers'][i]['text'])
        answers_text

        answer_length = []
        tokenizer = RegexpTokenizer(r'\w+')
        for i in range(len(self.data)):
            avg_lenght = 0
            for j in range(len(answers_text[i])):
                avg_lenght += len(tokenizer.tokenize(answers_text[i][j]))
            av = avg_lenght / (len(answers_text[i]))
            answer_length.append(av)

        return answer_length

    def show_ents(self, doc):
        """Function for detecting the named entities

        Args:
            doc (Doc): string processed with nlp()

        Returns:
            list: list of entities  # TODO: nice, but also use return type annotation - FOR ALL METHODS, PLS
        """
        entities = []
        if doc.ents:
            for ent in doc.ents:
                entities.append(ent.label_)
        return entities

    def count_similar_NER_from_context_to_answer(self):
        """Function for similar entities heuristic
        Computed the number of similar entities between the answer and context 

        Args:
            data (Pandas Dataframe): dataset

        Returns:
            int: number of similar entities
        """
        context_ents = []
        answer_ents = []
        nlp = spacy.load('en_core_web_sm')  # load spacy

        for row in range(len(self.data)):
            context = nlp(self.data['context'][row])
            context_ents.append(self.show_ents(context))
            for ans in self.data['answers'][row]['text']:
                act_ans_ents_ = []
                act_ans = nlp(ans)
                act_ans_ents_.append(self.show_ents(act_ans))
            answer_ents.append(act_ans_ents_)

        max_sim_ents = []

        for row, cont in zip(answer_ents, context_ents):
            max = 0
            for items in row:
                for item in items:
                    if cont.count(item) > max:
                        max = cont.count(item)
            max_sim_ents.append(max)

        return max_sim_ents

    def doc_pieces(self, doc):
        """Function for sentence subject detection

        Args:
            doc (Doc): string processed with nlp()

        Returns:
            list: list of subjects
        """
        subjects = []
        for ent in doc:
            if ent.dep_ == 'nsubj':
                subjects.append(ent.text)
        return subjects

    def extract_answer_position_with_respect_to_subject(self):
        """Function for the subject position heuristic
        Computes the position of question's subject in the context regarding the index of correct answer

        Args:
            data (Pandas Dataframe): dataset

        Returns:
            int: number representing the answer is before the extracted subject or after the occurence
        """
        q_subjects = []
        nlp = spacy.load('en_core_web_sm')  # load spacy

        for item in self.data['question']:
            question = nlp(item)
            q_subjects.append(self.doc_pieces(question))

        positions = []

        for context, q_sub, answer in zip(self.data['context'], q_subjects, self.data['answers']):
            pos = 0
            max = 0
            for item in q_sub:
                if item in context:
                    indexes = [m.start() for m in re.finditer(item, context)]
                    counter = 0
                    for index in indexes:
                        if answer['answer_start'][0] < index:
                            break
                        else:
                            counter += 1
                    if max < counter:
                        max = counter
                    pos = max
                else:
                    pos = -1
            positions.append(pos)

        return positions

    def compute_all_heuristics(self):
        # TODO: docs
        self.data['similar_words'] = self.count_similar_words_in_question_and_context()
        self.data['distances'], self.data[
            'closest_words'] = self.count_lowest_position_of_word_from_question_in_context()
        self.data['kth_sentence'] = self.identify_in_which_sentence_answer_is()
        self.data['cosine_similarity'] = self.compute_similarity_between_context_and_question()
        self.data['answer_length'] = self.average_answer_length()
        self.data['max_sim_ents'] = self.count_similar_NER_from_context_to_answer()
        self.data['answer_subject_positions'] = self.extract_answer_position_with_respect_to_subject()

    def compute_heuristic(self, heuristic_name: str):
        # TODO: docs
        if heuristic_name == 'similar_words':
            self.data['similar_words'] = self.count_similar_words_in_question_and_context()
        elif heuristic_name == 'distances':
            self.data['distances'], self.data[
                'closest_words'] = self.count_lowest_position_of_word_from_question_in_context()
        elif heuristic_name == 'kth_sentence':
            self.data['kth_sentence'] = self.identify_in_which_sentence_answer_is()
        elif heuristic_name == 'cosine_similarity':
            self.data['cosine_similarity'] = self.compute_similarity_between_context_and_question()
        elif heuristic_name == 'answer_length':
            self.data['answer_length'] = self.average_answer_length()
        elif heuristic_name == 'max_sim_ents':
            self.data['max_sim_ents'] = self.count_similar_NER_from_context_to_answer()
        elif heuristic_name == 'answer_subject_positions':
            self.data['answer_subject_positions'] = self.extract_answer_position_with_respect_to_subject()

    def save_dataset_with_computed_heuristics(self, name: str):
        # TODO: docs
        self.data.to_json(f"{name}_with_computed_heuristics.json", orient='records')
