from bert.tokenization.bert_tokenization import FullTokenizer
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, pipeline
import numpy as np
import tensorflow as tf
import pandas as pd
import os

from src.intent_data import IntentDetectionData
from src.cloud_utils import cloud_predict

# ============== Models ================
bert_f = None
if os.path.exists('models/bert'):
    bert = tf.saved_model.load('models/bert')
    bert_f = bert.signatures["serving_default"]
if os.path.exists('models/bart_large'):
    bart_large = AutoModelForSequenceClassification.from_pretrained('models/bart_large')
else:
    bart_large = AutoModelForSequenceClassification.from_pretrained('facebook/bart-large-mnli')

# =============== Tokenizers ===========
MAX_SEQ_LEN = IntentDetectionData.MAX_SEQ_LEN  # careful, should match earlier
bert_tokenizer = FullTokenizer(vocab_file="data/vocab.txt")
bart_large_tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-mnli')


# ================ Constants ===========
class Response():
    def __init__(self, custom, value=""):
        '''
        Args:
          custom: if true then the subintent requires a custom response (have to be modified in the GUI function)
          value: the response for the subintent
        '''
        self.custom = custom
        self.value = value


# the intents that can be classified by bert
CLASSES = ['food', 'get help', 'make call', 'out_of_scope', 'send text', 'visitors']

# subintents that can be classified by bart-large (zero-shot learning)
SUBCLASSES = {"food": {"when is meal": Response(False, "Your next meal is in 2 hours"),
                       "get food menu": Response(False, "Here is the food menu"),
                       "get drink menu": Response(False, "Here is the drink menu"),
                       "order food or drink": Response(True)},
              "get help": {"get medical assistance": Response(False, "Calling for medical assistance..."),
                           "get 911": Response(False, "Calling 911..."),
                           "help changing clothes": Response(False, "Getting the nurse to help you get dressed..."),
                           "help going to bathroom": Response(False,
                                                              "Getting the nurse to help you to the bathroom...")},
              "make call": {"make call": Response(True)},
              "send text": {"send text": Response(True)},
              "visitors": {"visiting hours": Response(False, "Visiting hours are set for 3 to 5"),
                           "who is visiting": Response(False,
                                                       "The scheduled visitors are Josue, Michael, Ivy, and David"),
                           "visiting schedule": Response(False, "Here is the visiting schedule")}}

# the food that are offered in provided at the hospital in a particular day
FOODS = ["spaghetti", "burritos", "nacho cheese", "bread", "fruit", "soup",
         "salad", "meatloaf", "pizza", "milk", "juice", "tea", "coffee", "lemonade",
         "jelly", "chips", "hamburger", "yogurt", "eggs", "water", "pasta", "bagel", "tacos"]

# contacts of the patient
CONTACTS = ["mom", "dad", "brother", "sister", "william", "olivia", "emma", "ava", "charlotte", "sophia",
            "amelia", "isabella", "mia", "evelyn", "harper", "camila", "gianna",
            "abigail", "luna", "ella", "elizabeth", "sofia", "emily", "avery",
            "mila", "scarlett", "eleanor", "madison", "layla", "penelope", "aria",
            "chloe", "grace", "ellie", "nora", "hazel", "zoey", "riley", "victoria",
            "lily", "aurora", "violet", "nova", "hannah", "emilia", "zoe", "stella",
            "everly", "isla", "leah", "lillian", "addison", "willow", "lucy", "david"]


# ============== Predict helper functions ======
def get_bert_predictions(sentences, classes=CLASSES, tokenizer=bert_tokenizer, max_seq_len=MAX_SEQ_LEN):
    '''
    Args:
      sentences (list): unprocessed texts to be passed in
      classes (list): the labels of the classes
    '''
    pred_tokens = map(tokenizer.tokenize, sentences)
    pred_tokens = map(lambda tok: ["[CLS]"] + tok + ["[SEP]"], pred_tokens)
    pred_token_ids = list(map(tokenizer.convert_tokens_to_ids, pred_tokens))
    pred_token_ids = map(lambda tids: tids + [0] * (max_seq_len - len(tids)), pred_token_ids)
    pred_token_ids = list(pred_token_ids)
    if bert_f:
        pred_token_ids = np.array(pred_token_ids)
        predictions = bert_f(tf.convert_to_tensor(pred_token_ids))['dense_1'].numpy()
        predictions = predictions.argmax(axis=-1)
    else:
        # print('inputs:', tf.convert_to_tensor(pred_token_ids))
        # predictions = cloud_predict({1: [list(pred_token_ids)]})
        predictions = cloud_predict(
            instances=pred_token_ids
        )

        predictions = np.array(predictions.predictions).argmax(axis=-1)
    return list(map(lambda index: classes[index], predictions))


def get_bart_large_predictions(sentences, classes, model=bart_large, tokenizer=bart_large_tokenizer):
    '''
    Args:
      sentences (list or str): unprocessed texts to be passed in
      classes (list): the labels of the classes
    '''
    zsc = pipeline(task='zero-shot-classification', tokenizer=tokenizer, model=model)
    return zsc(sequences=sentences, candidate_labels=classes, multi_label=False)


def find_name(names, sentence):
    '''
    Check if a name is in the list; if so return name otherwise "unknown"
    '''
    response = "unknown"
    for name in names:
        if name in sentence:
            response = name
            break
    return response


def GUI(sentence, classes=CLASSES, contacts=CONTACTS, foods=FOODS):
    '''
      Args:
        sentence (str):
        contacts (list): contact names of the patient
        foods (list): food provided at the hospital in a particular day
    '''
    sentence = sentence.lower()
    intent = get_bert_predictions([sentence], classes=classes)[0]
    response = ""
    if intent == "out_of_scope":
        response = "Sorry, I did not understand"
        subintent = "out_of_scope"
    else:
        subintent = get_bart_large_predictions(sentence, classes=[*SUBCLASSES[intent]])["labels"][0]
        response_obj = SUBCLASSES[intent][subintent]
        if response_obj.custom is True:
            if subintent == 'order food or drink':
                food = find_name(foods, sentence)
                if response == "unknown":
                    response = "Sorry, that item is currently not on our menu. Here is our menu."
                else:
                    response = f"Okay, I will order {food}"
            elif subintent == 'make call':
                contact = find_name(contacts, sentence)
                if contact == "unknown":
                    response = "Sorry, I couldn't find that person in your contacts."
                else:
                    response = f"Calling {contact}..."
            elif subintent == 'send text':
                contact = find_name(contacts, sentence)
                if contact == "unknown":
                    response = "Sorry, I couldn't find that person in your contacts."
                else:
                    response = f"Drafing text to {contact}..."
        else:
            response = response_obj.value
    return dict(query=sentence, response=response, intent=intent, subintent=subintent)
