import numpy as np
import tensorflow as tf
from tensorflow import keras
from bert.loader import StockBertConfig, map_stock_config_to_params, load_stock_weights
from bert import BertModelLayer
from datasets import load_dataset
import pandas as pd
import subprocess
import json
import os


def generate_dataset():
    dataset = load_dataset(
        'clinc_oos', 'small')

    # Takes 5 examples from the first 100 intents in the training dataset
    df = pd.DataFrame(dataset["train"])
    dict_result = {"intent": "out_of_scope", "text": []}
    for i in range(100):
        temp_list = list(df[df["intent"] == i]["text"][:5].values)
        for j in temp_list:
            dict_result["text"].append(j)
    OOS_df = pd.DataFrame(dict_result)
    OOS_df['text'] = OOS_df['text'].apply(lambda x: x.lower())

    os.makedirs('tmp/output', exist_ok=True)
    subprocess.run(['python', '-m', 'chatette', 'data/chatette_text.txt', '-o', 'tmp/output', '-f'])
    file_data = open("tmp/output/train/output.json")
    data = json.load(file_data)

    orig_intent = np.empty([len(data['rasa_nlu_data']['common_examples']), 1], dtype=object)
    orig_text = np.empty([len(data['rasa_nlu_data']['common_examples']), 1], dtype=object)
    for i, item in enumerate(data['rasa_nlu_data']['common_examples']):
        orig_intent[i] = item['intent'].lower()
        orig_text[i] = item['text'].lower()

    df = pd.DataFrame(np.concatenate([orig_text, orig_intent], axis=1), columns=['text', 'intent'])
    # strip underscore of the intents
    df['intent'] = df['intent'].apply(lambda x: x.replace('_', ' '))
    # combine generated intent dataset with OOS dataset
    df = pd.concat([df, OOS_df], axis=0)
    df = df.sample(frac=1).reset_index(drop=True)

    # taking unique intents and convert them to list
    classes = sorted(df['intent'].unique().tolist())
    classes_without_oos = classes.copy()
    classes_without_oos.remove('out_of_scope')
    classes_gate = ['in_scope', 'out_of_scope']
    print('classes: ', classes)
    # print(classes_without_oos)
    # print(classes_gate)

    train_frac = 0.7
    valid_frac = 0
    test_frac = 0.3
    assert train_frac + valid_frac + test_frac <= 1, 'make sure train_frac + valid_frac + test_frac <= 1'

    train_valid_split = int(train_frac * len(df))
    valid_test_split = train_valid_split + int(valid_frac * len(df))
    train = df.iloc[:train_valid_split]
    valid = df.iloc[train_valid_split:valid_test_split]
    test = df.iloc[valid_test_split: int(test_frac * len(df)) + valid_test_split]

    def convert_in_scope(x):
        if x != 'out_of_scope':
            x = 'in_scope'
        return x

    df_gate = df.copy()
    df_gate['intent'] = df_gate['intent'].apply(convert_in_scope)
    train_gate = df_gate.iloc[:train_valid_split]
    valid_gate = df_gate.iloc[train_valid_split:valid_test_split]
    test_gate = df_gate.iloc[valid_test_split: int(test_frac * len(df)) + valid_test_split]

    train_without_oos = train[train['intent'] != 'out_of_scope']
    valid_without_oos = valid[valid['intent'] != 'out_of_scope']
    test_without_oos = test[test['intent'] != 'out_of_scope']

    return {'train': train, 'valid': valid, 'test': test, 'classes': classes}


def generate_class_weights(y, classes):
    '''
    Args:
      y (int array): (n_samples,) from 0, ..., n_classes
    '''
    weights = len(y) / (len(classes) * np.bincount(y))
    class_weights = {}
    for i, class_label in enumerate(classes):
        class_weights[i] = weights[i]
    return class_weights


def create_model(max_seq_len, bert_ckpt_file, bert_config_file, num_classes):
    # open the config file using tensorflow for reading
    with tf.io.gfile.GFile(bert_config_file, "r") as reader:
        bc = StockBertConfig.from_json_string(reader.read())
        bert_params = map_stock_config_to_params(bc)
        bert_params.adapter_size = None
        bert = BertModelLayer.from_params(bert_params, name="bert")
    # create the input layer with seq. len as input
    input_ids = keras.layers.Input(shape=(max_seq_len,), dtype='int32', name="input_ids")
    bert_output = bert(input_ids)
    cls_out = keras.layers.Lambda(lambda seq: seq[:, 0, :])(bert_output)
    cls_out = keras.layers.Dropout(0.5)(cls_out)
    logits = keras.layers.Dense(units=768, activation="tanh")(cls_out)
    logits = keras.layers.Dropout(0.5)(logits)
    logits = keras.layers.Dense(units=num_classes, activation="softmax")(logits)
    model = keras.Model(inputs=input_ids, outputs=logits)
    model.build(input_shape=(None, max_seq_len))
    load_stock_weights(bert, bert_ckpt_file)
    return model
