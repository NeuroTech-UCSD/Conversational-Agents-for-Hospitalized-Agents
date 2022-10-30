from tensorflow import keras
import os
import sys
from bert.tokenization.bert_tokenization import FullTokenizer
import pickle
import tensorflow as tf

from src.intent_data import IntentDetectionData
from src.model_utils import generate_class_weights, create_model, generate_dataset

# ============ Constants ================
bert_model_name = "uncased_L-12_H-768_A-12"
# specify where the  check point directory is
bert_ckpt_dir = os.path.join("tmp/model/", bert_model_name)
# specify the checkpoint file itself
bert_ckpt_file = os.path.join(bert_ckpt_dir, "bert_model.ckpt")
# specify the configuration file
bert_config_file = os.path.join(bert_ckpt_dir, "bert_config.json")
os.makedirs('models', exist_ok=True)

# ============ Tokenizer & Dataset ===================
bert_tokenizer = FullTokenizer(vocab_file=os.path.join(bert_ckpt_dir, "vocab.txt"))
# compiling the model
dataset = generate_dataset()
train = dataset['train']
test = dataset['test']
classes = dataset['classes']
data = IntentDetectionData(train, test, bert_tokenizer, classes)
# ============ Compile Model ==========================
bert_model = create_model(data.max_seq_len, bert_ckpt_file, bert_config_file, num_classes=len(classes))
bert_model.compile(
    # optimizer is Adam with learning rate 0.00001 ,  this is recommended by authors of bert model paper
    optimizer=keras.optimizers.Adam(1e-5),
    # using SparseCategoricalCrossentropy
    # because we have not used OneHotencoding here
    loss=keras.losses.SparseCategoricalCrossentropy(),
    # specify the metrics from keras
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")]
)
# ================== Training ======================
class_weights = generate_class_weights(data.train_y.tolist(), data.classes)
# finally fit the model
history = bert_model.fit(
    x=data.train_x,
    y=data.train_y,
    # validation split to be 10% of the data
    validation_split=0.1,
    # recommendation from authors , batch to be 16
    batch_size=16,
    # yes we want to shuffle
    shuffle=True,
    # another recommendation from authors (epochs = 5)
    epochs=5,
    class_weight=class_weights,
    # specify the call back
    callbacks=[]
)
# bert_model.save('./models/Bert.h5')
tf.saved_model.save('./models/bert')
# os.makedirs('logs', exist_ok=True)
# with open('logs/history.pkl', 'wb') as f:
#     pickle.dump(history, f)