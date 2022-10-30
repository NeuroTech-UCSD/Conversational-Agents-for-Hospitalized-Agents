import numpy as np
from bert.tokenization.bert_tokenization import FullTokenizer
from tqdm import tqdm

# pre processing in which we have to use the tokenizer
# for getting tokens so that we can feed the model
# creating a class which is a little generic for doing our tokenizing process
class IntentDetectionData:
    # we had two columns to which have given names according to use
    DATA_COLUMN = "text"
    LABEL_COLUMN = "intent"
    MAX_SEQ_LEN = 30
    '''
    creating constructor having train data , test data , tokenzier , max_seq_len because NLP tasks work on a fixed 
    number of elements 
    the data set we are having has different lengths of sequences , that's why we have to do same length for all  , 
    either we can cut the sequence 
    lengths to some minimal point or we can add padding up to the maximum existing 
    '''

    def __init__(self, train, test, tokenizer: FullTokenizer, classes, max_seq_len_force=True):
        self.tokenizer = tokenizer
        # setting initial to 0
        self.max_seq_len = 0
        self.classes = classes
        '''
        create train_X and train_Y 
        train_X will have the vectors , train_Y will have the target labels 
        same we will do for the testing part
        '''
        # train, test = map(lambda df: df.reindex(df[IntentDetectionData.DATA_COLUMN].str.len().sort_values().index),
        # [train, test])
        # we gave to set the variables to which we can map values using the above function
        ((self.train_x, self.train_y), (self.test_x, self.test_y)) = map(self._prepare, [train, test])
        # setting the actual value to max_seq_len
        if max_seq_len_force:
            self.max_seq_len = self.MAX_SEQ_LEN
        else:
            self.max_seq_len = min(self.max_seq_len, self.MAX_SEQ_LEN)
        # adding padding in the following line
        self.train_x, self.test_x = map(self._pad, [self.train_x, self.test_x])

    def _prepare(self, df):
        # initialising two empty arrays
        x, y = [], []
        # iterate over each row and use tqdm to visualize that
        for _, row in tqdm(df.iterrows()):
            # extracting the seq. and labels
            text, label = row[IntentDetectionData.DATA_COLUMN], row[IntentDetectionData.LABEL_COLUMN]
            # create an object named token and initialise it
            tokens = self.tokenizer.tokenize(text)
            # adding two special tokens which will surround each token
            # from front and back
            tokens = ["[CLS]"] + tokens + ["[SEP]"]
            # convert the token ids to numbers using a helper
            # function already provided for feeding
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            # calculate the length of seq. by taking max.
            self.max_seq_len = max(self.max_seq_len, len(token_ids))
            # appends the token ids to x
            x.append(token_ids)
            # indexes of labels to the y
            y.append(self.classes.index(label))
        # returning these in the form of nparrays
        return np.array(x), np.array(y)

    # takes the ids and not the tokens itself
    def _pad(self, ids):
        # create an empty list again
        x = []
        # iterating over each id
        for input_ids in ids:
            # we have to cut the longer seq. , self.max_seq_len - 2
            # because two positions are reserved for the special tokens as mentioned before
            input_ids = input_ids[:min(len(input_ids), self.max_seq_len - 2)]
            # adding padding to the smaller ones
            input_ids = input_ids + [0] * (self.max_seq_len - len(input_ids))
            # appending
            x.append(np.array(input_ids))
        return np.array(x)
