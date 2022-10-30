import pandas as pd
from src.predict_utils import GUI

sentences = ["can I get tacos",
             "can you open the window",
             "call dad please",
             "send david a text",
             "open the window would you",
             "can you get the nurse to help me change cloth",
             "get nurse nancy to help me go to the bathroom",
             "any one visiting me today",
             "when's today's visiting hours",
             "I'm hungry",
             "call the doctor please",
             "call 911 now",
             "can you get the nurse",
             "can you call nora",
             "text michael",
             "I want to drink lemonade"]

df = pd.DataFrame()
for sentence in sentences:
  output = GUI(sentence)
  df = df.append(output, ignore_index=True)

print(df)
