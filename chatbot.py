#Importing libriries
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
from keras.models import load_model
import json
import numpy as np
import random
from tkinter import *

# load the saved model file
lemmatizer = WordNetLemmatizer()
model = load_model('chatbot.h5')
intents = json.loads(open("intents.json").read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))


def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)

    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)

    # bag of words - matrix of N words, vocabulary matrix
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:

                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return (np.array(bag))


def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    error = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > error]

    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []

    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list
# function to get the response from the model

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

# function to predict the class and get the response

def chatbot_response(text):
    ints = predict_class(text, model)
    res = getResponse(ints, intents)
    return res

# GUI using tkinter
# You can use it or comment it and use start_chat() function by removing comment on it below the gui code
def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)

    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#0000cd", font=("Arial", 12, 'bold' ))
        res = chatbot_response(msg)
        ChatLog.insert(END, "Code Clause :  " + res + '\n\n\n\n')
        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)

chat = Tk()
# Form shape
chat.title("Codeclause")
chat.geometry("1350x500")
chat.resizable(width=True, height=True)
Icon = PhotoImage(file ="Icon.png")
chat.iconphoto(False,Icon)

# Referncing the containers and buttons
ChatLog = Text(chat, bd=0, bg="yellow",height=10, width=100)
SendButton = Button(chat, font=("Verdana",20,'bold'), text="Send", width=10, height=3,bd=1,bg = "#a52a2a",foreground = "white",command= send)
EntryBox = Text(chat, bd=2, bg="#add8e6",width=29, height=3, font=("Arial" ,12 , 'bold') ,foreground="black")
scrollbar = Scrollbar(chat, command=ChatLog.yview)

ChatLog.config(state=DISABLED)
ChatLog['yscrollcommand'] = scrollbar.set

# Placing all containers in the form
scrollbar.place(x=1310,y=6, height=380)
ChatLog.place(x=6,y=6,height = 380 , width = 1300)
EntryBox.place(relheight=0.100,relwidth=0.77,relx=0.01,rely=0.80)
SendButton.place(relx=0.79,rely=0.80,relheight=0.100,relwidth=0.20)

chat.mainloop()

# function start_chat() to start the chat bot which will continue till the user type 'end'
# !!!!  Note ;)
# You can use this function to use the bot in the console instead of using GUI

# def start_chat():
#     name = input("Enter Your Name: ")
#     print("Welcome " + name + " to the Code Clause Bot Service! Let me know how can I help you?\n")
#     while True:
#         inp = str(input()).lower()
#         if inp.lower()=="end":
#             break
#         if inp.lower()== '' or inp.lower()== '*':
#             print('Please re-phrase your query!')
#             print("-"*50)
#         else:
#             print(f"Code Clause: {chatbot_response(inp)}"+'\n')
#             print("-"*50)

# start the chat bot
# start_chat()