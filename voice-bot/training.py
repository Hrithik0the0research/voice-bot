
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Flatten,Dense,Activation,Dropout,LSTM,Input,Conv1D,MaxPool1D,concatenate,Embedding
import tensorflow as tf
#from tensorflow.python.keras.optimizers import SGD
lemmatizer=WordNetLemmatizer()
intents=json.loads(open('intents.json').read())
words=[]
classes = []
documents = []
ignore_letters = ['?', '!']
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list=nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list,intent['tag'])) 
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
words=[lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words=sorted(set(words))
classes=sorted(set(classes))
pickle.dump(words,open('words.pk1','wb'))
pickle.dump(classes,open('classes.pk1','wb'))
training=[]
output_empty=[0]*len(classes)
for document in documents:
    bag=[]
    word_patterns=document[0]
    word_patterns=[lemmatizer.lemmatize(word.lower())for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    output_Row=list(output_empty)
    output_Row[classes.index(document[1])]=1
    training.append([bag,output_Row])
random.shuffle(training)
training=np.array(training)
train_x=list(training[:,0])
train_y=list(training[:,1])
"""
model=Sequential()
model.add(Dense(128,input_shape=(len(train_x[0]),),activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]),activation='softmax'))
#sgd=SGD(learning_rate=0.01,decay=1e-6,momentum=0.9,nesterov=True)
model.compile(loss="categorical_crossentropy",optimizer="RMSprop",metrics=['accuracy'])
hist=model.fit(np.array(train_x),np.array(train_y),epochs=200,batch_size=5,verbose=1)
#model.save("voicebot_model.h5",hist)
#train_x=train_x.reshape(1,train_x.shape[0],train_x.shape[1])
#hist=model.fit(train_x,np.array(train_y),epochs=200,batch_size=5,verbose=1)
model.save("voicebot_model_dnn.h5",hist)
print('Done')
##cnn-model
"""

train_x=np.array(train_x)
train_y=np.array(train_y)
#train_x=train_x[:122]
#train_y=train_y[:122]
print(train_x.shape,train_y.shape)
model=Sequential()
#model.add(Input(shape=(len(train_x[0]), )))

model.add(Embedding(len(train_x[0]), 2000, input_length = len(train_x[0])))

model.add(Conv1D(filters=2, kernel_size=3, activation='relu',kernel_initializer="he_normal"))
model.add(Conv1D(filters=4, kernel_size=3, activation='relu',kernel_initializer="he_normal"))
#model.add(Conv1D(filters=12, kernel_size=3, activation='relu',kernel_initializer="he_normal"))

# concat1 = Concatenate()([conv1, conv2, conv3])
#concat1 = concatenate([conv1, conv2])
model.add(MaxPool1D(1))
model.add(Dropout(0.2))

model.add(Conv1D(filters=2, kernel_size=3, activation='relu',kernel_initializer="he_normal"))
model.add(Conv1D(filters=4, kernel_size=3, activation='relu',kernel_initializer="he_normal"))
#model.add(Conv1D(filters=12, kernel_size=3, activation='relu',kernel_initializer="he_normal"))

#concat2 = concatenate([conv4, conv5])
model.add(MaxPool1D(1))
model.add(Dropout(0.2))

model.add(Conv1D(filters=12, kernel_size=3, activation='relu',kernel_initializer="he_normal"))
model.add(Flatten())
model.add(Dropout(0.2))

#Dense1 = Dense(128, activation='relu', kernel_initializer="he_normal",kernel_regularizer=regularizers.l2(0.001))(dropout)
model.add(Dense(128, activation='relu', kernel_initializer="he_normal",kernel_regularizer=regularizers.l2(0.001)))
model.add(Dense(len(train_y[0]), activation='softmax'))

#model = Model(inputs=input,outputs=output)
model.summary()
model.compile(loss="categorical_crossentropy",optimizer="Adam",metrics=['accuracy'])
hist=model.fit(np.array(train_x),np.array(train_y),epochs=200,batch_size=5,verbose=1)
#model.save("voicebot_model.h5",hist)
#train_x=train_x.reshape(1,train_x.shape[0],train_x.shape[1])
#hist=model.fit(train_x,np.array(train_y),epochs=200,batch_size=5,verbose=1)
model.save("voicebot_model_cnn.h5",hist)
#print('Done')
