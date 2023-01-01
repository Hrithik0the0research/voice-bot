from keras.models import load_model
import numpy as np
from keras.layers import Conv2D,Dense,MaxPool2D,Flatten,MaxPooling2D,Dropout
from keras.models import Model
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import keras.backend as k1
from keras.layers.merging import concatenate
from keras.utils import plot_model
k1.clear_session()
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
train_x=np.array(train_x)
train_y=np.array(train_y)
#train_x=train_x[:104]
#train_y=train_y[:104]
mddnn=load_model("voicebot_model_dnn.h5")
mdcnn=load_model("voicebot_model_cnn.h5")
all_model=[]
all_model.append(mddnn)
all_model.append(mdcnn)
def define_stacked_model(members,train_y):
	
	# update all layers in all models to not be trainable
	for i in range(len(members)):
		model = members[i]
		for layer in model.layers:
			# make not trainable
			layer.trainable = False
			# rename to avoid 'unique layer name' issue
			layer._name = 'ensemble_' + str(i+1) + '_' + layer.name
	# define multi-headed input
	ensemble_visible = [model.input for model in members]
	# concatenate merge output from each model
	ensemble_outputs = [model.output for model in members]
	merge = concatenate(ensemble_outputs)
	hidden = Dense(10, activation='relu')(merge)
	output = Dense(len(train_y), activation='softmax')(hidden)
	print("train_y ",len(train_y))
	model = Model(inputs=ensemble_visible, outputs=output)
	# plot graph of ensemble
	#plot_model(model, show_shapes=True, to_file='model_graph.png')
	# compile
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.summary()
	return model
def fit_stacked_model(model, inputX, inputy):
	# prepare input data
	inputX=np.array(inputX)
	print(inputX)
	X = [inputX for _ in range(len(model.input))]
	print(X)
	# encode output data
	#inputy_enc = to_categorical(inputy)
	# fit model
	x1=np.array(X)
	#print(inputy_enc.shape,x1.shape)
	
	# encode output data
	#inputy_enc = to_categorical(inputy)
	# fit model
	#x1=np.array(X)
	#print(inputy_enc.shape,x1.shape)
	hist=model.fit(X,np.array(inputy), batch_size=5,epochs=200, verbose=1)
	model.save("voicebot_model_ensemble.h5",hist)

# make a prediction with a stacked model
def predict_stacked_model(model, inputX):
	# prepare input data
	X = [inputX for _ in range(len(model.input))]
	# make prediction
	return model.predict(X, verbose=0)


k1.clear_session()
stacked_model = define_stacked_model(all_model,train_y)
# fit stacked model on test dataset
fit_stacked_model(stacked_model,train_x, train_y)
# make predictions and evaluate
