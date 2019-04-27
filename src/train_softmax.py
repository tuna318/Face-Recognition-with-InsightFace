from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from softmax import SoftMax
import numpy as np
import argparse
import pickle

# Construct the argumet parser and parse the argument
ap = argparse.ArgumentParser()

ap.add_argument("--embeddings", default="outputs/embeddings.pickle",
                help="path to serialized db of facial embeddings")
ap.add_argument("--model", default="outputs/my_model.h5",
                help="path to output trained model")
ap.add_argument("--le", default="outputs/le.pickle",
                help="path to output label encoder")

args = vars(ap.parse_args())

# Load the face embeddings
data = pickle.loads(open(args["embeddings"], "rb").read())

# Encode the labels
le = LabelEncoder()
labels = le.fit_transform(data["names"])
num_classes = len(np.unique(labels))
labels = labels.reshape(-1, 1)
one_hot_encoder = OneHotEncoder(categorical_features = [0])
labels = one_hot_encoder.fit_transform(labels).toarray()

embeddings = np.array(data["embeddings"])

# Initialize Softmax training model arguments
BATCH_SIZE = 32
EPOCHS = 20
input_shape = embeddings.shape[1]

# Build sofmax classifier
softmax = SoftMax(input_shape=(input_shape,), num_classes=num_classes)
model = softmax.build()

# Create KFold
cv = KFold(n_splits = 5, random_state = 42, shuffle=True)
history = {'acc': [], 'val_acc': [], 'loss': [], 'val_loss': []}
# Train
for train_idx, valid_idx in cv.split(embeddings):
    X_train, X_val, y_train, y_val = embeddings[train_idx], embeddings[valid_idx], labels[train_idx], labels[valid_idx]
    his = model.fit(X_train, y_train,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    verbose=1,
                    validation_data=(X_val, y_val))
    print(his.history['acc'])

    history['acc'] += his.history['acc']
    history['val_acc'] += his.history['val_acc']
    history['loss'] += his.history['loss']
    history['val_loss'] += his.history['val_loss']


# write the face recognition model to output
model.save(args['model'])
f = open(args["le"], "wb")
f.write(pickle.dumps(le))
f.close()

# Plot
plt.figure(1)
# Summary history for accuracy
plt.subplot(211)
plt.plot(history['acc'])
plt.plot(history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

# Summary history for loss
plt.subplot(212)
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('outputs/accuracy_loss.png')
plt.show()
