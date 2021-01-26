from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle

embeddingfile="output/embeddings.pickle"

labelencoderfile="output/labelencoder.pickle"
recognizerfile="output/recognizer.pickle"

file=pickle.loads(open(embeddingfile,"rb").read())

labelencoder=LabelEncoder()
labels=labelencoder.fit_transform(file["names"])

ml=SVC(C=1.0,kernel="linear",probability=True)
ml.fit(file["embeddings"],labels)

file1=open(labelencoderfile,"wb")
file1.write(pickle.dumps(labelencoder))
file1.close()

file2=open(recognizerfile,"wb")
file2.write(pickle.dumps(ml))
file2.close()
