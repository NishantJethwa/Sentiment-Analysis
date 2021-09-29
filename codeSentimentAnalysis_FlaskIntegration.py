# Importing the libraries
import pandas as pd # manipulate the dataset eg read, cut the dataset etc...
import re # helps in accesing set of strings and is used for manipulating the text 
import nltk
nltk.download('stopwords') # (the, is, an, in)big library which contaion useless words which do not contribute to the actual sentiment of the sentence eg (is of the etc..)
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer# brings the text to its root/base meaning
nltk.download('wordnet') # it is a lexical database for the English language(nouns, verbs, adjectives and adverbs), which was created by Princeton, and is part of the NLTK corpus.
# To count the iterations
from tqdm import tqdm #library used for iterative functions/tasks or representing progess bar etc...
from flask import Flask, render_template, request, flash, url_for
# Importing the dataset
dataset = pd.read_excel('Restaurant_Reviews_Detailed.xlsx')

def removeApostrophe(review):
    phrase = re.sub(r"won't", "will not", review)
    phrase = re.sub(r"can\'t", "can not", review)
    phrase = re.sub(r"n\'t", " not", review)
    phrase = re.sub(r"\'re", " are", review)
    phrase = re.sub(r"\'s", " is", review)
    phrase = re.sub(r"\'d", " would", review)
    phrase = re.sub(r"\'ll", " will", review)
    phrase = re.sub(r"\'t", " not", review)
    phrase = re.sub(r"\'ve", " have", review)
    phrase = re.sub(r"\'m", " am", review)
    return phrase

def removeAlphaNumericWords(review):
     return re.sub("\S*\d\S*", "", review).strip() 
 
def removeSpecialChars(review):
     return re.sub('[^a-zA-Z]', ' ', review)

to_remove = ['not'] 
def doTextCleaning(review):
#    review = removeHTMLTags(review)
    review = removeApostrophe(review)
    review = removeAlphaNumericWords(review)
    review = removeSpecialChars(review) 
   
    # Lower casing
    review = review.lower() # 'the' and 'THE' will be considered as two different words though they have the same meaning
    
    #Tokenization
    review = review.split() # pre-step for removing the STOPWORDS and creating SPARSE MATRIX
    
    #Removing Stopwords and Lemmatization
    lmtzr = WordNetLemmatizer()# Converts the word to its root meaning (LOVED/LOVING ------> LOVE)
    review = [lmtzr.lemmatize(word, 'v') for word in review if not word in set(stopwords.words('english')).difference(to_remove)] 
    
    # all the words which are in review but not a part of STOPWORDS will be in review
    review = " ".join(review)    
    return review

# creating the document corpus
corpus = []   # collection of clean text.
for index, row in tqdm(dataset.iterrows()):
    review = doTextCleaning(row['Review'])
    corpus.append(review)
    

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

cv = CountVectorizer(analyzer='word', ngram_range=(2, 2), max_features = 2000)
cv2 = TfidfVectorizer(analyzer='word', ngram_range=(2, 2), max_features = 2000) # (creating an object for TFIDF VECTORIZER )Creates a sparse matrix of unique words as colums and the rows will be the reviews  
#and the individual cell value will be the frequency of that word in that sentence 

X = cv.fit_transform(corpus)
y = dataset.iloc[:,1].values
X = X.toarray()

X2 = cv2.fit_transform(corpus)
y2 = dataset.iloc[:,1].values
X2 = X2.toarray()

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size = 0.20, random_state = 0)


# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()# creating an object of Naive Bayes Classifier
classifier.fit(X_train, y_train)



# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Predict the sentiment for new review
def predictNewReview(y):
    if y =='':
        print('Invalid Review')  
        prediction = "Enter A Valid Review"
    else:
        y = doTextCleaning(y)
        new_review = cv.transform([y]).toarray()  
        prediction =  classifier.predict(new_review)
        if prediction == 'Positive':            
            print( "Positive Review" )

        else:
            print( "Negative Review" )

    return prediction         

# Save the trained model as a pickle string. 
from flask import Flask, render_template, request, flash, url_for
    
app = Flask(__name__)
review1 = ""    
review_data = " "
    
@app.route('/')
def home():
    return render_template("first_page_button.html")
    
@app.route("/page_new", methods=['GET','POST'])
def home1():
    result = request.form.to_dict();
    input_review = result['review']
    print(input_review)
    res = predictNewReview(input_review)
    if(res == 'Positive'):
        return render_template("envelope.html",value = predictNewReview(input_review))
    else:
        return render_template("envelope2.html",value = predictNewReview(input_review))
        
@app.route("/output", methods=['GET','POST'])
def home2():
    return render_template("page_new.html")
    
@app.route("/pos", methods=['GET','POST'])
def homeee():
    return render_template("postitive-env.html")

@app.route("/neg", methods=['GET','POST'])
def homeee2():
    return render_template("negative-env.html")
    
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
debug=False
