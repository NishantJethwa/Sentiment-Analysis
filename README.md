# Purpose

Now a days the competition is very neck to neck and in this condition, a company cannot afford to neglect the reviews of its users.
Reviews help the company to know their standing in the market along with their competitors. Reviews can be online and offline(face-to-face). The online reviews are in order of millions, this makes it impossible for humans to determine the actual sentiment behind all the reviews i.e., positive, negative or neutral etc.
So, the companies use Sentiment Analysis which helps them process all the information (all the reviews on their particular product) and come to a conclusion whether or not their product is like or disliked in the market.
For example, hotel/restaurants can use sentiment analysis to analyse the customers perspective (good, bad or neutral) on their food.

# **What is Sentiment Analysis**

Sentiment analysis is a machine learning technique that automatically analyzes data and detects the sentiment of text. By identifying the sentiment towards products, brands or services, businesses can understand how their customers are talking in online conversations.
Sentiment analysis models detect polarity within a text (e.g. a positive or negative opinion), whether it’s a whole document, paragraph, sentence, or clause.
Understanding people’s emotions is essential for businesses since customers are able to express their thoughts and feelings more openly than ever before. By automatically analyzing customer feedback, from survey responses to social media conversations, brands are able to listen attentively to their customers, and tailor products and services to meet their needs.

# **Types of sentiment analysis**

1.	Fine-grained Sentiment Analysis
2.	Emotion detection
3.	Aspect-based Sentiment Analysis
4.	Multilingual sentiment analysis


# **How the code Works?**

Step 1: Import the required libraries. 
Step 2: Import the dataset and the wordnet (database of semantic relationship between words).
Step 3: Performed Pre-processing where we clean the dataset text by removing HTMLtags, Apostrophe’s, AlphaNumericWords, SpecialChars and convert all the sentences to lower case and then splitting all the words and we also removed all the stop words.
Step 4: After the text is cleaned, we append the cleaned text to the corpus.
Step 5: Then we create the bag of words algorithm where we perform vectorization on the corpus using TFIDF vectorizer, which will basically find all the unique words form the corpus and form it in an array based on its frequency.
Step 6: Then we divide the dataset into Train and Test dataset.
Step 7: Then we fit the training Dataset to our Naïve Bayes Classifier. 
Step 8: Then we Predict the Output of the Test Dataset and then output the confusion matrix and we determine the accuracy of the model.
Step 9: Then we perform the same operation on a new user comment and determine its sentiment. 


# **FLASK**

Flask is a lightweight WSGI web application framework. It is designed to make getting started quick and easy, with the ability to scale up to complex applications. It began as a simple wrapper around Werkzeug and Jinja and has become one of the most popular Python web application frameworks.
Flask offers suggestions, but doesn't enforce any dependencies or project layout. It is up to the developer to choose the tools and libraries they want to use. There are many extensions provided by the community that make adding new functionality easy.
Using Flask, we were able to host our webpage on a local server and were also able to access this webpage on our ANDROID phones (As seen in the screenshots).

# **Screenshots**

![homepage](https://user-images.githubusercontent.com/42881984/135728860-7b0bb4dd-baf9-4b6d-ac28-461f532a80f3.png)
![review page 1](https://user-images.githubusercontent.com/42881984/135728816-afe5b01c-35fc-44a2-8820-3b4d479acb80.png)
![review page 2](https://user-images.githubusercontent.com/42881984/135728822-b5afe04f-6d3e-461c-b9c0-be2d19cca7ad.png)
![review page 3](https://user-images.githubusercontent.com/42881984/135728823-db1491a3-e0f3-4317-9b8b-ed54d6d14543.png)
![4](https://user-images.githubusercontent.com/42881984/135728824-81f232fd-4e7f-48cd-935f-c08d046f2706.png)
![Screenshot_20200306-185625](https://user-images.githubusercontent.com/42881984/135728830-2c9c610f-b530-4aa5-8604-e022ba1a74eb.jpg)
![Screenshot_20200306-185553](https://user-images.githubusercontent.com/42881984/135728831-02f220c8-098c-4b91-9c10-6b7d94792b6d.jpg)
![Screenshot_20200306-185611](https://user-images.githubusercontent.com/42881984/135728832-c33805d4-d0f9-4bbe-ad1c-9ef54a367b89.jpg)
![Screenshot_20200306-185621](https://user-images.githubusercontent.com/42881984/135728833-bbb08635-8ba6-49b2-8e16-24dbe5b259a0.jpg)
