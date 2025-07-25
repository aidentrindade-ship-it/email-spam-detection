#!/usr/bin/env python
# coding: utf-8

# In[23]:


# Import necessary libraries
import pandas as pd                     # For data handling
import matplotlib.pyplot as plt         # For plotting graphs
import seaborn as sns                   # For advanced plots
from sklearn.datasets import load_iris  # To load Iris dataset
from sklearn.preprocessing import StandardScaler  # To scale features
from sklearn.decomposition import PCA   # For dimensionality reduction
from sklearn.cluster import KMeans      # For KMeans clustering
import scipy.cluster.hierarchy as sch  # For dendrogram
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
import chardet


# In[24]:


with open('spam.csv', 'rb') as f:
    result = chardet.detect(f.read(10000))  # Read first 10 KB
    encoding = result['encoding']
    print(f"Detected encoding: {encoding}")


# In[25]:


# Load CSV file into a DataFrame
df = pd.read_csv('spam.csv', skiprows=1, header=None, encoding='Windows-1252')


# In[26]:


# If more than 2 columns, merge column 2 onwards into a single column
if df.shape[1] > 2:
    df[1] = df.iloc[:, 1:].astype(str).agg(' '.join, axis=1)
    df = df[[0, 1]]  # Keep only the first two columns
# Rename for clarity (optional)
df.columns = ['category', 'email']

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
df.describe()


# In[27]:


value_counts = df['category'].value_counts()
print(value_counts)


# In[28]:


dup = df.duplicated().sum()
print(f'number of duplicated rows are {dup}')


# In[29]:


# Missing Values/Null Values Count
df.isnull().sum()


# In[30]:


# Chart - 1 Pie Chart Visualization Code For Distribution of Spam vs Ham Messages
spread = df['category'].value_counts()
plt.rcParams['figure.figsize'] = (5,5)

# Set Labels
spread.plot(kind = 'pie', autopct='%1.2f%%', cmap='Set1')
plt.title(f'Distribution of Spam vs Ham')

# Display the Chart
plt.show()



# In[46]:


# Encode target labels
df['label'] = df['category'].map({'ham': 0, 'spam': 1})

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['email'], df['label'], test_size=0.25, random_state=42)

# Vectorize text
vectorizer = CountVectorizer(stop_words=None, encoding='Windows-1252')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
#print(y_train.describe())
#print(y_test.describe())


# In[47]:


# 1. Multinomial Naive Bayes
nb = MultinomialNB()
nb.fit(X_train_vec, y_train)
y_pred_nb = nb.predict(X_test_vec)
print("Multinomial Naive Bayes Results:")
print(classification_report(y_test, y_pred_nb))
print("Accuracy:", accuracy_score(y_test, y_pred_nb))


# In[33]:


result = pd.DataFrame({
    'email': X_test.values,
    'Prediction': y_pred_nb,
    'Actual': y_test.values
})
result


# In[34]:


# 2. Neural Network (MLPClassifier)
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
mlp.fit(X_train_vec, y_train)
y_pred_mlp = mlp.predict(X_test_vec)
print("\nNeural Network (MLPClassifier) Results:")
print(classification_report(y_test, y_pred_mlp))
print("Accuracy:", accuracy_score(y_test, y_pred_mlp))


# In[35]:


result = pd.DataFrame({
    'email': X_test.values,
    'Prediction': y_pred_mlp,
    'Actual': y_test.values
})
result


# In[36]:


# Defining a function for the Email Spam Detection System
def detect_spam_mnb(email_text):
    email_vec = vectorizer.transform([email_text])  # Vectorize the input email
    prediction = nb.predict(email_vec)             # Predict using the vectorized input

    if prediction[0] == 0:
        return "This is a Ham Email!"
    else:
        return "This is a Spam Email!"


# In[37]:


# Defining a function for the Email Spam Detection System
def detect_spam_mlp(email_text):
    email_vec = vectorizer.transform([email_text])  # Vectorize the input email
    prediction = mlp.predict(email_vec)             # Predict using the vectorized input

    if prediction[0] == 0:
        return "This is a Ham Email!"
    else:
        return "This is a Spam Email!"


# In[60]:


# Example of how to use the function
#sample_email = 'click link below to register'
#sample_email = 'congratulations! you have won a lottery.'
#sample_email = 'congratulations! you have been promoted'
#sample_email = 'good news! you have been selected'
#sample_email = 'you have been selected for a training on AI ML at ESCI Hyderabad'
sample_email = 'send you account details to get an amazing offer'
result_mnb = detect_spam_mnb(sample_email)
result_mlp = detect_spam_mlp(sample_email)
print(f"MNB result : {result_mnb}")
print(f"NN result  : {result_mlp}")

