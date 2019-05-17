import nltk
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import PyPDF2
import string

# importing required modules

# creating a pdf file object
pdfFileObj = open('JavaBasics-notes.pdf', 'rb')

# creating a pdf reader object
pdfReader = PyPDF2.PdfFileReader(pdfFileObj)

# printing number of pages in pdf file
# print(pdfReader.numPages)

# creating a page object
pageObj =[]

for i in range(0,22):
    pageObj.append(pdfReader.getPage(i))
# extracting text from page

text=[]

for i in range(0,22):
    text.append(pageObj[i].extractText())
# print(text)
def text_process(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]
    print(nopunc)
    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    print(nopunc)

    # Now just remove any stopwords
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

df=pd.DataFrame(index=[i for i in range(22)],columns=['Data'],data=pageObj)
print(df.head())
print(pageObj[0])
# closing the pdf file object
# pdfFileObj.close()
