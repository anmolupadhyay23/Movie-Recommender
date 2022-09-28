import numpy as np
import pandas as pd
import ast # it'll help to make the data in simple form to make it easier to work with
import pickle # to send the name of movie to the site

# functions
def convert(obj):
    L=[]
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L
def convertCast(obj):
    L=[]
    counter=0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter+=1
        else:
            break
    return L
def convertCrew(obj):
    L=[]
    for i in ast.literal_eval(obj):
        if i['job']=='Director':
            L.append(i['name'])
            break
    return L
def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)
def recommend(movie):
    movie_index=new_df[new_df['title']==movie].index[0]
    distances=similarity[movie_index]
    movie_list=sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    for i in movie_list:
        print(new_df.iloc[i[0]].title)

movies=pd.read_csv('tmdb_5000_movies.csv')
credits=pd.read_csv('tmdb_5000_credits.csv')

# merging both the datasets into one on the basis of title
movies=movies.merge(credits,on='title')

# removing extra columns from the dataset
# We will keep the columns: genre, id, keywords, title, overview, cast, crew

# seeing all the columns of the new dataset
# print(movies.info())

# making the dataset to keep our decided columns
movies=movies[['movie_id','title','overview','genres','keywords','cast','crew']]

# data preprocessing
# print(movies.isnull().sum()) # to check for empty columns
movies.dropna(inplace=True) # ignoring the 3 empty columns beacuse 3 is very small in comparision to 5000
# print(movies.isnull().sum())
# print(movies.duplicated().sum()) # checking for duplicated rows
# making a new column 'tag' with the help of 'overview', 'keywords', 'genres', 'cast', 'crew'
# making the data in simple format to make the 'tag'
movies['genres']=movies['genres'].apply(convert)
movies['keywords']=movies['keywords'].apply(convert)
# for the cast, we will take the first three main cast, rest we'll ignore
movies['cast']=movies['cast'].apply(convertCast)
# we'll take only director from the crew
movies['crew']=movies['crew'].apply(convertCrew)
# making the string in overview to a list so that every word is different in the list
movies['overview']=movies['overview'].apply(lambda x:x.split())
# now we've to replace spaces between words to make the model work better
movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","")for i in x])
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","")for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","")for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","")for i in x])
# making the 'tags' column by adding 'overview', 'keywords', 'genres', 'cast', 'crew'
movies['tags']=movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']
# now we don't need the extra column, so we'll make a new datframe with 'movie_id', 'title', 'tags'
new_df=movies[['movie_id','title','tags']]
# now we've to convert the list of words in 'tags' back into string
new_df['tags']=new_df['tags'].apply(lambda x:" ".join(x))
# now we'll convert the 'tags' string to lowercase
new_df['tags']=new_df['tags'].apply(lambda x:x.lower())

# text vectorization: we'll make every string a vector and recommend movies which are closest to that vector
# we'll use 'bag of words' for text vectorization
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(stop_words='english',max_features=5000)
vectors=cv.fit_transform(new_df['tags']).toarray()
# print(vectors)
# print(cv.get_feature_names())
# now we'll use stemming technique to remove extra words:
    # for example: action, actions will be converted into one word
    # for example: love, loved, loving will be converted into one word
# we'll use the library nltk
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
new_df['tags']=new_df['tags'].apply(stem)

# now we'll use cosine distance instead of euclidean because this is a 5000 Dimensional dataset
    # Euclidean distance fails in higher dimensions
from sklearn.metrics.pairwise import cosine_similarity
similarity=cosine_similarity(vectors)
print(recommend('Batman Begins'))

# website
pickle.dump(new_df.to_dict(),open('movies_dict.pkl','wb'))
pickle.dump(similarity,open('similarity.pkl','wb'))
