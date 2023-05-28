# Anime-recommendation

[Go to my LinkedIn](https://www.linkedin.com/in/lawrence-mondal/) 🌐

When it comes to anime, there's a prevalent misconception that it is solely a form of entertainment intended for children.However, this assumption couldn't be further from the truth
Welcome to the world of anime! In this project, we delve into the captivating realm of anime and leverage the power of cosine similarity to provide personalized suggestions. By employing a technique that measures the similarity between anime titles, ratings and the platform of streaming.

![naruto - Imgur](https://github.com/023lawrence/Anime-recommendation/assets/66831315/d5820404-1b7e-4522-b2a3-432b40459436)

## Introduction

This data set contains information on user preference data from 73,516 users on 12,294 anime. Each user is able to add anime to their completed list and give it a rating and this data set is a compilation of those ratings.

![a0eeabadf50400a7ebd09ca29efc97db](https://github.com/023lawrence/Anime-recommendation/assets/66831315/86d1f109-a41c-4a9d-92bf-8b0fbe97bac4)

## Data Id 📋

Anime Dataset
This dataset is named anime. The dataset contains a set of 12,294 records under 7 attributes:

| Column Name | Description |
| --- | --- |
| anime_id | myanimelist.net's unique id identifying an anime. |
| name | full name of anime. |
| genre | comma separated list of genres for this anime. |
| type | movie, TV, OVA, etc. |
| episodes | how many episodes in this show. (1 if movie). |
| rating	| average rating out of 10 for this anime. |
| members	|number of community members that are in this anime's "group". |

Rating Dataset
This dataset is named rating. The dataset contains a set of 7,813,737 records under 3 attributes:

| Column Name| Description | 
| --- | --- |
| user_id | non identifiable randomly generated user id. | 
| anime_id | the anime that this user has rated. |
| rating	| rating out of 10 this user has assigned (-1 if the user watched without assigning) |


### Aim of the Notebook:
Building a better anime recommendation system based only on similiar anime.

## Libraries 📘

code :- 
```
import os #paths to file
import numpy as np # linear algebra
import pandas as pd # data processing
import warnings# warning filter
import scipy as sp #pivot engineering

#ML model
from sklearn.metrics.pairwise import cosine_similarity

#default theme and settings
pd.options.display.max_columns

#warning hadle
warnings.filterwarnings("always")
warnings.filterwarnings("ignore")
```

## . Preprocessing and Data Analysis 💻
### Import data from google drive. 📂
code :- 
```
from google.colab import drive
drive.mount('/content/drive')
```
code :- 
```
anime_df = pd.read_csv('/content/drive/MyDrive/anime.csv')
anime_df.head()
```
Output :- 

![1](https://github.com/023lawrence/Anime-recommendation/assets/66831315/652a724d-cf53-4922-b948-a3fddd7e9160)

code :- 
```
rating_df = pd.read_csv('/content/drive/MyDrive/rating.csv' ,  on_bad_lines='skip') #on_bad_lines = 'skip' :- this will cause the offending lines to be skipped.
rating_df.head()
```
Output :- 

![2](https://github.com/023lawrence/Anime-recommendation/assets/66831315/35c66596-2045-45f4-9095-accf6459616e)

### Data shapes and info
code :- 
```
print("Anime:- \n")
print(anime_df.info())
print("\n" , "*"*50 , "\n Rating :- \n" )
print(rating_df.info())
```
Output :- 

![3](https://github.com/023lawrence/Anime-recommendation/assets/66831315/ee68bfd6-e92f-4b9a-a4b2-b67d6672fd3f)

### Handling Missing values 🚫
code :- 
```
print("Anime_id missing values(%) \n")
print(round(anime_df.isnull().sum().sort_values(ascending=False)/len(anime_df.index) , 4)*100)
print("rating missing values(%) \n")
print(round(rating_df.isnull().sum().sort_values(ascending=False)/len(rating_df.index) , 4)*100)
```
Output :- 

![4](https://github.com/023lawrence/Anime-recommendation/assets/66831315/90030721-a905-47b8-83a2-e86e32e784e9)

code :- 
```
print(anime_df['type'].mode())
print(anime_df['genre'].mode())
```
Output :- 

![5](https://github.com/023lawrence/Anime-recommendation/assets/66831315/424a1b58-0c58-414e-ba67-7d7fcef99bdc)

**Weirdly enough the mode value of genre is Hentai, the mode value of type is TV.**

![tenor](https://github.com/023lawrence/Anime-recommendation/assets/66831315/2681c7f9-5493-4bf1-ad07-dd47eb16f689)

## . Cosine Similarity Model
## . Conclusion




	




