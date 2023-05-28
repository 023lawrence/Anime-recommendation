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

code :- 
```
# deleting anime with 0 rating
anime_df=anime_df[~np.isnan(anime_df["rating"])]

# filling mode value for genre and type
anime_df['genre'] = anime_df['genre'].fillna(
anime_df['genre'].dropna().mode().values[0])

anime_df['type'] = anime_df['type'].fillna(
anime_df['type'].dropna().mode().values[0])

#checking if all null values are filled
anime_df.isnull().sum()
```
Output :- 

![6](https://github.com/023lawrence/Anime-recommendation/assets/66831315/e552e849-c995-41a8-8bff-44b115315924)

**This will help to delete anime with 0 rating and fill the mode value for in genre in case of missing **

### Filling Nan values
In general the value -1 suggests the user did not register a raiting so we will foll with Nan values.

code :- 
```
rating_df['rating'] = rating_df['rating'].apply(lambda x: np.nan if x==-1 else x)
rating_df.head(20)
```
Output :- 

![7](https://github.com/023lawrence/Anime-recommendation/assets/66831315/c6dc7622-b890-47cb-9993-1e910e9dae3b)

### Pivot Table for similarity
We will create a pivot table of users as rows and tv show names as columns. The pivot table will help us will be analized for the calcuations of similarity.

code :- 
```
pivot = rated_anime.pivot_table(index=['user_id'], columns=['name'], values='rating')
pivot.head()
```
Output :-

![8](https://github.com/023lawrence/Anime-recommendation/assets/66831315/6ae9bf24-4801-4f1f-bd51-b912f675ccbf)

### Now we will change our pivot table in the following steps:
1. Value normalization.
2. Filling Nan values as 0.
3. Transposing the pivot for the next step.
4. Dropping columns with the values of 0 (unrated).
5. Using scipy package to convert to sparse matrix format for the similarity computation.

code :- 
```
# step 1
pivot_n = pivot.apply(lambda x: (x-np.mean(x))/(np.max(x)-np.min(x)), axis=1)

# step 2
pivot_n.fillna(0, inplace=True)

# step 3
pivot_n = pivot_n.T

# step 4
pivot_n = pivot_n.loc[:, (pivot_n != 0).any(axis=0)]

# step 5
piv_sparse = sp.sparse.csr_matrix(pivot_n.values)
```

## . Cosine Similarity Model

![9](https://github.com/023lawrence/Anime-recommendation/assets/66831315/413503b2-4212-4ab7-ba3c-91a42d2b762b)
![10](https://github.com/023lawrence/Anime-recommendation/assets/66831315/54104b55-7822-4e76-a560-849253a3fd49)

**Cosine similarity measures the similarity between two vectors of an inner product space. It is measured by the cosine of the angle between two vectors and determines whether two vectors are pointing in roughly the same direction.**

code :- 

```
#model based on anime similarity
anime_similarity = cosine_similarity(piv_sparse)

#Df of anime similarities
ani_sim_df = pd.DataFrame(anime_similarity, index = pivot_n.index, columns = pivot_n.index)
```
code :- 
```
def anime_recommendation(ani_name):
    """
    This function will return the top 5 shows with the highest cosine similarity value and show match percent
    
    example:
    >>>Input: 
    
    anime_recommendation('Death Note')
    
    >>>Output: 
    
    Recommended because you watched Death Note:

                    #1: Code Geass: Hangyaku no Lelouch, 57.35% match
                    #2: Code Geass: Hangyaku no Lelouch R2, 54.81% match
                    #3: Fullmetal Alchemist, 51.07% match
                    #4: Shingeki no Kyojin, 48.68% match
                    #5: Fullmetal Alchemist: Brotherhood, 45.99% match 

               
    """
    
    number = 1
    print('Recommended because you watched {}:\n'.format(ani_name))
    for anime in ani_sim_df.sort_values(by = ani_name, ascending = False).index[1:6]:
        print(f'#{number}: {anime}, {round(ani_sim_df[anime][ani_name]*100,2)}% match')
        number +=1  
```	
Code :- 
```
anime_recommendation('Dragon Ball Z')
```
Output :- 

![11](https://github.com/023lawrence/Anime-recommendation/assets/66831315/fe12403b-17ae-4ebb-8bc4-06d554d814ee)

## . Conclusion ✔

In this notebook, a recommendation algorithm based on cosine similarity was created. For further analysis i sugggest prediction based on genres, or a user-user approach or use K- Means Clustering.

All the files, datasets, workbooks, and icons displayed above have been uploaded. Fell free to use the resources from this project for your future projects. Give this project and  a star if you enjoy it, or just let me know. :)


[Go to my LinkedIn](https://www.linkedin.com/in/lawrence-mondal/) 🌐

![giphy](https://github.com/023lawrence/Anime-recommendation/assets/66831315/d5c4f371-1621-494a-b7f6-f9e2541def7c)


	




