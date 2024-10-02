import os
import praw as praw
import pandas as pd
import csv
import datetime
import matplotlib.pyplot as plt
import nltk
nltk.download('vader_lexicon')
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# reddit = praw.Reddit(
#     client_id=os.getenv("mG0s_3OPp0QR85VVULmUDw"),
#     client_secret=os.getenv("TeNFxx-MeKinFbmTkfuh6sO_NpE5JA"),
#     password=os.getenv("Harshnema1234"),
#     user_agent=os.getenv("sentiment"),
#     username=os.getenv("nema_harsh")
# )

reddit = praw.Reddit(
    client_id="mG0s_3OPp0QR85VVULmUDw",
    client_secret="TeNFxx-MeKinFbmTkfuh6sO_NpE5JA",
    password="Harshnema1234",
    user_agent="sentiment",
    username="nema_harsh" 
)


a=str(input("enter the keyword to search: "))
subreddit=reddit.subreddit(a)
analyzer = SentimentIntensityAnalyzer()

# Fetch top posts from the subreddit
top_posts = subreddit.top(limit=10)

results = []
for post in top_posts:
    title = post.title
    sentiment = analyzer.polarity_scores(title)
    results.append({'title': title, 'sentiment': sentiment})
# In[201]:


from datetime import datetime
new_data_list=[]
for post in subreddit.hot(limit=90):
     new_data_list.append({
         "title":post.title,
         "author":post.author,
         "link":post.shortlink,
         "comment_ID":post.id,
         "time":datetime.fromtimestamp(post.created_utc)
     })
      

# In[202]:


field_names=["title","author","link","comment_ID","time"]
with open('product_sales.csv', 'w', newline='',encoding="utf-8") as csvfile:
    writer=csv.DictWriter(csvfile,fieldnames=field_names)
    writer.writeheader()
    writer.writerows(new_data_list)
df=pd.read_csv(r'product_sales.csv')


# In[203]:


# In[ ]:





# In[204]:


def check_word(word,data):
    contains=data['title'].str.contains(word,case=False)
    return contains
df[a]=check_word(a,df)


# In[205]:


# In[206]:


df['time']=pd.to_datetime(df['time'])                          
df=df.set_index("time")
mean_a=df[a].resample('D').mean()


# In[207]:


import matplotlib.pyplot as plt
plt.plot( mean_a.index,mean_a ,marker='o', color='r')
plt.title('mentions over time')
plt.xlabel('Minutes')
plt.ylabel('Frequency')
plt.legend([a])
plt.xticks(rotation=45)
plt.show()
# In[208]:
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid= SentimentIntensityAnalyzer()

sentiment_score=df['title'].apply(sid.polarity_scores)
sentiment_score.iloc[0]
# In[209]:
# In[210]:
sentiment=sentiment_score.apply(lambda x: x['compound'])
print(df[sentiment >0.6]['title'].values[0])
print(df[sentiment <0.6]['title'].values[0])
# In[212]:
sentiment_a=sentiment[check_word(a,df)].resample('D').mean()
# In[213]:
plt.plot(sentiment_a.index,sentiment_a,color='blue',marker='o')
plt.xlabel('Minute')
plt.ylabel('Sentiment')
plt.title('Sentiment of keywords')
plt.legend([a])
plt.xticks(rotation=45)
plt.show()
