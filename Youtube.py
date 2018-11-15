import operator
from googleapiclient.discovery import build
import pandas as pd
from bidi.algorithm import get_display
import time
import matplotlib.pyplot as plt
import arabic_reshaper
from nltk.probability import *
from nltk.corpus import stopwords
from pandas.plotting import table
import matplotlib.ticker as tick
import numpy as np

DEVELOPER_KEY = "XXXXXX"
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"

youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=DEVELOPER_KEY)

# Retrieve videos from a certain channel with a certain order
def Videos(youtube, channelId, order):
    list_of_videos = youtube.search().list(
        channelId=channelId,
        type="video",
        part="id,snippet",
        maxResults=50,
        order=order
    ).execute()

    return list_of_videos.get("items", [])

Videos = Videos(youtube, "UCsP3Clx2qtH2mNZ6KolVoZQ","viewCount")

# Retrieve statistics(likes,comments,...etc) and tags of each video
def Videos_Statistics_And_Tags(videos):
    video_List = {}
    for video in videos:
         if video["id"]["kind"] == "youtube#video":
             video_List[video["id"]["videoId"]] = video["snippet"]["title"]

    s = ','.join(video_List.keys())
    Obtained_Videos = youtube.videos().list(id=s,part='id,statistics,snippet').execute()
    result = []
    for i in Obtained_Videos['items']:
         temp_result = dict(VideoTitle=get_display(arabic_reshaper.reshape(video_List[i['id']])))
         temp_result.update(i['statistics'])
         temp_result.update(i['snippet'])
         result.append(temp_result)

    df = pd.DataFrame.from_dict(result)
    df['viewCount'] = df['viewCount'].map(lambda x: float(x))
    df['likeCount'] = df['likeCount'].map(lambda x: float(x))
    df['Like to View Ratio'] = df['likeCount'].map(lambda x: float(x))/df['viewCount'].map(lambda x: float(x))
    df['dislikeCount'] = df['dislikeCount'].map(lambda x: float(x))
    df['favoriteCount'] = df['favoriteCount'].map(lambda x: float(x))
    df['commentCount'] = df['commentCount'].map(lambda x : float(x))
    df = df.sort_values('viewCount', ascending=0).head(50)
    df = df.reset_index(drop=True)
    for i in range(11):
        for j in range(len(df['tags'][i])):
            df['tags'][i][j] = get_display(arabic_reshaper.reshape(df['tags'][i][j]))
    for i in range(11):
        df['tags'][i] = df['tags'][i][:6]
    return df

# Retrieve 50 comments from each video
def Comments(youtube, videos):
    temp_Comments = []
    for video in videos:
        time.sleep(1.0)
        results = youtube.commentThreads().list(
            part="snippet",
            videoId=video["id"]["videoId"],
            textFormat="plainText",
            maxResults=50,
            order='relevance'
        ).execute()
        for item in results["items"]:
            comment = item["snippet"]["topLevelComment"]
            tempComment = dict(videoId=video["id"]["videoId"], videoName=video["snippet"]["title"],
                               nbrReplies=item["snippet"]["totalReplyCount"],
                               author=comment["snippet"]["authorDisplayName"], likes=comment["snippet"]["likeCount"],
                               publishedAt=comment["snippet"]["publishedAt"],
                               text=comment["snippet"]["textDisplay"].encode('utf-8').strip())
            temp_Comments.append(tempComment)
    data1 = pd.DataFrame.from_dict(temp_Comments)
    return data1

# Perform frequent words mining on the comments using NLTK(Natural Language ToolKit) library
def Frequent_mining():
    comments_df = Comments(youtube, Videos)
    stop_english = stopwords.words('arabic')
    tokens = []
    for txt in comments_df.text:
        tokenized = [t.decode('utf-8').strip(":,.!?") for t in txt.split()]
        tokens.extend(tokenized)
    hashtags = [w for w in tokens if w.startswith('#')]
    ghashtags = [w for w in tokens if w.startswith('+')]
    mentions = [w for w in tokens if w.startswith('@')]
    links = [w for w in tokens if w.startswith('http') or w.startswith('www')]
    filtered_tokens = [w for w in tokens if not w in stop_english and w.isalpha() and not len(
        w) < 3 and not w in hashtags and not w in ghashtags and not w in links and not w in mentions]
    fd = FreqDist(filtered_tokens)

    sortedTuples = sorted(fd.items(), key=operator.itemgetter(1), reverse=True)
    df = pd.DataFrame(sortedTuples)
    df.columns = ['Word', 'Freq']
    for i in range(len(df['Word'])):
        df['Word'][i]= get_display(arabic_reshaper.reshape(df['Word'][i]))
    df.head(10).plot(kind='bar', x=0, y=1,legend=False)
    plt.gcf().subplots_adjust(bottom=0.25)
    ax = plt.gca()
    ax.grid(which='major', axis='y', linestyle='--')
    plt.xlabel('Words', fontsize=18)
    plt.ylabel('Frequency', fontsize=18)
    plt.suptitle('Top 10 Most Frequent Words', fontsize=20, color='red')


# This only works on channels that don't have disabled comments on their videos
Frequent_mining()


# Plot Video tags Table
df2= Videos_Statistics_And_Tags(Videos)
ax2 = plt.gca()
plt.axis('off')
tbl = table(ax2, df2.head(10)[['VideoTitle','tags']], loc='center',cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)
plt.suptitle('Top 10 Videos Tags', fontsize=20,color='red')
plt.tight_layout()

Videos_info = Videos_Statistics_And_Tags(Videos)


# Format the unit of numbers for proper viewing
def y_fmt(tick_val,pos):
    if tick_val > 1000000:
        val = int(tick_val)/1000000
        return '{:.0f} M'.format(val)
    elif tick_val > 1000:
        val = int(tick_val) / 1000
        return '{:.0f} k'.format(val)
    else:
        return tick_val

# Plot Most Viewed Videos bar chart
Videos_info.sort_values('viewCount',ascending=0).head(10).plot(kind='bar', x='VideoTitle',y='viewCount',legend=False)
plt.gcf().subplots_adjust(bottom=0.55)
plt.suptitle('Top 10 Most Viewed Videos', fontsize=20,color='red')
ax = plt.gca()
ax.grid(which='major', axis='y', linestyle='--')
ax.yaxis.set_major_formatter(tick.FuncFormatter(y_fmt))

# Plot Most Liked Videos bar chart
Videos_info.sort_values('likeCount',ascending=0).head(10).plot(kind='bar', x='VideoTitle',y='likeCount',legend=False)
plt.gcf().subplots_adjust(bottom=0.55)
plt.suptitle('Top 10 Most Liked Videos', fontsize=20,color='red')
ax = plt.gca()
ax.grid(which='major', axis='y', linestyle='--')
ax.yaxis.set_major_formatter(tick.FuncFormatter(y_fmt))


# Plot Best Like-To-View-Ratio Videos  horizontal bar chart
Videos_info.sort_values('Like to View Ratio',ascending=0).head(10).plot(kind='barh', x='VideoTitle',y='Like to View Ratio',legend=False)
plt.gcf().subplots_adjust(left=0.49)
ax = plt.gca()
ax.grid(which='major', axis='x', linestyle='--')
plt.suptitle('Top 10 Best Like-To-View-Ratio Videos', fontsize=20,color='red')
plt.xticks([], [])
plt.show()
