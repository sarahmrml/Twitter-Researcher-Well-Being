import preprocessor as p
import re
import pandas as pd
from nltk.corpus import stopwords
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
import json

timelines_sentiments = pd.read_json("/home/mueller/MA_researcher_wellbeing/data/timelines_sentiments_nomiss.json", lines=True)

# only keep the text column
data = timelines_sentiments.filter(["text"], axis=1)

#work with a sample of 10000 tweets
#data = data_tweet.iloc[0:10000]

# data cleaning
p.set_options(p.OPT.URL, p.OPT.EMOJI,p.OPT.MENTION,p.OPT.SMILEY,p.OPT.NUMBER )

#forming a separate feature for cleaned tweets
for i,v in enumerate(data['text']):
    data.loc[i,'text_clean'] = p.clean(v)

# remove the hashtag sign and keep the word after
#REMOVE ONLY THE '#'NOT THE WORD AFTER
def remove_hashtag_sign(text):
    text = re.sub(r'#', '', text)
    return text

data['text_clean'] = data['text_clean'].apply(lambda x:remove_hashtag_sign(x))

#Remove extra white spaces, punctuation and apply lower casing
data['text_clean'] = data['text_clean'].str.lower().str.replace('[^\w\s]',' ').str.replace('\s\s+', ' ')

#prepare tweet list
tweet_list = data.text_clean.tolist()

#remove english stop words
stopwords = stopwords.words('english') 

#Use CountVectorizer to remove stopwords
vectorizer_model = CountVectorizer(stop_words= stopwords)

# define seed topics
seed_topic_list = [["positive emotions", "happiness", "joy", "positivity", "optimism", "satisfaction", "fun", "amazement"],
                   ["engagement", "involvement", "enthusiasm", "focus", "commitment", "flow", "submission"],
                   ["relationship", "networking", "collaboration", "social bonds", "connection", "interaction", "friend", "colleague"],
                   ["meaning", "purpose", "significance", "fulfillment", "values", "beliefs"],
                   ["accomplishment", "achievement", "success", "goals", "progress", "attainment", "publication", "award"],
                   ["physical health", "exercise", "nutrition", "wellness", "fitness", "health care", "workout"],
                   ["mindset", "attitude", "resilience", "growth mindset", "positivity", "self-belief", "optimism", "pessimism"],
                   ["work environment", "workplace culture", "team dynamics", "job satisfaction", "office atmosphere", "work-life balance", "lab", "institute"],
                   ["economic security", "financial stability", "income", "savings", "financial well-being", "job security", "funding", "grant", "money"]]


# create our BERTopic model using BERTweet as a sentence embedding
topic_model = BERTopic(vectorizer_model=vectorizer_model, seed_topic_list=seed_topic_list)

# generate topics and probabilites
topics, probs = topic_model.fit_transform(tweet_list)

# get an topic overview
overview = topic_model.get_topic_info()

# use the find_topics() function
similar_topics1, similarity1 = topic_model.find_topics("positive emotions", top_n=5)
similar_topics2, similarity2 = topic_model.find_topics("happiness", top_n=5)
similar_topics3, similarity3 = topic_model.find_topics("joy", top_n=5)
similar_topics4, similarity4 = topic_model.find_topics("positivity", top_n=5)
similar_topics5, similarity5 = topic_model.find_topics("optimism", top_n=5)
similar_topics6, similarity6 = topic_model.find_topics("satisfaction", top_n=5)
similar_topics7, similarity7 = topic_model.find_topics("fun", top_n=5)
similar_topics8, similarity8 = topic_model.find_topics("amazement", top_n=5)
similar_topics9, similarity9 = topic_model.find_topics("engagement", top_n=5)
similar_topics10, similarity10 = topic_model.find_topics("involvement", top_n=5)
similar_topics11, similarity11 = topic_model.find_topics("enthusiasm", top_n=5)
similar_topics12, similarity12 = topic_model.find_topics("focus", top_n=5)
similar_topics13, similarity13 = topic_model.find_topics("commitment", top_n=5)
similar_topics14, similarity14 = topic_model.find_topics("flow", top_n=5)
similar_topics15, similarity15 = topic_model.find_topics("submission", top_n=5)
similar_topics16, similarity16 = topic_model.find_topics("relationship", top_n=5)
similar_topics17, similarity17 = topic_model.find_topics("networking", top_n=5)
similar_topics18, similarity18 = topic_model.find_topics("collaboration", top_n=5)
similar_topics19, similarity19 = topic_model.find_topics("social bonds", top_n=5)
similar_topics20, similarity20 = topic_model.find_topics("connection", top_n=5)
similar_topics21, similarity21 = topic_model.find_topics("interaction", top_n=5)
similar_topics22, similarity22 = topic_model.find_topics("friend", top_n=5)
similar_topics23, similarity23 = topic_model.find_topics("colleague", top_n=5)
similar_topics24, similarity24 = topic_model.find_topics("meaning", top_n=5)
similar_topics25, similarity25 = topic_model.find_topics("purpose", top_n=5)
similar_topics26, similarity26 = topic_model.find_topics("significance", top_n=5)
similar_topics27, similarity27 = topic_model.find_topics("fulfillment", top_n=5)
similar_topics28, similarity28 = topic_model.find_topics("values", top_n=5)
similar_topics29, similarity29 = topic_model.find_topics("beliefs", top_n=5)
similar_topics30, similarity30 = topic_model.find_topics("accomplishment", top_n=5)
similar_topics31, similarity31 = topic_model.find_topics("achievement", top_n=5)
similar_topics32, similarity32 = topic_model.find_topics("success", top_n=5)
similar_topics33, similarity33 = topic_model.find_topics("goals", top_n=5)
similar_topics34, similarity34 = topic_model.find_topics("progress", top_n=5)
similar_topics35, similarity35 = topic_model.find_topics("attainment", top_n=5)
similar_topics36, similarity36 = topic_model.find_topics("publication", top_n=5)
similar_topics37, similarity37 = topic_model.find_topics("award", top_n=5)
similar_topics38, similarity38 = topic_model.find_topics("physical health", top_n=5)
similar_topics39, similarity39 = topic_model.find_topics("exercise", top_n=5)
similar_topics40, similarity40 = topic_model.find_topics("nutrition", top_n=5)
similar_topics41, similarity41 = topic_model.find_topics("wellness", top_n=5)
similar_topics42, similarity42 = topic_model.find_topics("fitness", top_n=5)
similar_topics43, similarity43 = topic_model.find_topics("health care", top_n=5)
similar_topics44, similarity44 = topic_model.find_topics("workout", top_n=5)
similar_topics45, similarity45 = topic_model.find_topics("mindset", top_n=5)
similar_topics46, similarity46 = topic_model.find_topics("attitude", top_n=5)
similar_topics47, similarity47 = topic_model.find_topics("resilience", top_n=5)
similar_topics48, similarity48 = topic_model.find_topics("growth mindset", top_n=5)
similar_topics49, similarity49 = topic_model.find_topics("positivity", top_n=5)
similar_topics50, similarity50 = topic_model.find_topics("self-belief", top_n=5)
similar_topics51, similarity51 = topic_model.find_topics("optimism", top_n=5)
similar_topics52, similarity52 = topic_model.find_topics("pessimism", top_n=5)
similar_topics53, similarity53 = topic_model.find_topics("work environment", top_n=5)
similar_topics54, similarity54 = topic_model.find_topics("workplace culture", top_n=5)
similar_topics55, similarity55 = topic_model.find_topics("team dynamics", top_n=5)
similar_topics56, similarity56 = topic_model.find_topics("job satisfaction", top_n=5)
similar_topics57, similarity57 = topic_model.find_topics("office atmosphere", top_n=5)
similar_topics58, similarity58 = topic_model.find_topics("work-life balance", top_n=5)
similar_topics59, similarity59 = topic_model.find_topics("lab", top_n=5)
similar_topics60, similarity60 = topic_model.find_topics("institute", top_n=5)
similar_topics61, similarity61 = topic_model.find_topics("economic security", top_n=5)
similar_topics62, similarity62 = topic_model.find_topics("financial stability", top_n=5)
similar_topics63, similarity63 = topic_model.find_topics("income", top_n=5)
similar_topics64, similarity64 = topic_model.find_topics("savings", top_n=5)
similar_topics65, similarity65 = topic_model.find_topics("financial well-being", top_n=5)
similar_topics66, similarity66 = topic_model.find_topics("job security", top_n=5)
similar_topics67, similarity67 = topic_model.find_topics("funding", top_n=5)
similar_topics68, similarity68 = topic_model.find_topics("grant", top_n=5)
similar_topics69, similarity69 = topic_model.find_topics("money", top_n=5)

similar_topics = similar_topics1 + similar_topics2 + similar_topics3 + similar_topics4 + similar_topics5 + similar_topics6 + similar_topics7 + similar_topics8 + similar_topics9 + similar_topics10 + similar_topics11 + similar_topics12 + similar_topics13 + similar_topics14 + similar_topics15 + similar_topics16 + similar_topics17 + similar_topics18 + similar_topics19 + similar_topics20 + similar_topics21 + similar_topics22 + similar_topics23 + similar_topics24 + similar_topics25 + similar_topics26 + similar_topics27 + similar_topics28 + similar_topics29 + similar_topics30 + similar_topics31 + similar_topics32 + similar_topics33 + similar_topics34 + similar_topics35 + similar_topics36 + similar_topics37 + similar_topics38 + similar_topics39 + similar_topics40 + similar_topics41 + similar_topics42 + similar_topics43 + similar_topics44 + similar_topics45 + similar_topics46 + similar_topics47 + similar_topics48 + similar_topics49 + similar_topics50 + similar_topics51 + similar_topics52 + similar_topics53 + similar_topics54 + similar_topics55 + similar_topics56 + similar_topics57 + similar_topics58 + similar_topics59 + similar_topics60 + similar_topics61 + similar_topics62 + similar_topics63 + similar_topics64 + similar_topics65 + similar_topics66 + similar_topics67 + similar_topics68 + similar_topics69
similarities = similarity1 + similarity2 + similarity3 + similarity4 + similarity5 + similarity6 + similarity7 + similarity8 + similarity9 + similarity10 + similarity11 + similarity12 + similarity13 + similarity14 + similarity15 + similarity16 + similarity17 + similarity18 + similarity19 + similarity20 + similarity21 + similarity22 + similarity23 + similarity24 + similarity25 + similarity26 + similarity27 + similarity28 + similarity29 + similarity30 + similarity31 + similarity32 + similarity33 + similarity34 + similarity35 + similarity36 + similarity37 + similarity38 + similarity39 + similarity40 + similarity41 + similarity42 + similarity43 + similarity44 + similarity45 + similarity46 + similarity47 + similarity48 + similarity49 + similarity50 + similarity51 + similarity52 + similarity53 + similarity54 + similarity55 + similarity56 + similarity57 + similarity58 + similarity59 + similarity60 + similarity61 + similarity62 + similarity63 + similarity64 + similarity65 + similarity66 + similarity67 + similarity68 + similarity69

searchterms = ["positive emotions"]*5 + ["happiness"]*5 + ["joy"]*5 + ["positivity"]*5+ ["optimism"]*5 + ["satisfaction"]*5 + ["fun"]*5 + ["amazement"]*5 + ["engagement"]*5 + ["involvement"]*5 + ["enthusiasm"]*5 + ["focus"]*5 + ["commitment"]*5 + ["flow"]*5 + ["submission"]*5 + ["relationship"]*5 + ["networking"]*5 + ["collaboration"]*5 + ["social bonds"]*5 + ["connection"]*5 + ["interaction"]*5 + ["friend"]*5 + ["colleague"]*5 + ["meaning"]*5 + ["purpose"]*5 + ["significance"]*5 + ["fulfillment"]*5 + ["values"]*5 + ["beliefs"]*5 + ["accomplishment"]*5 + ["achievement"]*5 + ["success"]*5 + ["goals"]*5 + ["progress"]*5 + ["attainment"]*5 + ["publication"]*5 + ["award"]*5 + ["physical health"]*5 + ["exercise"]*5 + ["nutrition"]*5 + ["wellness"]*5 + ["fitness"]*5 + ["health care"]*5 + ["workout"]*5 + ["mindset"]*5 + ["attitude"]*5 + ["resilience"]*5 + ["growth mindset"]*5 + ["positivity"]*5 + ["self-belief"]*5 + ["optimism"]*5 + ["pessimism"]*5 + ["work environment"]*5 + ["workplace culture"]*5 + ["team dynamics"]*5 + ["job satisfaction"]*5 + ["office atmosphere"]*5 + ["work-life balance"]*5 + ["lab"]*5 + ["institute"]*5 + ["economic security"]*5 + ["financial stability"]*5 + ["income"]*5 + ["savings"]*5 + ["financial well-being"]*5 + ["job security"]*5 + ["funding"]*5 + ["grant"]*5 + ["money"]*5

# combine the lists to one df
findtopics_output = pd.DataFrame({'search_term': searchterms,'similar_topic': similar_topics,'similarity': similarities})

# save the dfs
timelines_sentiments["topic"] = topics
timelines_sentiments["prob"] = probs

timelines_sentiments.to_json("/home/mueller/MA_researcher_wellbeing/data/timelines_topics.json", orient = "records", lines = True)
overview.to_json("/home/mueller/MA_researcher_wellbeing/data/topic_overview.json", orient = "records", lines = True)
findtopics_output.to_json("/home/mueller/MA_researcher_wellbeing/data/findtopics_output.json", orient = "records", lines = True)