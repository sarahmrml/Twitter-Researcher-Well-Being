#################################################
##### 1) preprocessing of twitter timelines #####
#################################################

# input: timelines.json
# output: timelines_filtered.json


###### 1. prepare session #####
# activate virtual environment "masterenv"
import os
activate_this = os.path.join("/home/mueller/MA_researcher_wellbeing/masterenv", "bin", "activate_this.py") 
exec(open(activate_this).read(), {'__file__': activate_this})

# import necessary packages
import json
import tqdm
import pandas as pd
import gender_guesser.detector as gender

# load dataset in chunks to avoid memory overload
# timelines = twitter timelines (inkl. metadata like user_id or language) of 15,987 psychological researchers on twitter
timelines = pd.DataFrame()
for chunk in tqdm.tqdm(pd.read_json("/home/mueller/MA_researcher_wellbeing/data/timelines.json", lines=True, chunksize=200000)):
    timelines = pd.concat([timelines, chunk])


##### 2. filtering #####
# 1 exclude tweets before 03/11/2019
timelines["created_at"] = pd.to_datetime(timelines["created_at"])
timelines_filtered = timelines[~(timelines["created_at"] < '2019-03-11')]
print(f"{len(timelines_filtered)} rows in the filtered (1) df (only tweets since 2019/11/03)")

# 2 keep only english tweets
timelines_filtered = timelines_filtered[timelines_filtered["lang"] == "en"]
print(f"{len(timelines_filtered)} rows in the filtered (2) df (only english tweets)")

# 3 exclude retweets
timelines_filtered = timelines_filtered[timelines_filtered["is_retweet"] == False]
print(f"{len(timelines_filtered)} rows in the filtered (3) df (without retweets)")

# 4 exclude informational tweets
keywords_infotweets = ["our new", "et al.", "now available", "new paper", "preprint", "hiring", "now out", "we show", "apply(?:ing)", "phd position", "postdoc", "work with me", "work with us", "save the date", "workshop on", "pdf", "new work", "job alert", "we are hosting", "open access", "openaccess"]
timelines_filtered = timelines_filtered[~timelines_filtered["text"].str.contains('|'.join(keywords_infotweets), case=False)]
print(f"{len(timelines_filtered)} rows in the filtered (4) df (without informational tweets)")


##### 3. create columns with relevant information and drop irrelevant ones #####
# create is_reply column
timelines_filtered["is_reply"] = timelines_filtered["reply_to_screen_name"].notna()

### derive gender from first names
# extract first names from names
titles_characters = ["dr", "dr.", "professor", "prof", "prof.", "(assoc.)", "a/prof.", "associate", "doc", "mr", "psych"]
timelines_filtered["firstname"] = timelines_filtered["name"].apply(lambda x: x.split()[0].lower() if x.split()[0].lower() not in titles_characters else x.split()[1].lower())

# derive gender with gender-guesser 
# https://gist.github.com/morkapronczay/49593dcb260213233fffb9d08376c111
# instatiate the detector
d = gender.Detector(case_sensitive=False)

# get gender from first names (gender in column 88)
l = len(timelines_filtered)
genders = []

for i in range (0, l):
    gender = d.get_gender(timelines_filtered.iloc[i, 87])
    genders.append(gender)

# add the genders list of strings to the filtered timelines df
timelines_filtered["gender"] = genders

### drop columns we are not interested in
columns_to_keep = ["user_id", "created_at", "text", "is_reply", "gender"]
timelines_filtered = timelines_filtered[columns_to_keep]

### save the filtered df (timelines_filtered) as JSON
timelines_filtered.to_json("/home/mueller/MA_researcher_wellbeing/data/timelines_filtered.json", orient = "records", lines = True)