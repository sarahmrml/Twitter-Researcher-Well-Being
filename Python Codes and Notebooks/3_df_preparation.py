#################################
####### 3) df preparation #######
#################################

# input: timelines_sentiments.json
# output: h1_df.json, h2_df_female.json, h2_df_male.json

###### 1. prepare session #####
# activate virtual environment "masterenv"
import os
activate_this = os.path.join("/home/mueller/MA_researcher_wellbeing/masterenv", "bin", "activate_this.py")
exec(open(activate_this).read(), {'__file__': activate_this})

# import necessary packages
import pandas as pd
import numpy as np
import json

# import df
timelines_sentiments = pd.read_json("/home/mueller/MA_researcher_wellbeing/data/timelines_sentiments_nomiss.json", lines=True)



###### 2. prepare dfs for analysis ######
### weekly average
# H1
h1_df = timelines_sentiments[["created_at", "neg_sent", "pos_sent"]]
n = h1_df.resample("W", on="created_at").size()
h1_df = h1_df.resample("W", on="created_at").mean()
h1_df["n_all"] = n

# H2 male
h2_df_male = timelines_sentiments[timelines_sentiments["gender"] == "male"]
n = h2_df_male.resample("W", on="created_at").size()
h2_df_male = h2_df_male[["created_at", "neg_sent", "pos_sent"]]
h2_df_male = h2_df_male.resample("W", on="created_at").mean()
h2_df_male["n"] = n
h2_df_male["gender"] = [0] *len(h2_df_male)

# H2 female
h2_df_female = timelines_sentiments[timelines_sentiments["gender"] == "female"]
n = h2_df_female.resample("W", on="created_at").size()
h2_df_female = h2_df_female[["created_at", "neg_sent", "pos_sent"]]
h2_df_female = h2_df_female.resample("W", on="created_at").mean()
h2_df_female["n"] = n
h2_df_female["gender"] = [1] *len(h2_df_female)


### add variables for structural break analysis for interrupted time series (ITS)
# week (T)
data_length = len(h1_df)
weeks = list(range(1, data_length + 1))
h1_df["week(T)"] = weeks
h2_df_male["week_T"] = weeks
h2_df_female["week_T"] = weeks

# pandemic (D)
covid = [0]*52 + [1]*50
h1_df["pandemic(D)"] = covid
h2_df_male["pandemic_D"] = covid
h2_df_female["pandemic_D"] = covid


### save dfs as JSON!
h1_df.to_json("/home/mueller/MA_researcher_wellbeing/data/h1_df.json", orient = "records", lines = True)
h2_df_female.to_json("/home/mueller/MA_researcher_wellbeing/data/h2_df_female.json", orient = "records", lines = True)
h2_df_male.to_json("/home/mueller/MA_researcher_wellbeing/data/h2_df_male.json", orient = "records", lines = True)