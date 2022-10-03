import pandas as pd
import datetime as dt
pd.set_option("display.max_columns",None)
pd.set_option("display.max_rows",None)
pd.set_option("display.width",500)
pd.set_option("display.expand_frame_repr",False)
from mlxtend.frequent_patterns import apriori, association_rules

#Step 1: Read the "data.csv" thoroughly.
#Data cannot be shared because it is private
df=pd.read_csv("")
df.head()

#Step 2: ServiceID represents a different service for each CategoryID. Combine ServiceID and CategoryID with "_" to create a new variable to represent these services. Output to be achieved:
df["Service"]=df["ServiceId"].astype(str) + "_" + df["CategoryId"].astype(str)

#Step 3: The data set consists of the date and time the services were received, there is no basket definition (invoice, etc.).In order to apply Association Rule Learning, a basket (invoice, etc.) definition must be created. Here, the basket definition is the services each customer receives monthly.
df["CartId"]=df["UserId"].astype(str) + "_" + pd.to_datetime(df["CreateDate"]).dt.to_period("M").astype(str)

#Task 2: Generate and Suggest Association Rules

#Step 1: Create basket, service pivot table as below.
df_pivot = pd.pivot_table(df,index="CartId",columns="Service",aggfunc={"CategoryId":"count"}).fillna("0").astype(int)
df_pivot=df_pivot.astype(bool).astype(int)
df_pivot=df_pivot.droplevel(0,axis=1)
df_pivot.head()

#Step 2: Create association rules.
frequency_itemself=apriori(df_pivot,min_support=0.01,use_colnames=True)
rules=association_rules(frequency_itemself,metric="support",min_threshold=0.01)
rules=rules.sort_values("support",ascending=False)

#Step 3: Using the arl recommender function, recommend a service to a user who has received the 2_0 service in the last 1 month.
df.loc[df["Service"]=="2_0"].sort_values("CreateDate",ascending=False).head()
#10591

rules["antecedents"]=rules["antecedents"].apply(lambda x: list(x)[0]).astype("unicode")
rules["consequents"]=rules["consequents"].apply(lambda x: list(x)[0]).astype("unicode")
rules.loc[rules["antecedents"]=="2_0"]
recommendation_list= [rules["consequents"].iloc[i] for i,antecedent in enumerate(rules["antecedents"]) if antecedent == "2_0"]
