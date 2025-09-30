from flask import Flask, Response

app = Flask(__name__)

EXPERIMENTS_TEXT = '''Experiment 2 : 

import pandas as pd
df = pd.read_csv("sales.csv")
# print(df.head())
print("Slice :")
slice_df = df[df["Product"] == "Laptop"]
print(slice_df)
print("")
print("Dice :")
dice_df  = df[(df["Product"] == "Laptop") & (df["Region"] == "North")]
print(dice_df)
print("")
print("Roll up :")
roll_up = df.groupby("Region")["Sales"].sum().reset_index()
print(roll_up)
print("")
print("Drill down :")
drill_down = df.groupby(["Time", "Region"])["Sales"].sum().reset_index()
print(drill_down)
print("")
print("pivot table :")
pivot_table = pd.pivot_table(df, values="Sales", index="Region", columns="Product", aggfunc="sum", fill_value=0)
print(pivot_table)

Experiment 3

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
# Load data
df = pd.read_csv("drug.csv")
# Features and target
X = df[["Age", "Sex", "BP", "Cholesterol", "Na_to_K"]]
y = df["Drug"]
# Convert categorical variables into numbers
X_encoded = pd.get_dummies(X, drop_first=True)
# Train decision tree with entropy (ID3-like)
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)  # keep tree simple
clf.fit(X_encoded, y)
# Print tree rules
rules = export_text(clf, feature_names=list(X_encoded.columns))
print(rules)


Experiment 4

import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
df = pd.read_csv("insta.csv")
X = df[["Instagram visit score"]]
y = df["Spending_rank(0 to 100)"]
model = LinearRegression()
model.fit(X, y)
print("Intercept: ", model.intercept_)
print("Coefficient: ", model.coef_[0])
df["Predicted Spending"] = model.predict(X)
print(df.head(10))
plt.scatter(X, y, color="blue", label="Actual")
plt.plot(X, df["Predicted Spending"], color="red", linewidth=2, label="Predicted")
plt.xlabel("Instagram visit score")
plt.ylabel("Spending_rank(0 to 100)")
plt.title("Linear Regression Model")
plt.show()

Experiment 5

import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
df = pd.read_csv("insta.csv")
X = df[["Instagram visit score" , "Spending_rank(0 to 100)"]]
kmeans = KMeans(n_clusters=3, random_state=0)
df["Cluster"] = kmeans.fit_predict(X)
print(df.head())
plt.scatter(df["Instagram visit score"], df["Spending_rank(0 to 100)"], c=df["Cluster"], cmap = "viridis")
plt.xlabel("Instagram visit score")
plt.ylabel("Spending rank")
plt.title("KMeans Clustering of insta Users")
plt.show()

Experiment 6

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
# Step 1: Load dataset
df = pd.read_csv("insta.csv")
# Step 2: Select features
X = df[["Instagram visit score", "Spending_rank(0 to 100)"]]
# Step 3: Apply Hierarchical Clustering (Agglomerative)
hc = AgglomerativeClustering(n_clusters=3, linkage='ward')
df["Cluster"] = hc.fit_predict(X)
# Step 4: Print sample results
print(df.head())
# Step 5: Visualize clusters
plt.scatter(df["Instagram visit score"], df["Spending_rank(0 to 100)"],
            c=df["Cluster"], cmap="rainbow")
plt.xlabel("Instagram visit score")
plt.ylabel("Spending rank")
plt.title("Hierarchical Clustering of Insta Users")
plt.show()


Experiment 7

import pandas as py 
from mlxtend.frequent_patterns import apriori, association_rules
df = pd.read_csv("transactions_onehot.csv")
basket = df.drop('TransactionID', axis=1)
frequent_itemsets = apriori(basket, min_support=0.05, use_colnames = True)
print("Frequent Itemsets: ")
print(frequent_itemsets.sort_values('support', ascending=False).head(10))
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.3)
rules = rules.sort_values('lift', ascending=False)
print("Top Association Rules:")
Print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))
best = rules.iloc[0]
print(f"Best rule : {list(best['antecedents'])} -> {list(best['consequents'])}")

create an api in python that provides this text as result when curl
'''

@app.route("/", methods=["GET"])
def experiments():
    # return plain text so `curl` prints it directly
    return Response(EXPERIMENTS_TEXT, mimetype="text/plain; charset=utf-8")

if __name__ == "__main__":
    # development server; set host=0.0.0.0 to be reachable from other machines
    app.run(host="0.0.0.0", port=5000, debug=True)