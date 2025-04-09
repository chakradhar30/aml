exp 1:

import numpy as np
import matplotlib.pyplot as plt

def mean_squared_error(y_true, y_predicted):
    cost = np.mean((y_true - y_predicted) ** 2)
    return cost

def gradient_descent(x, y, iterations=1000, learning_rate=0.0001, stopping_threshold=1e-6):
    current_coef = 0.1
    current_intercept = 0.01
    n = float(len(x))
    costs = []
    coef = []
    previous_cost = None

    for i in range(iterations):
        y_predicted = (current_coef * x) + current_intercept
        current_cost = mean_squared_error(y, y_predicted)

        if previous_cost and abs(previous_cost - current_cost) <= stopping_threshold:
            break

        previous_cost = current_cost
        costs.append(current_cost)
        coef.append(current_coef)

        coef_derivative = -(1 / n) * np.sum(x * (y - y_predicted))
        intercept_derivative = -(1 / n) * np.sum(y - y_predicted)

        current_coef -= learning_rate * coef_derivative
        current_intercept -= learning_rate * intercept_derivative

        print(f"Iteration {i + 1}: Cost {current_cost}, coef: {current_coef}, intercept: {current_intercept}")

    plt.figure(figsize=(8, 6))
    plt.plot(coef, costs)
    plt.scatter(coef, costs, marker='o', color='red')
    plt.title("Cost vs coef")
    plt.ylabel("Cost")
    plt.xlabel("coef")
    plt.show()

    return current_coef, current_intercept

def main():
    X = np.array([32.50234527, 53.42680403, 61.53035803, 47.47563963, 59.81320787,
                  55.14218841, 52.21179669, 39.29956669, 48.10504169, 52.55001444,
                  45.41973014, 54.35163488, 44.1640495, 58.16847072, 56.72720806,
                  48.95588857, 44.68719623, 60.29732685, 45.61864377, 38.81681754])

    Y = np.array([31.70700585, 68.77759598, 62.5623823, 71.54663223, 87.23092513,
                  78.21151827, 79.64197305, 59.17148932, 75.3312423, 71.30087989,
                  55.16567715, 82.47884676, 62.00892325, 75.39287043, 81.43619216,
                  60.72360244, 82.89250373, 97.37989686, 48.84715332, 56.87721319])

    estimated_coef, estimated_intercept = gradient_descent(X, Y, iterations=2000)

    print(f"Estimated coef: {estimated_coef}\nEstimated intercept: {estimated_intercept}")

    Y_pred = estimated_coef * X + estimated_intercept

    plt.figure(figsize=(8, 6))
    plt.scatter(X, Y, marker='o', color='red')
    plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='blue', linestyle='dashed')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Linear Regression: Actual vs Predicted")
    plt.show()

if __name__ == "__main__":
    main()

exp 2:

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

df = pd.read_csv("housingdata.csv")
df_filled = df.fillna(df.mean(numeric_only=True))
X = df_filled.drop(columns=['MEDV'])
y = df_filled['MEDV']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_rmse = mean_squared_error(y_test, lr_pred, squared=False)
print("Simple Linear Regression RMSE:", lr_rmse)

lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train, y_train)
lasso_pred = lasso_model.predict(X_test)
lasso_rmse = mean_squared_error(y_test, lasso_pred, squared=False)
print("Lasso Regression RMSE:", lasso_rmse)

ridge_model = Ridge(alpha=0.1)
ridge_model.fit(X_train, y_train)
ridge_pred = ridge_model.predict(X_test)
ridge_rmse = mean_squared_error(y_test, ridge_pred, squared=False)
print("Ridge Regression RMSE:", ridge_rmse)

elasticnet_model = ElasticNet(alpha=0.1, l1_ratio=0.5)
elasticnet_model.fit(X_train, y_train)
elasticnet_pred = elasticnet_model.predict(X_test)
elasticnet_rmse = mean_squared_error(y_test, elasticnet_pred, squared=False)
print("Elastic Net Regression RMSE:", elasticnet_rmse)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, lr_pred, color='blue', label='Predicted')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Actual')
plt.title('Simple Linear Regression')
plt.xlabel('Actual MEDV')
plt.ylabel('Predicted MEDV')
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(y_test, lasso_pred, color='blue', label='Predicted')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Actual')
plt.title('Lasso Regression')
plt.xlabel('Actual MEDV')
plt.ylabel('Predicted MEDV')
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(y_test, ridge_pred, color='blue', label='Predicted')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Actual')
plt.title('Ridge Regression')
plt.xlabel('Actual MEDV')
plt.ylabel('Predicted MEDV')
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(y_test, elasticnet_pred, color='blue', label='Predicted')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Actual')
plt.title('Elastic Net Regression')
plt.xlabel('Actual MEDV')
plt.ylabel('Predicted MEDV')
plt.legend()
plt.show()


exp 3:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

data = pd.read_csv("AML/Social_Network_Ads.csv")
x = data.iloc[:, [2, 3]].values
y = data.iloc[:, 4].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

classifier = RandomForestClassifier(n_estimators=10, criterion='gini')
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)
print(classification_report(y_test, y_pred))

sns.heatmap(cm, annot=True)
plt.title("Confusion Matrix Heatmap")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


exp 4 :

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf

data = pd.read_csv('AML/vimana.csv') 
print(data.head())

ts = data['demand']

ts.plot(figsize=(10, 6))
plt.title('Time Series Data')
plt.xlabel('Time')
plt.ylabel('Demand')
plt.show()

train_size = int(len(ts) * 0.8)
train_data, test_data = ts[:train_size], ts[train_size:]

p = len(train_data) // 2

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_pacf(train_data, lags=p, ax=axes[0])
plot_acf(train_data, lags=p, ax=axes[1])
axes[0].set_title('Partial Autocorrelation Function (PACF)')
axes[1].set_title('Autocorrelation Function (ACF)')
plt.show()

p = 2

ar_model = sm.tsa.AutoReg(train_data, lags=p)
arima_result = ar_model.fit() 

forecast = arima_result.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1)

plt.figure(figsize=(10, 6))
plt.plot(test_data.index, test_data.values, label='Actual')
plt.plot(test_data.index, forecast, label='Predicted')
plt.title('AR Model Forecast')
plt.xlabel('Time')
plt.ylabel('Demand')
plt.legend()
plt.show()

exp 5 :

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

data = pd.read_csv("AML/airline_passengers.csv")
data['Month'] = pd.to_datetime(data['Month'])
data.set_index('Month', inplace=True)

if data.isnull().sum().sum() > 0:
    print("Warning: There are missing values in the dataset.")

plt.figure(figsize=(10, 6))
plt.plot(data)
plt.title('Airline Passenger Data')
plt.xlabel('Year')
plt.ylabel('Passenger Count')
plt.show()

def check_stationarity(data):
    result = adfuller(data)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t{}: {}'.format(key, value))

check_stationarity(data)

diff_data = data.diff().dropna()

plt.figure(figsize=(10, 6))
plt.plot(diff_data)
plt.title('Differenced Airline Passenger Data')
plt.xlabel('Year')
plt.ylabel('Differenced Passenger Count')
plt.show()

check_stationarity(diff_data)

plot_acf(diff_data, lags=20)
plt.title('Autocorrelation Function (ACF)')
plt.show()

plot_pacf(diff_data, lags=20)
plt.title('Partial Autocorrelation Function (PACF)')
plt.show()

p = 2
d = 1
q = 2

model = ARIMA(data, order=(p, d, q))
model_fit = model.fit()

print(model_fit.summary())

residuals = pd.DataFrame(model_fit.resid)
plt.figure(figsize=(10, 6))
plt.plot(residuals)
plt.title('Residuals')
plt.xlabel


exp 6 :

import pandas as pd
import numpy as np
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from apyori import apriori

dataset = pd.read_csv("AML/Groceries_dataset.csv")
print('Dimensions of dataset are:', dataset.shape)
dataset = dataset.drop(columns='Member_number')
dataset = dataset[dataset['itemDescription'] != 'bags']
dataset['Date'] = pd.to_datetime(dataset['Date'], format='%d-%m-%Y')
dataset = dataset.groupby('Date')['itemDescription'].apply(list).reset_index()

transactions = []
for indexer in range(len(dataset)):
    transactions.append(dataset['itemDescription'].iloc[indexer])

rules = apriori(transactions=transactions,
                min_support=0.00412087912,
                min_confidence=0.6,
                min_lift=1.9,
                min_length=2,
                max_length=2)

results = list(rules)

def inspect(results):
    lhs = [tuple(result[2][0][0])[0] for result in results]
    rhs = [tuple(result[2][0][1])[0] for result in results]
    supports = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidences, lifts))

resultsinDataFrame = pd.DataFrame(inspect(results),
                                  columns=['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])

resultsinDataFrame.sort_values('Confidence', ascending=False)

resultFrame = resultsinDataFrame.iloc[:, 0:-3]
print(resultFrame)

resultFrame = resultFrame.groupby('Left Hand Side')['Right Hand Side'].apply(list).reset_index()
resultFrame = resultFrame.rename(columns={'Left Hand Side': 'Product Purchased', 'Right Hand Side': 'Also Purchased Along'})
print(resultFrame)


exp 7 :

import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine, correlation

rating_df = pd.read_csv("AML/ratings.csv")
rating_df.head(5)
rating_df.drop('timestamp', axis=1, inplace=True)
print(len(rating_df.userId.unique()))
print(len(rating_df.movieId.unique()))

user_movies_df = rating_df.pivot(index='userId', columns='movieId', values='rating').reset_index(drop=True)
print(user_movies_df.head())

user_movies_df.index = rating_df.userId.unique()
user_movies_df.iloc[0:5, 0:15]

user_movies_df.fillna(0, inplace=True)
user_movies_df.iloc[0:5, 0:10]

user_sim = 1 - pairwise_distances(user_movies_df.values, metric='cosine')
user_sim_df = pd.DataFrame(user_sim)
print(user_sim_df)

user_sim_df.index = rating_df.userId.unique()
user_sim_df.columns = rating_df.userId.unique()
user_sim_df.iloc[0:5, 0:5]

rating_mat = rating_df.pivot(index='movieId', columns='userId', values='rating').reset_index(drop=True)
rating_mat.fillna(0, inplace=True)

movie_sim = 1 - pairwise_distances(rating_mat.values, metric='cosine')

movie_sim_df = pd.DataFrame(movie_sim)
movie_sim_df.iloc[0:5, 0:5]
print(movie_sim_df)

movies_df = pd.read_csv("AML/movies.csv")
movies_df[0:5]
movies_df.drop('genres', axis=1, inplace=True)

def get_user_similar_movies(user1, user2):
    common_movies = rating_df[rating_df.userId == user1].merge(rating_df[rating_df.userId == user2], on='movieId', how='inner')
    return common_movies.merge(movies_df, on='movieId')

common_movies = get_user_similar_movies(2, 332)
print(common_movies)

movie_sim_df.shape

def get_similar_movies(movieid, topN=5):
    movieidx = movies_df[movies_df.movieId == movieid].index[0]
    movies_df['similarity'] = movie_sim_df.iloc[movieidx]
    top_n = movies_df.sort_values(['similarity'], ascending=False)[0:topN]
    return top_n

movies_df[movies_df.movieId == 231]
print(get_similar_movies(231))


exp 8:

import pandas as pd
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("AML/IMDB Dataset.csv")
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
print(df)
x = df['review']
y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

vectorizer = CountVectorizer(stop_words='english')
X_train_counts = vectorizer.fit_transform(X_train)
clf = MultinomialNB()
clf.fit(X_train_counts, y_train)

X_test_counts = vectorizer.transform(X_test)
y_pred = clf.predict(X_test_counts)

print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(classification_report(y_test, y_pred))

def classify_review(review):
    review_counts = vectorizer.transform([review])
    return 'positive' if clf.predict(review_counts)[0] == 1 else 'negative'

review = "the movie was phenomenal"
print(f'Review: {review}\nSentiment: {classify_review(review)}')

review2 = "the movie was a torture"
print(f'Review2: {review2}\nSentiment: {classify_review(review2)}')


exp 9 :

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab') 

data = pd.read_csv("AML/sentiment_train.txt", delimiter='\t')
print(data)

X = data['text']
y = data['sentiment']

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token not in string.punctuation]
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    cleaned_text = ' '.join(tokens)
    return cleaned_text

X_cleaned = X.apply(preprocess_text)
print(X_cleaned)

X_train, X_test, y_train, y_test = train_test_split(X_cleaned, y, test_size=0.2, random_state=42)

vectorizer = CountVectorizer(ngram_range=(1, 2))
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

classifier = MultinomialNB()
classifier.fit(X_train_vectorized, y_train)
y_pred = classifier.predict(X_test_vectorized)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


exp 10 :

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

data = pd.read_csv("AML\diabetes.csv")
print(data.head())

X = data.drop('Outcome', axis=1) 
y = data['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

matrix = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)
matrix.plot()
plt.show()

def classify_patient(data):
    data_df = pd.DataFrame([data], columns=X.columns)
    prediction = classifier.predict(data_df)
    return 'Diabetic' if prediction[0] == 1 else 'Non-Diabetic'

new_patient = {
    'Pregnancies': 2,
    'Glucose': 90,
    'BloodPressure': 70,
    'SkinThickness': 20,
    'Insulin': 180,
    'BMI': 25.5,
    'DiabetesPedigreeFunction': 0.5,
    'Age': 33
}

print(f"New patient classification: {classify_patient(new_patient)}")
