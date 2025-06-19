import pandas as pd
import frappe
from surprise import Dataset, Reader, SVD, KNNBasic, accuracy
from surprise.model_selection import cross_validate
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.naive_bayes import MultinomialNB
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.pipeline import make_pipeline

# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.cluster import KMeans
# from sklearn.metrics.pairwise import cosine_similarity
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
# import matplotlib.pyplot as plt

def sample_data_forecasting():
	# Load the data (replace with your file path)
	data = pd.read_csv('/home/danyal/Downloads//Historical Product Demand.csv')

	# 1. Clean the data
	data['Date'] = pd.to_datetime(data['Date'])
	data['Order_Demand'] = pd.to_numeric(data['Order_Demand'].str.strip(), errors='coerce')
	data = data.dropna(subset=['Order_Demand'])
	data = data[data['Order_Demand'] > 0]

	# 2. Filter data for a specific product (e.g., Product_0993)
	product_data = data[data['Product_Code'] == 'Product_0993'].groupby('Date').sum().reset_index()
	product_data.set_index('Date', inplace=True)

	# 3. Split the data into training and testing sets (80% train, 20% test)
	train_size = int(len(product_data) * 0.8)
	train_data, test_data = product_data.iloc[:train_size], product_data.iloc[train_size:]

	# 4. Fit the ARIMA model (p=5, d=1, q=0 as an example)
	arima_model = ARIMA(train_data['Order_Demand'], order=(5, 1, 0))
	arima_model_fit = arima_model.fit()

	# 5. Make predictions
	predictions = arima_model_fit.forecast(steps=len(test_data))

	# 6. Evaluate the model
	mse = mean_squared_error(test_data['Order_Demand'], predictions)
	rmse = np.sqrt(mse)

	print(f'RMSE: {rmse}')

	# 7. Plot the actual vs predicted values
	# plt.plot(test_data.index, test_data['Order_Demand'], label='Actual')
	# plt.plot(test_data.index, predictions, label='Predicted', color='red')
	# plt.legend()
	# plt.show()



def load_data():
	df = pd.read_csv('/home/danyal/Downloads/Reviews.csv')

	# Inspect the first few rows to ensure correct structure
	print(df.head())

	# Define the columns needed: UserID, ProductID, and Rating
	# Adjust column names based on your file structure
	df = df[['user_id', 'product_id', 'rating']]

	# Define a Reader to specify the rating scale
	reader = Reader(rating_scale=(1, 5))

	# Load the dataset into Surprise's format
	data = Dataset.load_from_df(df[['user_id', 'product_id', 'rating']], reader)

	# Split the dataset into training and testing sets
	trainset, testset = train_test_split(data, test_size=0.25)

	# Use the SVD algorithm
	model = SVD()

	# Train the model on the training set
	model.fit(trainset)

	# Test the model on the test set
	predictions = model.test(testset)
	accuracy.rmse(predictions)

	# Function to get product recommendations for a specific user
	def get_top_n_recommendations(user_id, n=5):
		# Get a list of all product ids
		all_products = df['product_id'].unique()

		# Predict ratings for all products that the user hasn't rated yet
		rated_products = df[df['user_id'] == user_id]['product_id']
		unrated_products = [prod for prod in all_products if prod not in rated_products]

		predictions = [model.predict(user_id, prod) for prod in unrated_products]
		predictions.sort(key=lambda x: x.est, reverse=True)

		# Return the top n recommendations
		top_n = predictions[:n]
		return [(pred.iid, pred.est) for pred in top_n]

	# Example usage: get 5 recommendations for a given user_id
	user_id = "ARYVQL4N737A1"  # Replace with the actual user ID
	recommendations = get_top_n_recommendations(user_id, n=5)
	print("Top 5 product recommendations for user:", recommendations)



# Function to get top N item qty recommendations for a given user
def get_top_n_recommendations(algo, user_id, item_ids, n=10):
    # Predict the ratings for the specified user and items
    predictions = [algo.predict(user_id, item_id) for item_id in item_ids]

    # Sort the predictions by estimated rating
    predictions.sort(key=lambda x: x.est, reverse=True)

    # Return the top N recommendations
    return predictions[:n]


def item_recommend_process():
    d = frappe.db.sql(""" Select si.company, sii.item_code, sum(sii.qty) AS rating, si.posting_date
                            from `tabSales Invoice` as si
                            inner join `tabSales Invoice Item` as sii
                            on si.name = sii.parent
                            where si.company="Sultan Group Healthcare"
                            and si.posting_date between '2020-01-01' and '2022-03-31'
                            and si.docstatus = 1 and si.is_return != 1 and si.owner ="osman@sghealthcare.ae" and sii.item_code = "98500005"
                            group by customer, si.name, item_code""", as_dict=1)
    processed_data = []
    for record in d:
        processed_record = {
            'user_id': record.get('company'),
            'item_id': record.get('item_code'),
            'rating': record.get('rating')
        }
        processed_data.append(processed_record)

    data = pd.DataFrame(processed_data)
    print(data.head())
    print(data['rating'].describe())

    cap_value = data['rating'].quantile(0.99)
    data['rating'] = data['rating'].apply(lambda x: min(x, cap_value))

    scaler = MinMaxScaler(feature_range=(0, 10))
    data['rating'] = scaler.fit_transform(data[['rating']])

    # Define a Reader object to parse the DataFrame
    reader = Reader(rating_scale=(data['rating'].min(), data['rating'].max()))

    # Load the dataset into a Surprise Dataset object
    dataset = Dataset.load_from_df(data[['user_id', 'item_id', 'rating']], reader)

    # Use the SVD (Singular Value Decomposition) algorithm
    model = SVD()

    # Evaluate the performance of the algorithm using cross-validation
    cross_validate(model, dataset, measures=['RMSE', 'MAE'], cv=10, verbose=True)

    # Train the algorithm on the whole dataset
    trainset = dataset.build_full_trainset()
    model.fit(trainset)

    # Get all unique item IDs
    all_item_ids = data['item_id'].unique()

    # Get top 5 recommendations for user with user_id=1
    user_id = "Sultan Group Healthcare"
    top_n_recommendations = get_top_n_recommendations(model, user_id, all_item_ids, n=50)

    # Print the recommendations
    print("Top 50 recommendations for user {user_id}:".format(user_id=user_id))
    for pred in top_n_recommendations:
        print("Item ID: {}, Estimated Rating: {:.2f}".format(pred.iid, pred.est))

    # Inspect a single prediction in detail
    # single_prediction = algo.predict(user_id, 49003029)
    # print("Details for Item ID 49003029 - User ID 1:")
    # print("Estimated Rating: {0}".format(single_prediction.est))
    # print("Actual Rating: {0}".format(single_prediction.r_ui if single_prediction.r_ui else 'N/A'))
    # print("Details: {0}".format(single_prediction.details))



# Spam filtering

def filter_spam_emails():
    # Create a synthetic dataset
   #  data = {'keyword_money': [0, 1, 0, 1, 0, 1, 0, 1],
   #         'email_length': [500, 150, 300, 200, 400, 180, 600, 100], 'spam': [0, 1, 0, 1, 0, 1, 0, 1]}
   #  df = pd.DataFrame(data)
   #  # Features and target variable
   #  X = df[['keyword_money', 'email_length']]
   #  y = df['spam']
   # # Splitting the dataset into training and testing sets
   #  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
   #
   #  # Train a linear regression model
   #  linear_model = LinearRegression()
   #  linear_model.fit(X_train, y_train)
   #  # Predict on the test set
   #  y_pred_lin = linear_model.predict(X_test)
   #  # Threshold predictions for classification
   #  y_pred_lin = [1 if pred > 0.5 else 0 for pred in y_pred_lin]
   #  print('Linear Regression Classification Results: ')
   #  print(classification_report(y_test, y_pred_lin))
   #
   #  # Train a k-NN classifier
   #  knn_model = KNeighborsClassifier(n_neighbors=3)
   #  knn_model.fit(X_train, y_train)
   #  # Predict on the test set
   #  y_pred_knn = knn_model.predict(X_test)
   #  print('k - Nearest Neighbors Classification Results: ')
   #  print(classification_report(y_test, y_pred_knn))

    # URL for the Spambase dataset (modify if using another dataset)
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"

    # Load the data into a Pandas DataFrame
    data = pd.read_csv(url, header=None)

    # The dataset does not include column headers, so we need to specify them if known
    # For Spambase, the last column is the label (1 for spam, 0 for non-spam)
    X = data.iloc[:, :-1]  # Features
    y = data.iloc[:, -1]  # Labels
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Initialize the Naive Bayes classifier
    model = MultinomialNB()
    # Train the model
    model.fit(X_train, y_train)
    # Predict the labels for the test set
    y_pred = model.predict(X_test)
    # Print the classification report and accuracy
    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))

def customer_satisfaction():
    from scikits import statsmodels
    # Generate synthetic data
    data = {
           'Age': [25, 30, 35, 40, 45, 50, 55, 60, 20, 65],
           'Purchase_Amount': [120, 200, 150, 300, 250, 500, 450, 100, 180, 400],
           'Store_Location': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
        # Example: 1 for urban, 2 for suburban
        'Customer_Satisfaction': [8, 9, 7, 10, 6, 8, 9, 7, 6, 10]
    }
    df = pd.DataFrame(data)
    # Convert Store Location to categorical variable
    df['Store_Location'] = df['Store_Location'].astype('category')
    # Prepare the independent variables (add a constant term to allow the intercept to be fitted)
    X = df[['Age', 'Purchase_Amount', 'Store_Location']]
    X = statsmodels.add_constant(X)  # adding a constant
    # Dependent variable
    y = df['Customer_Satisfaction']
    # Fit the regression model
    model = statsmodels.OLS(y, X).fit()
    # Print out the statistics
    print(model.summary())

def predict_cost():
    import matplotlib.pyplot as plt

    # Example data
    data = {'Quantity': [10, 15, 20, 25, 30, 35, 40],
           'Price': [12, 15, 20, 22, 24, 25, 28]
    }
    df = pd.DataFrame(data)

    X = df[['Quantity']]
    y = df['Price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    plt.scatter(X, y, color='blue') # Actual points
    plt.plot(X, model.predict(X), color='red') # Regression line
    plt.title('Quantity vs Price')
    plt.xlabel('Quantity')
    plt.ylabel('Price')
    plt.show()


if __name__ == "__main__":
    frappe.init("blox-ml", "/home/danyal/ERPNEXT15/frappe-bench/sites/")
    frappe.connect()
	# sample_data_forecasting()
	# filter_spam_emails()
	# predict_cost()
	# load_data()
	# item_recommend_process()
