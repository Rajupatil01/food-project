
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# CSV लोड करें
df = pd.read_csv("food_delivery_orders_1k.csv")

# Restaurant-wise average delivery time
avg_delivery_time = df.groupby('Restaurant_ID')['Delivery_Time_Minutes'].mean().sort_values(ascending=False)

# Plotting
plt.figure(figsize=(12, 6))
sns.barplot(x=avg_delivery_time.index, y=avg_delivery_time.values, palette='viridis')
plt.xticks(rotation=45)
plt.title("🚚 Average Delivery Time per Restaurant")
plt.xlabel("Restaurant ID")
plt.ylabel("Avg Delivery Time (minutes)")
plt.tight_layout()
plt.show()


import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the CSV
df = pd.read_csv("food_delivery_orders_1k.csv")

# Combine Date & Time
df['Order_DateTime'] = pd.to_datetime(df['Order_Date'] + ' ' + df['Order_Time'])
df['Hour'] = df['Order_DateTime'].dt.hour

# Convert Weather to codes
df['Weather_Code'] = df['Weather'].astype('category').cat.codes

# Define features and target
X = df[['Hour', 'Weather_Code', 'Is_Holiday', 'Is_Weekend']]
y = df['Quantity']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("Mean Squared Error:", mse)
print("R² Score:", r2)

import pandas as pd
import mysql.connector
from sqlalchemy import create_engine

# Load CSV
df = pd.read_csv("food_delivery_orders_1k.csv")

# MySQL connection string (example)
engine = create_engine("mysql+mysqlconnector://root:system1234@localhost:3306/food1")

# Insert into MySQL
df.to_sql(name='orders', con=engine, if_exists='replace', index=False)
print(df)