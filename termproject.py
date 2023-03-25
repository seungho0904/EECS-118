import pymysql
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn import tree
import graphviz

# Connect to the database
db = pymysql.connect(host='localhost', user='mp', passwd='"eecs118"', db='flights')
cur = db.cursor()

# Define the queries
queries = {
    # 5 relational 
    1: "Find the cheapest non-stop flight given airports and a date.",
    2: "Find the flight and seat information for a customer.",
    3: "Find all non-stop flights for an airline.",
    4: "Find all flights that have at least one available seat on a given date and their corresponding departure and arrival airports.",
    5: "Find all customers who have booked a flight on a specific date and departure.",
    # 5 non relational
    6: "Show Line chart of fertility rate of specific country",
    7: "Compare fertility rate between specific country and OECD average.",
    8: "SGDRegressor to estimate future rate",
    9: "Linear regression for relationship between rate and years",
    10: "Visualize Decision tree for diabetes outcome."
}

# Main program loop
while True:
    # Load the data 
    df = pd.read_csv('/Users/seunghokim/Desktop/TermProject/fertility.csv', on_bad_lines='skip')
    # Show the list of queries to the user
    print("Please choose a query to start with:")
    for key, value in queries.items():
        print(key, "-", value)

    # Ask the user to choose a query
    choice = input("Enter your choice: ")

    # Execute the chosen query
    if choice == "1":
        departure = input("Please enter the airport code for the departure airport: ")
        destination = input("Please enter the airport code for the destination airport: ")
        date = input("What is the date of the flight in yyyy-mm-dd? ")
        sql =  f'''
            SELECT Flight_leg.Flight_number, MIN(Fare.Amount) AS Cheapest_fare
            FROM Flight_leg
            JOIN Leg_instance ON Flight_leg.Flight_number = Leg_instance.Flight_number AND Flight_leg.Leg_number = Leg_instance.Leg_number
            JOIN Fare ON Leg_instance.Flight_number = Fare.Flight_number
            WHERE Leg_instance.Leg_date = '{date}' AND Flight_leg.Departure_airport_code = '{departure}' AND Flight_leg.Arrival_airport_code = '{destination}'
            GROUP BY Flight_leg.Flight_number
            HAVING COUNT(*) = 1
            ORDER BY Cheapest_fare ASC
            LIMIT 1;
            '''
        cur.execute(sql)
        result = cur.fetchone()

        if result is None:
            print("No flights found matching the given criteria.")
        else:
            print(f"The cheapest flight is {result[0]}, and the cost is {result[1]}.")
    

    elif choice == "2":
        name = input("Please enter the customer's name: ")
        sql = f'''
            SELECT Flight_number, Seat_number
            FROM Seat_reservation
            WHERE Customer_name = '{name}'
            '''
        cur.execute(sql)
        result = cur.fetchone()
        print(f"The flight number is {result[0]}, and the seat number is {result[1]}.")

    elif choice == "3":
        airline = input("What is the name of the airline? ")
        sql = f'''
            SELECT Flight_number 
            FROM Flight_leg 
            WHERE Flight_number 
            IN (SELECT Flight_number FROM FLight WHERE Airline='{airline}') 
            GROUP BY Flight_number 
            HAVING COUNT(*) = 1;
            '''
        cur.execute(sql)
        result = cur.fetchall()
        if len(result) == 0:
            print("No non-stop flights found for the given airline.")
        else:
            print("Non-stop flights for {}:".format(airline))
            for row in result:
                print(row[0])
        
    elif choice == "4":
        date = input("What is the date of the desired flight? ")
        sql = f'''
            SELECT Flight_leg.Flight_number, Flight_leg.Departure_airport_code, Flight_leg.Arrival_airport_code 
            FROM Flight_leg 
            JOIN Leg_instance 
            ON Flight_leg.Flight_number = Leg_instance.Flight_number AND Flight_leg.Leg_number = Leg_instance.Leg_number 
            WHERE Leg_instance.Leg_date = '{date}' AND Leg_instance.Number_of_available_seats > 0
            '''
        cur.execute(sql)
        result = cur.fetchall()
        print("The flights with available seats are:")
        for row in result:
            print(f"Flight number: {row[0]}, Departure airport: {row[1]}, Arrival airport: {row[2]}")
    
    elif choice == "5":
        date = input("Please enter the date (yyyy-mm-dd): ")
        departure_airport = input("Please enter the departure airport code: ")
        sql = f'''
            SELECT sr.Customer_name
            FROM Seat_reservation sr
            JOIN Flight_leg fl ON sr.Flight_number = fl.Flight_number AND sr.Leg_number = fl.Leg_number
            WHERE fl.Departure_airport_code = '{departure_airport}' AND sr.Reservation_date = '{date}'
            '''
        cur.execute(sql)
        results = cur.fetchall()
        if len(results) == 0:
            print("No customers have booked a flight on this date.")
        else:
            print("Customers who have booked a flight on this date:")
            for row in results: 
                print(row[0])
    #----------Non relational queries -----------
    elif choice == "6": 
        #print(list(df['CountryName']))
        name = input("Please enter the Country Name (Australia, India, Aruba etc): ")
        # Set the index to the name column
        df.set_index('CountryName', inplace=True)
        # Select the data you want to plot
        data = df.loc[name][5:]
        # Plot the data as a line chart
        plt.plot(data.index, data.values, marker='o')
        plt.grid(True)
        plt.ylim(0, 10)
        plt.xticks(rotation=90)
        plt.title(f'Fertility rate in {name} over time')
        plt.xlabel('Year')
        plt.ylabel('Fertility rate')
        plt.show()

    elif choice == "7":
        #print(list(df['CountryName']))
        name = input("Please enter the Country Name (Australia, India, Aruba etc): ")
        # Set the index to the name column
        df.set_index('CountryName', inplace=True)
        # Select the data you want to plot
        data = df.loc[name][5:]
        data2 = df.loc["OECD members"][5:]
        # plot the first line
        plt.plot(data.index, data.values, label=name, marker='o')
        # plot the second line
        plt.plot(data.index, data2.values, label='OECD', marker='o')
        plt.legend()
        plt.grid(True)
        plt.ylim(0, 10)
        plt.xticks(rotation=90)
        plt.title(f'Fertility rate in {name} and OECD over time')
        plt.xlabel('Year')
        plt.ylabel('Fertility rate')
        plt.show()


    elif choice == "8":
        # Prepare X and y data
        #X = df.drop(columns=["CountryName", "CountryCode", "IndicatorName", "IndicatorCode", "2021"])
        #y = df["2021"]
        df.set_index('CountryCode', inplace=True)
        data = df.loc['BEL'][5:]        
        X = data.index
        y = data.values  
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train = X_train.values.reshape(-1, 1)
        X_test = X_test.values.reshape(-1, 1)
        # Create the model
        model = SGDRegressor() 
        
        # Train the model with different learning rates and observe its performance
        learning_rates = [0.0001, 0.001, 0.01, 0.1, 1]
        mse_scores = []
        for lr in learning_rates:
            model.set_params(learning_rate="constant", eta0=lr)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test) / (10 ** 18)
            mse_scores.append(mean_squared_error(y_test, y_pred)) 

        # Visualize the performance of the model
        plt.plot(learning_rates, mse_scores)
        plt.title('Effect of Learning Rate on Model Performance')
        plt.xlabel('Learning Rate')
        plt.ylabel('Mean Squared Error')
        plt.show()

        # Fit the model with the optimal learning rate
        optimal_lr = learning_rates[mse_scores.index(min(mse_scores))]
        model.set_params(learning_rate="constant", eta0=optimal_lr)
        model.fit(X_train, y_train)

        # Predict the fertility rate for a given year and plot the results
        year = 2022
        fertility_rate = abs(model.predict([[year]])[0] / (10 ** 18))
        print(f'The predicted fertility rate for the year {year} is {fertility_rate:.2f}')
    
    elif choice == "9":
        #print(list(df['CountryCode']))
        name = input("Please enter the Country Code (ABW, BEL, CHN): ")
        # Set the index to the name column
        df.set_index('CountryCode', inplace=True)
        # Select the data you want to plot
        data = df.loc[name][5:]        
        X = data.index
        y = data.values  
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.5, random_state=1)
        X_train = X_train.values.reshape(-1, 1)
        X_test = X_test.values.reshape(-1, 1)
        # fit a linear regression model to the data
        lm = LinearRegression()
        lm.fit(X_train, Y_train)
        r_squared = lm.score(X_train, Y_train)
        a = lm.coef_ 
        b = lm.intercept_ 
        print("R squared score:", r_squared)
        print("a:", a)
        print("b:", b)
        # predict new y values using the fitted model
        y_pred = lm.predict(X_test)
        X_train = X_train.tolist()
        Y_train = Y_train.tolist()
        X_test = X_test.tolist()
        Y_test = Y_test.tolist()
        plt.scatter(X_train, Y_train, color='blue', label='Training data')
        plt.scatter(X_test, Y_test, color='green', label='Testing data')
        x_seq = np.arange(1960, 2021, 1)
        y_seq = lm.predict(x_seq.reshape(-1,1))
        plt.plot(x_seq, y_seq, color='red', label='Regression line')
        plt.title(f'Relationship between fertility rate and year in {name}')
        plt.xlabel('year')
        plt.ylabel('rate')
        plt.legend()
        plt.show()

    elif choice == "10":  
        df = pd.read_csv('./diabetes.csv')
        y = df['Outcome']              #get 'outcome' and assign it to y
        X = df.drop('Outcome', axis=1) #get all other columns as features to X
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, y_pred)
        dot_data = tree.export_graphviz(model, out_file=None, feature_names=X_train.columns, class_names=['0', '1'], filled=True, rounded=True, special_characters=True)
        # visualization on decision tree
        graph = graphviz.Source(dot_data)
        graph.render("diabetes_tree")
         


    else:
        print("Invalid choice. Please try again.")
        continue

    # Ask the user if they want to continue or quit
    answer = input("Do you want to choose another query? (y/n) ")
    if answer.lower() != "y":
        break

# Close the database connection
db.close()