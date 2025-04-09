import numpy as np
import matplotlib.pyplot as plt

# Function to compute the mean squared error (MSE)
def mean_squared_error(y_true, y_predicted):
    # Calculating the loss or cost
    cost = np.mean((y_true - y_predicted) ** 2)
    return cost

# Gradient Descent Function
def gradient_descent(x, y, iterations=1000, learning_rate=0.0001, stopping_threshold=1e-6):
    # Initializing weight, bias, learning rate, and iterations
    current_coef = 0.1
    current_intercept = 0.01
    n = float(len(x))
    costs = []
    coef = []
    previous_cost = None
   
    # Estimation of optimal parameters using Gradient Descent
    for i in range(iterations):
        # Making predictions
        y_predicted = (current_coef * x) + current_intercept
       
        # Calculating the current cost
        current_cost = mean_squared_error(y, y_predicted)
       
        # If the change in cost is less than or equal to stopping_threshold, we stop the gradient descent
        if previous_cost and abs(previous_cost - current_cost) <= stopping_threshold:
            break
       
        previous_cost = current_cost
        costs.append(current_cost)
        coef.append(current_coef)
       
        # Calculating the gradients
        coef_derivative = -(1 / n) * np.sum(x * (y - y_predicted))
        intercept_derivative = -(1 / n) * np.sum(y - y_predicted)
       
        # Updating weights and bias
        current_coef -= learning_rate * coef_derivative
        current_intercept -= learning_rate * intercept_derivative
       
        # Printing the parameters for each 1000th iteration
        print(f"Iteration {i + 1}: Cost {current_cost}, coef: {current_coef}, intercept: {current_intercept}")
   
    # Visualizing the weights and cost for all iterations
    plt.figure(figsize=(8, 6))
    plt.plot(coef, costs)
    plt.scatter(coef, costs, marker='o', color='red')
    plt.title("Cost vs coef")
    plt.ylabel("Cost")
    plt.xlabel("coef")
    plt.show()
   
    return current_coef, current_intercept

def main():
    # Data
    X = np.array([32.50234527, 53.42680403, 61.53035803, 47.47563963, 59.81320787,
                  55.14218841, 52.21179669, 39.29956669, 48.10504169, 52.55001444,
                  45.41973014, 54.35163488, 44.1640495, 58.16847072, 56.72720806,
                  48.95588857, 44.68719623, 60.29732685, 45.61864377, 38.81681754])
   
    Y = np.array([31.70700585, 68.77759598, 62.5623823, 71.54663223, 87.23092513,
                  78.21151827, 79.64197305, 59.17148932, 75.3312423, 71.30087989,
                  55.16567715, 82.47884676, 62.00892325, 75.39287043, 81.43619216,
                  60.72360244, 82.89250373, 97.37989686, 48.84715332, 56.87721319])
   
    # Estimating weight and bias using gradient descent
    estimated_coef, estimated_intercept = gradient_descent(X, Y, iterations=2000)
   
    print(f"Estimated coef: {estimated_coef}\nEstimated intercept: {estimated_intercept}")
   
    # Making predictions using estimated parameters
    Y_pred = estimated_coef * X + estimated_intercept
   
    # Plotting the regression line
    plt.figure(figsize=(8, 6))
    plt.scatter(X, Y, marker='o', color='red')
    plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='blue', linestyle='dashed')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Linear Regression: Actual vs Predicted")
    plt.show()

if __name__ == "__main__":
    main()
