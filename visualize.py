import matplotlib.pyplot as plt


# Plot the results
def visualize_results(actual_values, predictions):
    print("Plotting the results...")
    plt.plot(actual_values[:, -1], label="Actual rainfall")
    plt.plot(predictions[:, -1], label="Predicted rainfall")
    plt.xlabel("Year")
    plt.ylabel("Rainfall")
    plt.legend()
    plt.show()