if __name__ == "__main__":
    from k_means import calculate_error, lloyd_algorithm  # type: ignore
else:
    from .k_means import lloyd_algorithm, calculate_error

import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset, problem


@problem.tag("hw4-A", start_line=1)
def main():
    """Main function of k-means problem

    Run Lloyd's Algorithm for k=10, and report 10 centers returned.

    NOTE: This code takes a while to run. For debugging purposes you might want to change:
        x_train to x_train[:10000]. CHANGE IT BACK before submission.
    """
    (x_train, _), (x_test, _) = load_dataset("mnist")
    
    # Run Lloyd's Algorithm
    num_centers = 10
    epsilon = 1e-3
    centers, errors = lloyd_algorithm(x_train, num_centers, epsilon)
    # print("Center shape:", centers.shape)
    # print("First few centers:", centers[:5])

    plt.figure(figsize=(16, 9))
    for i in range(10):     
        plt.subplot(2, 5, i+1)
        plt.imshow(centers[i,:].reshape((28,28)))
        plt.xlabel(f"center {i+1}")

    plt.savefig('centers_colored.jpg')
    plt.show()


if __name__ == "__main__":
    main()
