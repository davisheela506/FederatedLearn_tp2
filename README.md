# TP1: Introduction to Flower Framework

In this tp1, I explored the implementation of a Federated Learning system using the Flower framework. 

#Step 1. Generated simulated distributed dataset with different class distributions across clients.

#Step 2. Designed and implemented a machine learning model to be used by clients in federated training.

#Step 3. Implemented a abstract class flwr.client.Client to define client-side operations.

#Step 4. Run Individual Clients: You will create a script run client that
creates a client with its assigned dataset and model.

#Step 5. Implementing the Server’s Client Manager: You will extend the ab-
stract class flwr.server.ClientManager that allows the server to man-
age participating clients.

#Step 6. Implementing a Basic Federated Learning Strategy: You will ex-
tend flwr.server.Strategy to define an basic aggregation strategy (Fe-
dAvg).

#Step 7. Running the Server: You will implement and a script start server
that launches the federated learning server. 

#Step 8. Running a Full FL Simulation: You will implement and execute the
final script run simulation that starts the server and deploys multiple
clients for federated training. 

#Step 9. Analyzing FL with Different Configurations: You will conduct mul-
tiple simulations with varying hyperparameters (e.g., number of clients,
data heterogeneity) and analyze the effects of them on model performance
and convergence.
