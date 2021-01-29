The data files included contain the following data;

generated_data.txt: [time, x, y, alpha, theta, omega]
dp_data.txt: [time, x_1, y_1, x_2, y_2, theta_1, omega_1, theta_2, omega_2]
 Where generated_data.txt contains the generated data for the single pendulum and
  dp_data.txt stores the produced data for the double pendulum.

 To use the algorithm run main.py in the terminal. This will prompt the question
  if the user would like to see the data visualised. After which another prompt
  will be shown, asking which system to run. Here a choice of three systems can
  be made:

  * The single pendulum; positional data
  * The single pendulum; angular data
  * The double pendulum

 This will run until the desired output has been met using the prior established
 values and input. To change these it is possible to add and remove items from
 the input\_var dictionary in main.py, as well as that it is possible to change
 the exit statement.


 Terminal input: 
 python3 main.py
