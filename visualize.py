################################################################################
# visualize.py
# 29/01/2021
# Caitlin Smith
# 11045132
#
# Universiteit van Amsterdam
# Bachelor Scriptie: Approximating the Hamiltonian of the double pendulum using
# an evolutionary algorithm
#
# Shows all graphs used in the report as well as an animation of the double
# pendulum used in the generated data_derivative
#
# double pendulum:
# https://matplotlib.org/3.1.1/gallery/animation/double_pendulum_sgskip.html
################################################################################
from numpy import sin, cos
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation

import random as rnd
import operator
import math
import re
import copy
import time

import step4_5 as s4
import step3 as s3

m = 1.0
l = 1.00
g = 9.81 #gravitational force
d = 0.0  #drag force

theta_val = 0.4 #radians
t = 0.0
dt = 0.1
dtheta = 0.0
x = [l * np.sin(theta_val)]
y = [-l * np.cos(theta_val)]
time_data = [t]

G = 9.8  # acceleration due to gravity, in m/s^2
L1 = 1.0  # length of pendulum 1 in m
L2 = 1.0  # length of pendulum 2 in m
M1 = 1.0  # mass of pendulum 1 in kg
M2 = 1.0  # mass of pendulum 2 in kg


def dottheta(theta, dtheta, t):
    return dtheta + doubledottheta(theta, dtheta, t)*dt

def doubledottheta(theta, dtheta, t):
    return -(g/l)*np.sin(theta)

def derivs(state, t):

    dydx = np.zeros_like(state)
    dydx[0] = state[1]

    delta = state[2] - state[0]
    den1 = (M1+M2) * L1 - M2 * L1 * cos(delta) * cos(delta)
    dydx[1] = ((M2 * L1 * state[1] * state[1] * sin(delta) * cos(delta)
                + M2 * G * sin(state[2]) * cos(delta)
                + M2 * L2 * state[3] * state[3] * sin(delta)
                - (M1+M2) * G * sin(state[0]))
               / den1)

    dydx[2] = state[3]

    den2 = (L2/L1) * den1
    dydx[3] = ((- M2 * L2 * state[3] * state[3] * sin(delta) * cos(delta)
                + (M1+M2) * G * sin(state[0]) * cos(delta)
                - (M1+M2) * L1 * state[1] * state[1] * sin(delta)
                - (M1+M2) * G * sin(state[2]))
               / den2)

    return dydx

def init():
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text


def animate(i):
    thisx = [0, x1[i], x2[i]]
    thisy = [0, y1[i], y2[i]]

    line.set_data(thisx, thisy)
    time_text.set_text(time_template % (i*dt))
    return line, time_text

def run_vis():
    # show single pendulum data as provided by Schmidt and Lipson
    time_data = []
    theta = []
    omega = []
    x = []
    y = []

    file = open("pendulum_h_1 copy.txt", "r")
    content = file.read().split("\n")
    for line in content[1:-1]:
        line = line.split()
        time_data.append(float(line[1]))
        theta.append(float(line[2]))
        omega.append(float(line[3]))

    plt.plot(time_data, theta, "b")
    plt.plot(time_data, omega, "g")
    plt.show()

    y = []
    for i in range(len(theta)):
        y.append(1.37*omega[i]*omega[i] + 3.27*math.cos(theta[i]))

    #d_da; time derivatie of theta
    #d_db; time derivative of omega
    d_da, d_db = s4.calc_time_derivatives(theta, omega, time_data)
    t_div = s4.time_derivative(d_da, d_db)

    # variables and their respective time derivatives
    plt.title("Theta and d(theta)/dt")
    plt.plot(time_data, theta, "r")
    plt.plot(time_data, d_da, "b")
    plt.show()

    plt.title("Omega and d(omega)/dt")
    plt.plot(time_data, omega, "r")
    plt.plot(time_data, d_db, "b")
    plt.show()

    time_data = time_data[1:]
    d_da = d_da[1:]
    d_db = d_db[1:]

    plt.title("d(theta)/dt and d(omega)/dt")
    plt.plot(time_data, d_da, "r")
    plt.plot(time_data, d_db, "b")
    plt.show()

    # derivative pairings for the single pendulum
    t_div_ab = s4.time_derivative(d_da, d_db)
    t_div_ba = s4.time_derivative(d_db, d_da)

    fig, axs = plt.subplots(2)
    fig.suptitle('Theta / omega vs Omega / theta')
    axs[0].plot(time_data[10:], t_div_ab[10:], "r")
    axs[1].plot(time_data[10:], t_div_ba[10:], "b")
    plt.show()

    # generate data for the single pendulum

    m = 1.0
    l = 1.00
    g = 9.81 #gravitational force
    d = 0.0  #drag force

    theta_val = 0.4 #radians
    t = 0.0
    dt = 0.1
    dtheta = 0.0
    x = [l * np.sin(theta_val)]
    y = [-l * np.cos(theta_val)]
    time_data = [t]

    theta = [theta_val]
    omega = [dottheta(theta_val, dtheta, t)]
    alpha = [doubledottheta(theta_val, dtheta, t)]

    i = 0

    #Over 50 seconds
    while i < 500:
        ddtheta = doubledottheta(theta_val, dtheta, t)
        dtheta = dottheta(theta_val, dtheta, t)
        theta_val = theta_val + dtheta * dt
        t = t + dt

        x.append(l * np.sin(theta_val))
        y.append(-l * np.cos(theta_val))
        time_data.append(t)
        theta.append(theta_val)
        omega.append(dtheta)
        alpha.append(ddtheta)

        i += 1

    # avoid division by zero
    del theta[0]
    del omega[0]
    del alpha[0]
    del time_data[0]
    del x[0]
    del y[0]

    # used once to create generated data text file

    file = open("generated_data.txt","w+")
    for i in range(len(time_data)):
         file.write(str(time_data[i]) + " " + str(x[i]) + " " + str(y[i])+ " " + str(alpha[i])+ " " + str(theta[i]) + " " + str(omega[i]) + "\n")
    file.close()


    # show generated data
    plt.plot(time_data, x, "g")
    plt.plot(time_data, y, "r")
    plt.show()


    plt.plot(time_data, theta, "b")
    plt.plot(time_data, omega, "r")
    plt.plot(time_data, alpha, "g")
    plt.show()

    #d_da; time derivatie of theta
    #d_db; time derivative of omega
    d_da, d_db = s4.calc_time_derivatives(theta, omega, time_data)
    t_div = s4.time_derivative(d_da, d_db)

    # variables and their respective time derivatives
    plt.title("Theta and d(theta)/dt")
    plt.plot(time_data, theta, "r")
    plt.plot(time_data, d_da, "b")
    plt.show()

    plt.title("Omega and d(omega)/dt")
    plt.plot(time_data, omega, "r")
    plt.plot(time_data, d_db, "b")
    plt.show()

    time = time_data[1:]
    d_da = d_da[1:]
    d_db = d_db[1:]

    plt.title("d(theta)/dt and d(omega)/dt")
    plt.plot(time, d_da, "r")
    plt.plot(time, d_db, "b")
    plt.show()

    # derivative pairings for the single pendulum
    t_div_ab = s4.time_derivative(d_da, d_db)
    t_div_ba = s4.time_derivative(d_db, d_da)

    fig, axs = plt.subplots(2)
    fig.suptitle('Theta / omega vs Omega / theta')
    axs[0].plot(time[10:], t_div_ab[10:], "r")
    axs[1].plot(time[10:], t_div_ba[10:], "b")
    plt.show()

    # show double pendulum data as provided by Schmidt and Lipson
    time_double = []
    theta_1 = []
    omega_1 = []
    theta_2 = []
    omega_2 = []

    file = open("real_double_pend_h_1 copy.txt", "r")
    content = file.read().split("\n")
    for line in content[1:-1]:
        line = line.split()
        if float(line[0]) == 0:
            time_double.append(float(line[1]))
            theta_1.append(float(line[2]))
            theta_2.append(float(line[3]))
            omega_1.append(float(line[4]))
            omega_2.append(float(line[5]))

    plt.plot(time_double, theta_1, "b")
    plt.plot(time_double, omega_1, "g")
    plt.plot(time_double, theta_2, "y")
    plt.plot(time_double, omega_2, "r")
    plt.show()



    # generated double pendulum as found on matplotlib.org, seen in file desciption

    G = 9.8  # acceleration due to gravity, in m/s^2
    L1 = 1.0  # length of pendulum 1 in m
    L2 = 1.0  # length of pendulum 2 in m
    M1 = 1.0  # mass of pendulum 1 in kg
    M2 = 1.0  # mass of pendulum 2 in kg

    # create a time array from 0..100 sampled at 0.05 second steps
    dt = 0.05
    t = np.arange(0, 20, dt)

    # th1 and th2 are the initial angles (degrees)
    # w10 and w20 are the initial angular velocities (degrees per second)
    th1 = 160.0
    w1 = 0.0
    th2 = -30.0
    w2 = 0.0

    # initial state
    state = np.radians([th1, w1, th2, w2])

    # integrate your ODE using scipy.integrate.
    y = integrate.odeint(derivs, state, t)

    x1 = L1*sin(y[:, 0])
    y1 = -L1*cos(y[:, 0])

    x2 = L2*sin(y[:, 2]) + x1
    y2 = -L2*cos(y[:, 2]) + y1

    file = open("dp_data.txt", "w")
    for i in range(len(x1)):
        file.write(str(t[i]) + " "+ str(x1[i]) + " "+ str(y1[i]) + " "+ str(x2[i]) + " "+ str(y2[i]) + " "+ str(y[i][0]) + " "+ str(y[i][1]) + " "+ str(y[i][2]) + " "+ str(y[i][3]) + "\n")
    file.close()

    th_1 = []
    w_1 = []
    th_2 = []
    w_2 = []

    for i in range(len(x1)):
        th_1.append(y[i][0])
        w_1.append(y[i][1])
        th_2.append(y[i][2])
        w_2.append(y[i][3])

    plt.plot(t, th_1, "b")
    plt.plot(t, w_1, "g")
    plt.plot(t, th_2, "y")
    plt.plot(t, w_2, "r")
    #plt.legend((line1, line2, line3, line4), ('th_1', 'w_1', 'th_2', 'w_2'))
    plt.show()

    d_da, d_db = s4.calc_time_derivatives(th_1, w_1, t)
    d_de, d_df = s4.calc_time_derivatives(th_2, w_2, t)

    t_div_ab = s4.time_derivative(d_da, d_db)
    t_div_ba = s4.time_derivative(d_db, d_da)

    t_div_ae = s4.time_derivative(d_da, d_de)
    t_div_ea = s4.time_derivative(d_de, d_da)

    t_div_af = s4.time_derivative(d_da, d_df)
    t_div_fa = s4.time_derivative(d_df, d_da)

    t_div_be = s4.time_derivative(d_db, d_de)
    t_div_eb = s4.time_derivative(d_de, d_db)

    t_div_bf = s4.time_derivative(d_db, d_df)
    t_div_fb = s4.time_derivative(d_df, d_db)

    t_div_ef = s4.time_derivative(d_de, d_df)
    t_div_fe = s4.time_derivative(d_df, d_de)

    fig, axs = plt.subplots(2)
    fig.suptitle('Theta_1 & Omega_1')
    axs[0].plot(t[10:], t_div_ab[10:], "r")
    axs[1].plot(t[10:], t_div_ba[10:], "b")
    plt.show()

    fig, axs = plt.subplots(2)
    fig.suptitle('Theta_1 & Theta_2')
    axs[0].plot(t[10:], t_div_ae[10:], "r")
    axs[1].plot(t[10:], t_div_ea[10:], "b")
    plt.show()

    fig, axs = plt.subplots(2)
    fig.suptitle('Theta_1 & Omega_2')
    axs[0].plot(t[10:], t_div_af[10:], "r")
    axs[1].plot(t[10:], t_div_fa[10:], "b")
    plt.show()

    fig, axs = plt.subplots(2)
    fig.suptitle('Omega_1 & Theta_2')
    axs[0].plot(t[10:], t_div_be[10:], "r")
    axs[1].plot(t[10:], t_div_eb[10:], "b")
    plt.show()

    fig, axs = plt.subplots(2)
    fig.suptitle('Omega_1 & Omega_2')
    axs[0].plot(t[10:], t_div_bf[10:], "r")
    axs[1].plot(t[10:], t_div_fb[10:], "b")
    plt.show()

    fig, axs = plt.subplots(2)
    fig.suptitle('Theta_2 & Omega_2')
    axs[0].plot(t[10:], t_div_ef[10:], "r")
    axs[1].plot(t[10:], t_div_fe[10:], "b")
    plt.show()

