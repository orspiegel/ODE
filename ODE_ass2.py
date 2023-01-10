import numpy as np
import matplotlib.pyplot as plt
import math
from tabulate import tabulate
"""
Or Spiegel 
318720067
"""
def deriv(x, y_n):
    return (y_n / x) + 1

def numeric(x_val,h):
    x_0, y_0 = 1, 1
    # create x axis correlated to the h (step size)
    xAxis = np.arange(x_0, x_val, h)

    y_prev = y_0
    deriv_prev = 0
    y_vec = []
    # find y value for each x coordinate
    for x in xAxis:
        y_n = y_prev + h * deriv_prev
        deriv_prev = deriv(x, y_n)
        y_vec.append(y_n)
        y_prev = y_n
    return xAxis, y_vec

def numeric_improve(x_val, h):
    x_0, y_0 = 1, 1
    # create x axis correlated to the h (step size)
    xAxis = np.arange(x_0, x_val, h)

    y_prev = y_0
    deriv_prev = 0
    y_vec = []
    # find y value for each x coordinate
    for x in xAxis:
        y_n = y_prev + deriv_prev * h
        px = x + h / 2
        py = y_n + deriv(x, y_n) * (h / 2)
        deriv_prev = deriv(px, + py)
        y_vec.append(y_n)
        y_prev = y_n
    return xAxis, y_vec

def analythic_sol(x_val):
    h = 0.2
    x_0 = 1
    xAxis = np.arange(x_0, x_val, h)
    y_vec = []
    for x in xAxis:
        y_vec.append(x * np.log(x) + x)
    return xAxis, y_vec

def calc_error(y_pred_val, y_true):
    return np.abs(y_true - y_pred_val)

def plot_all(numeric_1X, numeric_1Y, numeric_2X, numeric_2Y,
             numeric_improve_1X, numeric_improve_1Y, numeric_improve_2X, numeric_improve_2Y ):
    """
    Plots all the graphs together.
    :param analythic_X: analythic solution x vector
    :param analythic_Y: analythic solution y vector
    :param numeric_1X: numeric solution x vector for h = 0.01
    :param numeric_1Y: numeric solution y vector for h = 0.01
    :param numeric_2X: numeric solution x vector for h = 0.2
    :param numeric_2Y: numeric solution y vector for h = 0.2
    :param numeric_improve_1X: improved (polygon) numeric solution x vector for h = 0.01
    :param numeric_improve_1Y: improved (polygon) numeric solution y vector for h = 0.01
    :param numeric_improve_2X: improved (polygon) numeric solution x vector for h = 0.2
    :param numeric_improve_2Y: improved (polygon) numeric solution x vector for h = 0.2
    :return: f(3) calculated by each method (idx0, ..., idx4)
    """
    fig, ax = plt.subplots(1,5, sharey=True)
    ax[0].plot(numeric_1X, numeric_1Y, c='r', linestyle = 'dashed')
    ax[0].set_title('numeric solution, h = 0.01', size=7)
    ax[0].set_xlabel('x', size=7)
    ax[0].set_ylabel('y', size=7)
    idx0 = np.where(abs(numeric_1X - 3.0) < 0.000001)[0][0]
    pt0 = (numeric_1X[idx0], numeric_1Y[idx0])
    ax[0].annotate('(%0.1f,%0.4f)' %pt0, xy=pt0)
    print(idx0)
    ax[0].plot(numeric_1X, numeric_1Y, markevery=[idx0], ls="", marker="o", label="points")

    ax[1].plot(numeric_2X, numeric_2Y, c='m',linestyle = 'dashed')
    ax[1].set_title('numeric solution, h = 0.2', size=7)
    ax[1].set_xlabel('x', size=7)
    ax[1].set_ylabel('y', size=7)
    idx1 = np.where(abs(numeric_2X - 3.0) < 0.000001)[0][0]
    pt1 = (numeric_2X[idx1], numeric_2Y[idx1])
    ax[1].annotate('(%0.1f,%0.4f)' % pt1, xy=pt1)
    ax[1].plot(numeric_2X, numeric_2Y, markevery=[idx1], ls="", marker="o", label="points")

    ax[2].plot(numeric_improve_1X, numeric_improve_1Y, c='g', linestyle = 'dotted')
    ax[2].set_title('improved numeric solution, h = 0.01',  size=7)
    ax[2].set_xlabel('x', size=7)
    ax[2].set_ylabel('y', size=7)
    idx2 = np.where(abs(numeric_improve_1X - 3.0) < 0.000001)[0][0]
    pt2 = (numeric_improve_1X[idx2], numeric_improve_1Y[idx2])
    ax[2].annotate('(%0.1f,%0.5f)' % pt2, xy=pt2)
    ax[2].plot(numeric_improve_1X, numeric_improve_1Y, markevery=[idx2], ls="", marker="o", label="points")


    ax[3].plot(numeric_improve_2X, numeric_improve_2Y, c='c', linestyle = 'dotted')
    ax[3].set_title('improved numeric solution, h = 0.2', size=7)
    ax[3].set_xlabel('x', size=7)
    ax[3].set_ylabel('y', size=7)
    idx3 = np.where(abs(numeric_improve_2X - 3.0) < 0.000001)[0][0]
    pt3 = (numeric_improve_2X[idx3], numeric_improve_2Y[idx3])
    ax[3].annotate('%0.1f,%0.5f)' % pt3, xy=pt3)
    ax[3].plot(numeric_improve_2X, numeric_improve_2Y, markevery=[idx3], ls="", marker="o", label="points")

    ax[4].plot(analythic_X, analythic_Y, c='k')
    ax[4].set_title('analythic solution', size=7)
    ax[4].set_xlabel('x', size=7)
    ax[4].set_ylabel('y', size=7)
    idx4 = np.where(abs(analythic_X - 3.0) < 0.000001)[0][0]
    pt4 = (analythic_X[idx4], analythic_Y[idx4])
    ax[4].annotate('(%0.1f,%0.5f)' % pt4, xy=pt4)
    ax[4].plot(analythic_X, analythic_Y, markevery=[idx4], ls="", marker="o", label="points")
    plt.show()
    return idx0, idx1, idx2, idx3, idx4

def plot_all_together(analythic_X, analythic_Y, numeric_1X, numeric_1Y, numeric_2X, numeric_2Y,
             numeric_improve_1X, numeric_improve_1Y, numeric_improve_2X, numeric_improve_2Y,
                      idx0, idx1, idx2, idx3, idx4):
    """
    Plots all the graphs together.
    :param analythic_X: analythic solution x vector
    :param analythic_Y: analythic solution y vector
    :param numeric_1X: numeric solution x vector for h = 0.01
    :param numeric_1Y: numeric solution y vector for h = 0.01
    :param numeric_2X: numeric solution x vector for h = 0.2
    :param numeric_2Y: numeric solution y vector for h = 0.2
    :param numeric_improve_1X: improved (polygon) numeric solution x vector for h = 0.01
    :param numeric_improve_1Y: improved (polygon) numeric solution y vector for h = 0.01
    :param numeric_improve_2X: improved (polygon) numeric solution x vector for h = 0.2
    :param numeric_improve_2Y: improved (polygon) numeric solution x vector for h = 0.2
    :param idx0: f(3) calculated by euler numeric method, h = 0.01
    :param idx1: f(3) calculated by euler numeric method, h = 0.2
    :param idx2: f(3) calculated by improved polygon numeric method, h = 0.01
    :param idx3: f(3) calculated by improved polygon numeric method, h = 0.2
    :param idx4: analythic f(3) value
    :return: None, only plots
    """
    plt.plot(numeric_1X, numeric_1Y, c='r', linestyle='--', label='euler, h = 0.01' )
    plt.plot(numeric_2X, numeric_2Y, c='m', linestyle='--', label='euler, h = 0.2')
    plt.plot(numeric_improve_1X, numeric_improve_1Y, c = 'b', linestyle='dotted', label='improved polygon, h = 0.01')
    plt.plot(numeric_improve_2X, numeric_improve_2Y, c = 'g', linestyle='dotted', label='improved polygon, h = 0.2')
    plt.plot(analythic_X, analythic_Y, c = 'k', label='analythic solution')

    plt.plot(numeric_1X, numeric_1Y, markevery=[idx0], ls="",c='r', marker="o", label=f'{3, numeric_1Y[idx0]}')
    plt.plot(numeric_2X, numeric_2Y, markevery=[idx1], ls="", c='m', marker="o", label=f'{3, numeric_2Y[idx1]}')
    plt.plot(numeric_improve_1X, numeric_improve_1Y, markevery=[idx2], ls="", c='b', marker="o", label=f'{3, numeric_improve_1Y[idx2]}')
    plt.plot(numeric_improve_2X, numeric_improve_2Y, markevery=[idx3], ls="", c='g', marker="o", label=f'{3, numeric_improve_2Y[idx3]}')
    plt.plot(analythic_X, analythic_Y, markevery=[idx4], ls="", marker="o", c='k',label=f'{3, analythic_Y[idx4]}')

    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Polygon method
    numeric_1X, numeric_1Y = numeric(x_val=3.01, h=0.01)
    numeric_2X, numeric_2Y = numeric(x_val=3.2, h=0.2)
    # Improved polygon method
    numeric_improve_1X, numeric_improve_1Y = numeric_improve(x_val=3.01, h=0.01)
    numeric_improve_2X, numeric_improve_2Y = numeric_improve(x_val=3.2, h=0.2)
    # Analythic solution
    analythic_X, analythic_Y = analythic_sol(x_val=3.05)

    # Plot everything, and save the y values we got
    idx0, idx1, idx2, idx3, idx4 = plot_all(numeric_1X, numeric_1Y, numeric_2X, numeric_2Y,
             numeric_improve_1X, numeric_improve_1Y, numeric_improve_2X, numeric_improve_2Y)


    true_val = analythic_Y[idx4]
    err_poly1 = calc_error(numeric_1Y[idx0], true_val)
    err_poly2 = calc_error(numeric_2Y[idx1], true_val)
    err_improved_poly1 = calc_error(numeric_improve_1Y[idx2], true_val)
    err_improved_poly2 = calc_error(numeric_improve_2Y[idx3], true_val)

    # create data
    data = [["Polygon method", "0.01",numeric_1Y[idx0] ,err_poly1],
            ["Polygon method", "0.2",numeric_2Y[idx1], err_poly2],
            ["Imprved polygon method", "0.01",numeric_improve_1Y[idx2], err_improved_poly1],
            ["Imprved polygon method", "0.2",numeric_improve_2Y[idx3], err_improved_poly2],
            ["Analythic solution", "not relevant",f"Calculated analytically - {true_val}", calc_error(true_val, true_val)]]
    # define header names
    col_names = ["Method", "h", "Numeric calculated y", "Error"]
    # display table
    print(tabulate(data, headers=col_names))

    plot_all_together(analythic_X, analythic_Y, numeric_1X, numeric_1Y, numeric_2X, numeric_2Y,
             numeric_improve_1X, numeric_improve_1Y, numeric_improve_2X, numeric_improve_2Y,
                      idx0, idx1, idx2, idx3, idx4)

