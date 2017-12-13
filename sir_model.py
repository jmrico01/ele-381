import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

import read_data

# optimize beta, gamma for both SIR SIS/D, give loss
# graphs:
#   model vs real data

# default:
# heat

def PrintProgress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()

def GetDeltasSIR(S, I, R, beta, gamma):
    # Takes in coefficients and the S, I, R of the previous time step
    # Returns deltas: [dS, dI, dR, dOut]
    dS = -beta * S * I
    dI = beta * S * I - gamma * I
    dR = gamma * I
    # we're only recording the number of new cases of I
    dOut = beta * S * I

    return [dS, dI, dR, dOut]

def RunSIR(beta, gamma, startS, startI, startR, T):
    S = np.empty((len(beta), T))
    I = np.empty((len(beta), T))
    R = np.empty((len(beta), T))
    out = np.empty((len(beta), T))
    S[:, 0] = startS
    I[:, 0] = startI
    R[:, 0] = startR
    out[:, 0] = 0
    for t in range(1, T):
        # This will change depending on the model
        [dS, dI, dR, dOut] = GetDeltasSIR(S[:, t-1], I[:, t-1], R[:, t-1], beta, gamma)
        S[:, t]   = S[:, t-1]   + dS
        I[:, t]   = I[:, t-1]   + dI
        R[:, t]   = R[:, t-1]   + dR
        out[:, t] = out[:, t-1] + dOut
    
    return [S, I, R, out]

def CalcError(model, data):
    xModel = np.linspace(0.0, 1.0, len(model))
    xData = np.linspace(0.0, 1.0, len(data))
    modelResample = np.interp(xData, xModel, model)
    assert(len(modelResample) == len(data))

    return np.linalg.norm(np.divide(data - modelResample) / data)

# -- Import data
dataFilePath = "ebola_data_db_format.csv"
countries = [
    "Sierra Leone",
    "Guinea",
    "Liberia"
]
categories = [
    read_data.CAT_CONF
]
[data, n, dateStart, dateEnd] = read_data.ReadData(dataFilePath,
    countries, categories, True, 5, False)
print("----- Data Imported -----")
print("Data points: " + str(n))
print("Start date:  " + str(dateStart))
print("End date:    " + str(dateEnd))
print()

def TestSIR():
    T = 100000

    startS = 7e6
    startI = 200
    startR = 0
    beta = [5e-9]
    #beta=3.2e-11
    betaRange = 0.5
    r0Min = 1.0
    r0Max = 1.0 + 5e-12
    r0 = r0Max
    gamma = [beta * startS / r0]
    print(gamma)

    [S, I, R, output] = RunSIR(startS, startI, startR, beta, gamma, T)

def InterpFunc(a, b, i, n):
    exp = 0.5
    return a + (b - a) * ((float(i) / float(n))**exp)

def Optimize():
    print("----- Optimizing SIR Parameters -----")
    T = 10000
    startS = 7e6 # population of Sierra Leone
    startI = 200 # TODO guess this in a better way
    startR = 0
    betaMin = 5e-12
    betaMax = 5e-8
    betaIters = 1000
    gammaMin = 0.02
    gammaMax = 0.04
    gammaIters = 1000

    minError = sys.float_info.max
    minParams = [-1, -1]
    beta = np.linspace(betaMin, betaMax, betaIters)
    gamma = np.linspace(gammaMin, gammaMax, gammaIters)
    [_, _, _, output] = RunSIR(beta, gamma, startS, startI, startR, T)

    """
    for i in range(betaIters):
        beta = InterpFunc(betaMin, betaMax, i, betaIters)
        for j in range(gammaIters):
            PrintProgress(i * gammaIters + j, betaIters * gammaIters - 1)
            gamma = InterpFunc(gammaMin, gammaMax, i, gammaIters)
            [_, _, _, output] = RunSIR(startS, startI, startR, beta, gamma, T)
            error = CalcError(output, data["Sierra Leone"])
            if error < minError:
                error = minError
                minParams = [beta, gamma]
    """
    
    print("-- Done --")
    print("beta:  " + str(minParams[0]))
    print("gamma: " + str(minParams[1]))
    print("error:  " + str(error))

Optimize()

exit()
fig, ax = plt.subplots()
plt.subplots_adjust(left = 0.35, bottom = 0.35)
t = np.linspace(0.0, 1.0, num = T, endpoint = True)
#lineS, = plt.plot(t, S, lw=2.0, color='orange', label="Susceptible")
#lineI, = plt.plot(t, I, lw=2.0, color='red', label="Infected")
#lineR, = plt.plot(t, R, lw=2.0, color='blue', label="Recovered")
#plt.xlabel("Time (normalized 0 to 1)")
#plt.ylabel("Number of people")
#plt.legend(handles=[lineS, lineI, lineR])
lineOut, = plt.plot(t, output, color="red")
#plt.axis([0, 1, -10, 10])

axisColor = 'lightgoldenrodyellow'
axisBeta = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axisColor)
axisR0 = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axisColor)
sliderBeta = Slider(axisBeta, 'beta',
    beta * (1.0 - betaRange), beta * (1.0 + betaRange),
    valinit = beta)
sliderR0 = Slider(axisR0, 'r0',
    r0Min, r0Max,
    valinit = r0)

def UpdateSIR(val):
    beta = sliderBeta.val
    r0 = sliderR0.val
    gamma = beta * startS / r0
    [S, I, R, output] = RunSIR(startS, startI, startR, beta, gamma, T)
    #lineS.set_ydata(S)
    #lineI.set_ydata(I)
    #lineR.set_ydata(R)
    lineOut.set_ydata(output)
    fig.canvas.draw_idle()

sliderBeta.on_changed(UpdateSIR)
sliderR0.on_changed(UpdateSIR)

axisPrint = plt.axes([0.8, 0.025, 0.1, 0.04])
buttonError = Button(axisPrint, 'Calc Error', color=axisColor, hovercolor='0.975')

def PlotCalcError(event):
    beta = sliderBeta.val
    r0 = sliderR0.val
    gamma = beta * startS / r0
    [_, _, _, output] = RunSIR(startS, startI, startR, beta, gamma, T)
    print("Error: " + str(CalcError(output, data["Sierra Leone"])))
buttonError.on_clicked(PlotCalcError)

plt.show()