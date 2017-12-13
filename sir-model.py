import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

import read-data

def GetDeltasSIR(S, I, R, beta, gamma):
    # Takes in coefficients and the S, I, R of the previous time step
    # Returns deltas: [dS, dI, dR, dOut]
    dS = -beta * S * I
    dI = beta * S * I - gamma * I
    dR = gamma * I
    # we're only recording the number of new cases of I
    dOut = beta * S * I

    return [dS, dI, dR, dOut]

def RunSIR(startS, startI, startR, beta, gamma, T):
    S = [0] * T
    I = [0] * T
    R = [0] * T
    S[0] = startS
    I[0] = startI
    R[0] = startR
    out = [0] * T
    for t in range(1, T):
        # This will change depending on the model
        [dS, dI, dR, dOut] = GetDeltasSIR(S[t-1], I[t-1], R[t-1], beta, gamma)

        S[t]    = S[t-1] + dS
        I[t]    = I[t-1] + dI
        R[t]    = R[t-1] + dR
        out[t]  = out[t-1] + dOut
    
    return [S, I, R, out]

read-data.ReadData("ebola_data_db_format.csv",
        ["Sierra Leone"], [CAT_CONF], True, 5, True)

T = 100000

startS = 12e6#7e6
startI = 1800#935
startR = 0
#beta = 3.2e-10
beta=3.2e-11
betaRange = 0.5
r0Min = 1.0
r0Max = 4.0
r0 = 3.0
gamma = beta * startS / r0

[S, I, R, output] = RunSIR(startS, startI, startR, beta, gamma, T)

fig, ax = plt.subplots()
plt.subplots_adjust(left = 0.35, bottom = 0.35)
t = np.linspace(0.0, 1.0, num = T, endpoint = True)
lineS, = plt.plot(t, S, lw=2.0, color='orange', label="Susceptible")
lineI, = plt.plot(t, I, lw=2.0, color='red', label="Infected")
lineR, = plt.plot(t, R, lw=2.0, color='blue', label="Recovered")
plt.xlabel("Time (normalized 0 to 1)")
plt.ylabel("Number of people")
plt.legend(handles=[lineS, lineI, lineR])
#lineOut, = plt.plot(t, output, color="red")
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
    lineS.set_ydata(S)
    lineI.set_ydata(I)
    lineR.set_ydata(R)
    #lineOut.set_ydata(output)
    fig.canvas.draw_idle()

sliderBeta.on_changed(UpdateSIR)
sliderR0.on_changed(UpdateSIR)

axisPrint = plt.axes([0.8, 0.025, 0.1, 0.04])
buttonPrint = Button(axisPrint, 'Print', color=axisColor, hovercolor='0.975')

def PrintValues(event):
    print("beta: " + str(sliderBeta.val))
    print("r0: " + str(sliderR0.val))
buttonPrint.on_clicked(PrintValues)

plt.show()