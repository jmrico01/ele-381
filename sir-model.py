import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

# SIR model
#   dS/dt = -beta S I
#   dI/dt = beta S I - gamma I
#   dR/dt = gamma I
def Step(S, I, R, beta, gamma):
    dS = -beta * S * I
    dI = beta * S * I - gamma * I
    dR = gamma * I
    return (S + dS, I + dI, R + dR)

def RunSIR(startS, startI, startR, beta, gamma, T):
    S = [startS] * T
    I = [startI] * T
    R = [startR] * T
    for t in range(1, T):
        S[t], I[t], R[t] = Step(S[t-1], I[t-1], R[t-1], beta, gamma)
    
    return S, I, R

T = 10000

startS = 12e6
startI = 1800
startR = 0
beta = 3.2e-10
betaRange = 0.5
r0Min = 1.0
r0Max = 4.0
r0 = (r0Min + r0Max) / 2.0
gamma = beta * startS / r0

S, I, R = RunSIR(startS, startI, startR, beta, gamma, T)

fig, ax = plt.subplots()
plt.subplots_adjust(left = 0.25, bottom = 0.25)
t = np.linspace(0.0, 1.0, num = T, endpoint = True)
lineS, = plt.plot(t, S, lw=1.5, color='yellow')
lineI, = plt.plot(t, I, lw=1.5, color='red')
lineR, = plt.plot(t, R, lw=1.5, color='blue')
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
    S, I, R = RunSIR(startS, startI, startR, beta, gamma, T)
    lineS.set_ydata(S)
    lineI.set_ydata(I)
    lineR.set_ydata(R)
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