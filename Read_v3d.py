import matplotlib.pyplot as plt
import numpy as np
from BiomechTools import low_pass, zero_crossing, max_min, simpson_nonuniform, critically_damped
#from scipy import interpolate

max_pts = np.zeros(29, dtype=np.int32)      # used to record the peak positive angular velocity of mla
min_pts = np.zeros(29, dtype=np.int32)      # used to record the peak negative angular velocity of mla
zero_pts = np.zeros(29, dtype=np.int32)     # used to find end of loading phase for mla

fn = 'D:/Alexis_Small_Files/Alexis_Data.txt'
fn2 = 'D:/Alfredo/B002/B002.txt'
with open(fn2) as infile:
    first_line = infile.readline()
    var_line = infile.readline()
    var_names = var_line.split('\t')            # 2nd line contains variable names
    for line in range(3):                       # the last line contains X, Y Z header info
        temp = infile.readline()
    header = temp.split('\t')
   # self.n = int(header[7]) - 2
  #  self.sampling_rate = int(header[8])
data = np.genfromtxt(fn2, delimiter='\t', skip_header=7)
n_rows = data.shape[0]                          # number of rows of array
n_cols = data.shape[1]                          # number of columns of array
print('n steps: ', n_cols)
#omega = np.zeros((n_rows, (n_cols+1)/2))
omega = np.zeros((99, 29))
t = data[:, 0]
fz = data[:, 1:30] * -1.0
mla = data[:, 31:59]
plt.plot(t, mla)
plt.show()

def get_max(curve, first_pt, last_pt):
    max_location = first_pt
    max_val = curve[first_pt]
    for i in range(first_pt, last_pt):
        if curve[i] > max_val:
            max_location = i
            max_val = curve[i]
    return max_location

def get_min(curve, first_pt, last_pt):
    min_location = first_pt
    min_val = curve[first_pt]
    for i in range(first_pt, last_pt):
        if curve[i] < min_val:
            min_location = i
            min_val = curve[i]
    return min_location


def compute_angular_velocity():
    for col in range(0, 28):
        for row in range(1, 98):
            omega[row][col] = (mla[row+1][col] - mla[row-1][col]) / (2.0 * 0.001)
        omega[0, col] = omega[1, col]
        omega[98, col] = omega[97, col]
        #omega[99, col] = omega[97, col]
    plt.plot(omega)
    plt.show()

def plot_force_and_mla():
    fig, ax1 = plt.subplots()                           # Creating plot with dataset_1

    color = 'tab:red'
    ax1.set_xlabel('Stance Time (%)')
    ax1.set_ylabel('Fz (N)', color=color)
    ax1.plot(t, fz, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    color = 'tab:blue'
    ax2 = ax1.twinx()                                   # Adding Twin Axes to plot using dataset_2    color = 'tab:green'
    ax2.set_ylabel('MLA (d)', color=color)
    ax2.plot(t, mla, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    plt.show()


compute_angular_velocity()

for col in range(28):
    zero_point = ()
    min_pts[col] = get_max(omega[:, col], 3, 40)
    zp = zero_crossing(omega[:, col], 0, 3, 40)
    zp = str(zp[0])
    zero_pts[col] = int(zp)                    #int(''.join(map(str, zero_point)))             # ignore the rising and falling

plot_force_and_mla()
#plt.plot(t, fz)
#plt.show()
