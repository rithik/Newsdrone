import math
import numpy as np
import matplotlib.pyplot as plot
from scipy.stats import norm
import random

x0 = 1000
a = 24693
c = 3967
K = 2**17

curr_rand = 0

def random_num_generator(i, x):
    if i == 0:
        return x / K
    else: 
        x = (a * x + c) % K 
    return random_num_generator(i-1, x)

def random_num_generator_iter(i):
    count = i
    x = x0
    while count >= 0:
        if count == 0:
            return x / K
        else:
            x = (a * x + c) % K 
        count -= 1

def simulate_one_run(iteration_num):
    prob = random_num_generator_iter(iteration_num)
    x = 57 * math.sqrt(2) * math.sqrt(math.log(1/(1-prob)))
    return x, prob
    
def sample_mean_n(num, itera):
    global curr_rand
    arr = []
    for i in range(0,num):
        x, prob = simulate_one_run(curr_rand)
        # x,prob = simulate_one_run(random.randint(0,10000))
        arr.append(x)
        curr_rand = (int) ((prob * itera * i * 23) % 10000) 
        # print(curr_rand)
    return (arr, sum(arr)/num)

def big_m_n(n):
    arr = []
    for i in range(1,111):
        # print(i)
        arr.append(sample_mean_n(n, i)[1])

    return (arr, avg)

def count_percent(array):
    count = 0
    for i in array:
        if i > 61 and i < 81:
            count+=1
    return count/len(array) * 100

def small_z_n(n, mean, stddev):
    z_n_arr = []
    arr, avg = big_m_n(n)
    for k in arr:
        z_n_arr.append((k - mean)/(stddev / math.sqrt(n)))
    return z_n_arr

def find_difference(arr):
    cdf_values = [(1-.9192), (1-.8413), (1-.6915), .5, .6915, .8413, .9212]
    diffs = []
    max_diff = 0
    max_index = None
    for k in range(0, len(arr)):
        # print(arr[k])
        diffs.append(abs(cdf_values[k] - arr[k]))
        if diffs[k] > max_diff:
            max_diff = diffs[k]
            max_index = k
        # max_diff = max(max_diff, diffs[k])
    return diffs, max_diff, max_index
    
def less_than_z(arr):
    z_values = [-1.4, -1.0, -0.5, 0, 0.5, 1.0, 1.4]
    counts = []
    for i in z_values:
        count = 0
        for j in arr:
            if j <= i:
                count +=1 
        counts.append(count/110)
    return counts

def calculate_all_small_z_n():
    all_values = [10, 30, 50, 100, 150, 250, 500, 1000]
    # all_values = [500]
    z_values = [-1.4, -1.0, -0.5, 0, 0.5, 1.0, 1.4]
    mean = 57 * math.sqrt(math.pi/2)
    stddev = math.sqrt((4 - math.pi) / (2 * (1/57)**2))
    norm_x = np.array(np.arange(-2.5, 2.5, 0.01))
    norm_y = [norm.cdf(xa) for xa in norm_x]

    for val in all_values:
        plot.clf()
        data_val = small_z_n(val,mean,stddev)
        cdf_prob_array = less_than_z(data_val)
        print(cdf_prob_array)
        diff_array, MAD, max_idx = find_difference(cdf_prob_array)
        print(val, diff_array, MAD)
        for k in range(0, len(z_values)):
            plot.plot(z_values[k], cdf_prob_array[k], "bo")
     
        plot.vlines(z_values[max_idx], 0, 1)
        # Create the plot
        plot.plot(norm_x, norm_y) 
        plot.xlabel("Z value")
        plot.ylabel("Cumulative Probability")
        plot.title("n = {}".format(val))
        
        plot.savefig('graph{}.png'.format(val))

def calculate_all_n():
    full_arr = []
    xvalues = [10] * 110 + [30] * 110 + [50] * 110 + [100] * 110 + [150] * 110 + [250] * 110 + [500] * 110 + [1000] * 110
    yvalues = big_m_n(10)[0] + big_m_n(30)[0] + big_m_n(50)[0] + big_m_n(100)[0] + big_m_n(150)[0] + big_m_n(250)[0] + big_m_n(500)[0] + big_m_n(1000)[0]
    xs = np.linspace(1,1000,200)
    horiz_line_data = np.array([71.34 for i in range(len(xs))])
    plot.plot(xs, horiz_line_data, 'r--') 

    plot.plot(xvalues, yvalues, "o")
    plot.xlabel("Sample size (n)")
    plot.ylabel("Sample Mean (m_n)")
    plot.title("Sample Mean vs Sample Size - Simulation of Newsdrone Drop Deviation")
    plot.savefig('newsdronegraph.png')

def simulate_n_times(n):
    w_times = []
    rand_nums_generated = []

    for k in range(0, n):
        w_times.append(simulate_one_run(k))

    return w_times
    

# collect_data(1000)
# print(random_num_generator_iter(1))
# print(random_num_generator_iter(2))
# print(random_num_generator_iter(3))
# print(random_num_generator_iter(51))
# print(random_num_generator_iter(52))
# print(random_num_generator_iter(53))
# print(simulate_one_run(200))


# print("{}: {}".format(10, sample_mean_n(10)))
# print("{}: {}".format(30, sample_mean_n(30)))
# print("{}: {}".format(50, sample_mean_n(50)))
# print("{}: {}".format(100, sample_mean_n(100)))
# print("{}: {}".format(150, sample_mean_n(150)))
# print("{}: {}".format(250, sample_mean_n(250)))
# print("{}: {}".format(500, sample_mean_n(500)))
# print("{}: {}".format(1000, sample_mean_n(1000, 1)))

# calculate_all_n()
calculate_all_small_z_n()
