#!/usr/bin/env python
# coding: utf-8


#start_scope()
from brian2 import *
import matplotlib.pyplot as plt
import numpy as np

map_size = 100
global foodx, foody, food_count, bug_plot, food_plot, sr_plot, sl_plot,outbugx,outbugy,outbugang,outfoodx,outfoody,outsrx,outsry,outslx,outsly, outaggx,outaggy,outsraggx,outsraggy,outslaggx,outslaggy

duration=4000

food_count = 0
foodx=25
foody=25
outbugx=np.zeros(int(duration/2))
outbugy=np.zeros(int(duration/2))
outbugang=np.zeros(int(duration/2))
outfoodx=np.zeros(int(duration/2))
outfoody=np.zeros(int(duration/2))
outsrx=np.zeros(int(duration/2))
outsry=np.zeros(int(duration/2))
outslx=np.zeros(int(duration/2))
outsly=np.zeros(int(duration/2))

food_a_count = 0
foodx2=-50
foody2=-50
outaggx = np.zeros(int(duration / 2))
outaggy = np.zeros(int(duration / 2))
outaggsrx = np.zeros(int(duration / 2))
outaggsry = np.zeros(int(duration / 2))
outaggslx = np.zeros(int(duration / 2))
outaggsly = np.zeros(int(duration / 2))

A1 = TimedArray((0,1),dt=2000*ms)
A2 = TimedArray((1,0),dt=2000*ms)

# Sensor neurons
#Izhikevich
a = 0.02
b = 0.2
c = -65
d = 0.5

I01=70
I02=70
tau_a = 1 *ms
g_peak = 0.4
g_synmaxv = g_peak / (tau_a*exp(-1)) * ms
E_syn = 10

tau_a_a = 1 *ms
g_peak_a = 0.4
g_synmaxv_a = g_peak_a / (tau_a_a*exp(-1)) * ms
E_syn = 10


sensor_eqs = '''
# equations for neurons
dv/dt = (0.04*v**2+5*v+140-u+r1*I+plasticity*(E_syn-v))/ms: 1
du/dt=a*(b*v-u)/ms: 1
plasticity = (foodcount+2)*0.1*g_a_syn : 1
I = I01 / sqrt(((x-foodxx)**2+(y-foodyy)**2)): 1
a:1
d:1
x : 1
y : 1
x_disp : 1
y_disp : 1
foodxx : 1
foodyy : 1
foodcount : 1
mag :1
r1:1
dz/dt = (-z/tau_a) : 1
dg_a/dt = (-g_a/tau_a) + z/ms: 1
g_synmax:1
g_a_syn :1
'''

sensor_agg_eqs = '''
# equations for neurons
dv/dt = (0.04*v**2+5*v+140-u+r1*I+plasticity1*(E_syn-v))/ms: 1
du/dt=a*(b*v-u)/ms: 1
plasticity1 = (foodcount1+2)*0.1*g_a_syn_a : 1
I = I02 / sqrt(((x-foodxx2)**2+(y-foodyy2)**2)): 1
a:1
d:1
x : 1
y : 1
x_disp : 1
y_disp : 1
foodxx2 : 1
foodyy2 : 1
foodcount1 : 1
mag :1
r1:1
dz_a/dt = (-z_a/tau_a_a) : 1
dg_a_a/dt = (-g_a_a/tau_a_a) + z_a/ms: 1
g_synmax_a:1
g_a_syn_a :1
'''

num_neurons=2

#Winner Take All Circuit with Naka Rushton Rate Neurons

eqs1 = '''
xn=(A1(t)*120-Iconct):1
rnakarush=int(xn>0)*((100.0*(xn)**2)/(120**2 + (xn)**2)):1
dr/dt = (-r + (rnakarush))/(taunr): 1
taunr:second
Iconct:1
I2:1
'''
eqs2 = '''
xn=(A2(t)*120-Iconct):1
rnakarush=int(xn>0)*((100.0*(xn)**2)/(120**2 + (xn)**2)):1
dr/dt = (-r + (rnakarush))/(taunr): 1
taunr:second
Iconct:1
I2:1
'''

#reset='r=49'

# Threshold and refractoriness are only used for spike counting
group1 = NeuronGroup(1, eqs1,clock=Clock(0.2*ms), threshold='r<=5', reset = 'r=1', method='euler')
group1.taunr=0.2*ms
group1.r=20

group2 = NeuronGroup(1, eqs2,clock=Clock(0.2*ms), threshold='r>=49', reset = 'r=49', method='euler')
group2.taunr=0.2*ms
group2.r=20


sensor_reset = '''
v = c
u = u + d
'''


# Sensor neurons
# right sensor 1
sr = NeuronGroup(1, sensor_eqs, clock=Clock(0.2*ms), threshold = "v>=30", reset = sensor_reset,method='euler')
sr.v = c
sr.u = c*b
sr.x_disp = 5
sr.y_disp = 5
sr.x = sr.x_disp
sr.y = sr.y_disp
sr.foodxx = foodx
sr.foodyy = foody
sr.foodcount = food_count
sr.mag=1

# left sensor 1
sl = NeuronGroup(1, sensor_eqs, clock=Clock(0.2*ms), threshold = "v>=30", reset = sensor_reset,method='euler')
sl.v = c
sl.u = c*b
sl.x_disp = -5
sl.y_disp = 5
sl.x = sl.x_disp
sl.y = sl.y_disp
sl.foodxx = foodx
sl.foodyy = foody
sl.foodcount = food_count
sl.mag=1

# right sensor 2
sr2 = NeuronGroup(1, sensor_agg_eqs, clock=Clock(0.2 * ms), threshold="v>=30", reset=sensor_reset, method='euler')
sr2.v = c
sr2.u = c * b
sr2.x_disp = 7
sr2.y_disp = 7
sr2.x = sr2.x_disp
sr2.y = sr2.y_disp
sr2.foodxx2 = foodx2
sr2.foodyy2 = foody2
sr2.foodcount1 = food_a_count
sr2.mag = 1

# left sensor 2
sl2 = NeuronGroup(1, sensor_agg_eqs, clock=Clock(0.2 * ms), threshold="v>=30", reset=sensor_reset, method='euler')
sl2.v = c
sl2.u = c * b
sl2.x_disp = -7
sl2.y_disp = 7
sl2.x = sl2.x_disp
sl2.y = sl2.y_disp
sl2.foodxx2 = foodx2
sl2.foodyy2 = foody2
sl2.foodcount1 = food_a_count
sl2.mag = 1

 
# Motor neurons
# right bug motor neuron
sbr = NeuronGroup(1, sensor_eqs, clock=Clock(0.2*ms), threshold = "v>=30", reset = sensor_reset,method='euler')
sbr.v = c
sbr.u = c*b
sbr.foodxx = foodx
sbr.foodyy = foody
sbr.foodcount = food_count
sbr.mag=0

# left bug motor neuron
sbl = NeuronGroup(1, sensor_eqs, clock=Clock(0.2*ms), threshold = "v>=30", reset = sensor_reset,method='euler')
sbl.v = c
sbl.u = c*b
sbl.foodxx = foodx
sbl.foodyy = foody
sbl.foodcount = food_count
sbl.mag=0


# right bug motor neuron 2
sbr2 = NeuronGroup(1, sensor_agg_eqs, clock=Clock(0.2 * ms), threshold="v>=30", reset=sensor_reset, method='euler')
sbr2.v = c
sbr2.u = c * b
sbr2.foodxx2 = foodx2
sbr2.foodyy2 = foody2
sbr2.foodcount1 = food_a_count
sbr2.mag = 0

# left bug motor neuron 2
sbl2 = NeuronGroup(1, sensor_agg_eqs, clock=Clock(0.2 * ms), threshold="v>=30", reset=sensor_reset, method='euler')
sbl2.v = c
sbl2.u = c * b
sbl2.foodxx2 = foodx2
sbl2.foodyy2 = foody2
sbl2.foodcount1 = food_a_count
sbl2.mag = 0


# The virtual bug
# What are the bug equations
# Equations for velovity in the notes

tau_motor = 2.5 *ms # mine 1*ms #0.25
base_speed = 1.5 # So it moves without activity
L = 35*Hz  #Decreasing will make a slower turning when it senses target
alpha = 0.05 #.2 # 0.25


bug_eqs = '''
#equations for movement here
dmotorr/dt = ((-motorr/tau_motor)) :1
dmotorl/dt = ((-motorl/tau_motor)) :1
vel = (motorr+motorl)/2 + base_speed :1
dangle/dt = ((motorr-motorl)*L) :1
dx/dt = (alpha*vel*cos(angle))/ms :1
dy/dt = (alpha*vel*sin(angle))/ms :1
'''
#These are the equation  for the motor and speed

bug = NeuronGroup(1, bug_eqs, clock=Clock(0.2*ms),method='euler')
bug.motorr = 0
bug.motorl = 0
bug.angle = pi/2
bug.x = 0
bug.y = 0

# Synapses (sensors communicate with bug motor)
C44 = Synapses(group1,group2,clock=Clock(0.2*ms),model='''
                               w : 1
                               Iconct_pre = (w) * (r_post) :1 (summed)  ''')
C44.connect(i=[0], j=[0])  
C44.w=3.0

C45 = Synapses(group2,group1,clock=Clock(0.2*ms),model='''
                               w : 1
                               Iconct_pre = (w) * (r_post) :1 (summed)  ''')
C45.connect(i=[0], j=[0])  
C45.w=3.0

syn1 = Synapses(group1, sr, clock=Clock(0.2*ms), model='''
                g_a_syn_post = g_a: 1 (summed)
                r1_post = r_pre :1 (summed)
                ''',
		on_pre='''
		z+= g_synmax
		''')
syn1.connect(i=[0],j=[0])
syn1.g_synmax=g_synmaxv

syn2 = Synapses(group1, sl, clock=Clock(0.2*ms), model='''
                g_a_syn_post = g_a: 1 (summed)
                r1_post = r_pre :1 (summed)
                ''',
		on_pre='''
		z+= g_synmax
		''')
syn2.connect(i=[0],j=[0])
syn2.g_synmax=g_synmaxv

syn3 = Synapses(group2, sl2, clock=Clock(0.2*ms), model='''
                g_a_syn_a_post = g_a_a: 1 (summed)
                r1_post = r_pre :1 (summed)
                ''',
		on_pre='''
		z_a+= g_synmax_a
		''')
syn3.connect(i=[0],j=[0])
syn3.g_synmax_a=g_synmaxv_a

syn4 = Synapses(group2, sr2, clock=Clock(0.2*ms), model='''
                g_a_syn_a_post = g_a_a: 1 (summed)
                r1_post = r_pre :1 (summed)
                ''',
		on_pre='''
		z_a+= g_synmax_a
		''')
syn4.connect(i=[0],j=[0])
syn4.g_synmax_a=g_synmaxv_a

we = 10
#Sensor 1
syn_rr=Synapses(sr, sbl, clock=Clock(0.2*ms), model='''
                g_a_syn_post = g_a: 1 (summed)
               
                ''',
		on_pre='''
		z+= g_synmax
		''')
syn_rr.connect(i=[0],j=[0])
syn_rr.g_synmax=g_synmaxv


syn_ll=Synapses(sl, sbr, clock=Clock(0.2*ms), model='''
                g_a_syn_post = g_a: 1 (summed)
               
                ''',
		on_pre='''
		z+= g_synmax
		''')
syn_ll.connect(i=[0],j=[0])
syn_ll.g_synmax=g_synmaxv


syn_r = Synapses(sbr, bug, clock=Clock(0.2*ms), on_pre='motorr += we')
syn_r.connect(i=[0],j=[0])


syn_l = Synapses(sbl, bug, clock=Clock(0.2*ms), on_pre='motorl += we')
syn_l.connect(i=[0],j=[0])



#Sensor 2

syn_rr2=Synapses(sr2, sbl2, clock=Clock(0.2*ms), model='''
                g_a_syn_a_post = g_a_a: 1 (summed)
               
                ''',
		on_pre='''
		z_a+= g_synmax_a
		''')
syn_rr2.connect(i=[0],j=[0])
syn_rr2.g_synmax_a=g_synmaxv_a


syn_ll2=Synapses(sl2, sbr2, clock=Clock(0.2*ms), model='''
                g_a_syn_a_post = g_a_a: 1 (summed)
               
                ''',
		on_pre='''
		z_a+= g_synmax_a
		''')
syn_ll2.connect(i=[0],j=[0])
syn_ll2.g_synmax_a=g_synmaxv_a

syn_r2 = Synapses(sbr2, bug, clock=Clock(0.2*ms), on_pre='motorr += we')
syn_r2.connect(i=[0],j=[0])

syn_l2 = Synapses(sbl2, bug, clock=Clock(0.2*ms), on_pre='motorl += we')
syn_l2.connect(i=[0],j=[0])


# f = figure(1)
# bug_plot = plot(bug.x, bug.y, 'ko')
# food_plot = plot(foodx, foody, 'b*')
# sr_plot = plot([0], [0], 'w')   # Just leaving it blank for now
# sl_plot = plot([0], [0], 'w')
# Additional update rules (not covered/possible in above eqns)



@network_operation()
def update_positions(t):
    global foodx, foody, food_count, foodx2, foody2, food_a_count, foodx3, foody3, food_l_count
    

    # 1
    sr.x = bug.x + sr.x_disp*sin(bug.angle)+ sr.y_disp*cos(bug.angle) 
    sr.y = bug.y + - sr.x_disp*cos(bug.angle) + sr.y_disp*sin(bug.angle) 

    sl.x = bug.x +  sl.x_disp*sin(bug.angle)+sl.y_disp*cos(bug.angle)
    sl.y = bug.y  - sl.x_disp*cos(bug.angle)+sl.y_disp*sin(bug.angle)
    
    # 2
    sr2.x = bug.x + sr2.x_disp*sin(bug.angle)+ sr2.y_disp*cos(bug.angle) 
    sr2.y = bug.y + - sr2.x_disp*cos(bug.angle) + sr2.y_disp*sin(bug.angle) 

    sl2.x = bug.x +  sl2.x_disp*sin(bug.angle)+sl2.y_disp*cos(bug.angle)
    sl2.y = bug.y  - sl2.x_disp*cos(bug.angle)+sl2.y_disp*sin(bug.angle) 
    

    if ((bug.x-foodx)**2+(bug.y-foody)**2) < 16 and t>=2000*ms:
        food_count += 1
        foodx = randint(-map_size+10, map_size-10)
        foody = randint(-map_size+10, map_size-10)
    if ((bug.x-foodx)**2+(bug.y-foody)**2) < 16 and t<2000*ms:
        food_a_count = 0
        foodx = randint(-map_size+10, map_size-10)
        foody = randint(-map_size+10, map_size-10)
    if ((bug.x-foodx2)**2+(bug.y-foody2)**2) < 16 and t<=2000*ms:
        food_a_count += 1
        foodx2 = randint(-map_size+10, map_size-10)
        foody2 = randint(-map_size+10, map_size-10)
    if ((bug.x-foodx2)**2+(bug.y-foody2)**2) < 16 and t>2000*ms:
        food_count = 0
        foodx2 = randint(-map_size+10, map_size-10)
        foody2 = randint(-map_size+10, map_size-10)
    if (bug.x < -map_size):
        bug.x = -map_size
        bug.angle = pi - bug.angle
    if (bug.x > map_size):
        bug.x = map_size
        bug.angle = pi - bug.angle
    if (bug.y < -map_size):
        bug.y = -map_size
        bug.angle = -bug.angle
    if (bug.y > map_size):
    	bug.y = map_size
    	bug.angle = -bug.angle
    
    sr.foodxx = foodx
    sr.foodyy = foody
    sr.foodcount = food_count
    sl.foodxx = foodx
    sl.foodyy = foody
    sl.foodcount = food_count
    
    sr2.foodxx2 = foodx2
    sr2.foodyy2 = foody2
    sr2.foodcount1 = food_a_count
    sl2.foodxx2 = foodx2
    sl2.foodyy2 = foody2
    sl2.foodcount1 = food_a_count
    
    sbr.foodxx = foodx
    sbr.foodyy = foody
    sbr.foodcount = food_count
    sbl.foodxx = foodx
    sbl.foodyy = foody
    sbl.foodcount = food_count
    
    sbr2.foodxx2 = foodx2
    sbr2.foodyy2 = foody2
    sbr2.foodcount1 = food_a_count
    sbl2.foodxx2 = foodx2
    sbl2.foodyy2 = foody2
    sbl2.foodcount1 = food_a_count

@network_operation(dt=2*ms)
def update_plot(t):
    global foodx, foody, bug_plot, food_plot, sr_plot, sl_plot,outbugx,outbugy,outbugang,outfoodx,outfoody,outsrx,outsry,outslx,outsly,outfoodx2, outfoody2, outsrx, outsry, outsrx2, outsry2, outslx, outsly, outslx2, outsly2
    indx=int(.5*t/ms+1)
    indx2=int(.5*t/ms+1)
    # bug_plot[0].remove()
    # food_plot[0].remove()
    # sr_plot[0].remove()
    # sl_plot[0].remove()
    bug_x_coords = [bug.x, bug.x-4*cos(bug.angle), bug.x-8*cos(bug.angle)]    # ant-like body
    bug_y_coords = [bug.y, bug.y-4*sin(bug.angle), bug.y-8*sin(bug.angle)]
    outbugx[indx-1]=bug.x[0]
    outbugy[indx-1]=bug.y[0]
    outbugang[indx-1]=bug.angle[0]
    outfoodx[indx-1]=foodx
    outfoody[indx-1]=foody
    outsrx[indx-1]=sr.x[0]
    outsry[indx-1]=sr.y[0]
    outslx[indx-1]=sl.x[0]
    outsly[indx-1]=sl.y[0]
    
    outaggx[indx2 - 1] = foodx2
    outaggy[indx2 - 1] = foody2
    outaggsrx[indx2 - 1] = sr2.x[0]
    outaggsry[indx2 - 1] = sr2.y[0]
    outaggslx[indx2 - 1] = sl2.x[0]
    outaggsly[indx2 - 1] = sl2.y[0]
    

    # bug_plot = plot(bug_x_coords, bug_y_coords, 'ko')     # Plot the bug's current position
    # sr_plot = plot([bug.x, sr.x], [bug.y, sr.y], 'b')
    # sl_plot = plot([bug.x, sl.x], [bug.y, sl.y], 'r')
    # food_plot = plot(foodx, foody, 'b*')
    # axis([-100,100,-100,100])
    # draw()
    # print "."
    # pause(0.01)

ML = StateMonitor(sl, ('v', 'r1', 'I', 'foodcount'), record=True)
ML2 = StateMonitor(sl2, ('v', 'r1', 'I', 'foodcount1'), record=True)
MR = StateMonitor(group1, ('r','Iconct'), record=True)
MR2 = StateMonitor(group2, ('r','Iconct'), record=True)
MLL = StateMonitor(sbl, ('v', 'r1', 'I','plasticity'), record=True)
MLL2 = StateMonitor(sbl2, ('v', 'r1', 'I','plasticity1'), record=True)
#M3=StateMonitor(syn_rr,('apre','apost','we'),record=True)

# MR = StateMonitor(sr, ('v', 'I'), record=True)
# MRR = StateMonitor(sbr, ('v'), record=True)
# MLL = StateMonitor(sbl, ('v'), record=True)
MB = StateMonitor(bug, ('motorl', 'motorr'), record = True)
run(duration * ms, report='text')
figure(1)
plot(ML2.t/ms, ML2.v[0])
plot(ML.t/ms, ML.v[0])
# figure(2)
# plot(MR.t/ms, MR.r[0])
# plot(MR2.t/ms, MR2.r[0])
figure(3)
plot(MLL2.t/ms, MLL2.v[0])
plot(MLL.t/ms, MLL.v[0])

figure(4)
plot(MB.t/ms, MB.motorl[0])
plot(MB.t/ms, MB.motorr[0])
figure(5)
plot(MLL.t/ms, MLL.plasticity[0])
plot(MLL2.t/ms, MLL2.plasticity1[0])
# figure(3)
# plot(M3.t/ms, M3.we[0])
# plot(M3.t/ms, M3.apre[0])
# plot(M3.t/ms, M3.apost[0])

np.save('outbugx', outbugx)
np.save('outbugy', outbugy)
np.save('outbugang', outbugang)
np.save('outfoodx', outfoodx)
np.save('outfoody', outfoody)
np.save('outsrx', outsrx)
np.save('outsry', outsry)
np.save('outslx', outslx)
np.save('outsly', outsly)
np.save('outaggx', outaggx)
np.save('outaggy', outaggy)
np.save('outaggsrx', outaggsrx)
np.save('outaggsry', outaggsry)
np.save('outaggslx', outaggslx)
np.save('outaggsly', outaggsly)