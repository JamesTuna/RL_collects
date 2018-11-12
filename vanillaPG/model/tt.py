#! /usr/bin/env python3
import matplotlib.pyplot as plt
l_r = []
l_run_r = []
with open('./record.txt','r') as f:
	for line in f:
		if line.startswith('episode:'):
			a = line.split(' ')
			ep,r,run_r = a[1],a[5],a[-1][:-1]
			l_r.append(int(r))
			l_run_r.append(int(run_r))
print(l_r,l_run_r)
plt.plot([i for i in range(1,len(l_r)+1)],l_r,'r-')
plt.plot([i for i in range(1,len(l_run_r)+1)],l_run_r,'b-')
plt.xlabel('episodes')
plt.ylabel('reward')
plt.show()