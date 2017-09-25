import matplotlib.pyplot as plt
plt.style.use('ggplot')

x, y = [], []

with open('testjack_snps.txt') as F:
    for line in F:
        if line.strip() == '':
            continue
        bl = line.strip().split()
        y.append(int(bl[0]))
        a = bl[1][bl[1].find('_')+1:bl[1].find('.')]
        x.append(int(a))

plt.figure()
plt.scatter(x,y, alpha=0.5)
plt.xlabel('Jackknife index')
plt.ylabel('Number of snp pairs')
plt.savefig('testjackSNPloss.pdf')