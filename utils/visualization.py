import matplotlib.pyplot as plt
import pandas as pd


tick_size=15
xlabel_size=15
title_size=20

mosaic = '''
ab
'''

df_svd = pd.read_csv('results\comp_acc_svd_mnist.csv')
df_cur = pd.read_csv('results\comp_acc_cur_mnist.csv')

compr_svd = df_svd['compr']
accuracy_svd = df_svd['accuracy']
params_svd = df_svd['parameters']

compr_cur = df_cur['compr']
accuracy_cur = df_cur['accuracy']
params_cur = df_cur['parameters']


# Accuracy depending on compression rate

fig, ax = plt.subplot_mosaic(mosaic)

ax['a'].plot(compr_svd, accuracy_svd, c='orange')
ax['a'].set_title('SVD VGG-11', fontsize=title_size)
ax['a'].tick_params(axis='both', labelsize=tick_size)
ax['a'].set_xlabel("Compression rate", fontsize=xlabel_size)
ax['a'].set_ylabel("Accuracy", fontsize=xlabel_size)
ax['a'].grid(True)


ax['b'].plot(compr_cur, accuracy_cur, c='b')
ax['b'].set_title('CUR VGG-11', fontsize=title_size)
ax['b'].tick_params(axis='both', labelsize=tick_size)
ax['b'].set_xlabel("Compression rate", fontsize=xlabel_size)
ax['b'].set_ylabel("Accuracy", fontsize=xlabel_size)
ax['b'].grid(True)

fig.set_size_inches(18.5, 6.5)
fig.suptitle('Accuracy depending on compression rate', fontsize=title_size)
fig.tight_layout(pad=1.1)

fig.savefig('plots/accuracy')
plt.show()

# Parameters depending on compression rate

fig, ax = plt.subplot_mosaic(mosaic)

ax['a'].plot(compr_svd, params_svd, c='orange')
ax['a'].set_title('SVD VGG-11', fontsize=title_size)
ax['a'].tick_params(axis='both', labelsize=tick_size)
ax['a'].set_xlabel("Compression rate", fontsize=xlabel_size)
ax['a'].set_ylabel("Parameters", fontsize=xlabel_size)
ax['a'].grid(True)

ax['b'].plot(compr_cur, params_cur, c='b')
ax['b'].set_title('CUR VGG-11', fontsize=title_size)
ax['b'].tick_params(axis='both', labelsize=tick_size)
ax['b'].set_xlabel("Compression rate", fontsize=xlabel_size)
ax['b'].set_ylabel("Parameters", fontsize=xlabel_size)
ax['b'].grid(True)

fig.set_size_inches(18.5, 6.5)
fig.suptitle('Parameters depending on compression rate', fontsize=title_size)
fig.tight_layout(pad=1.1)

fig.savefig('plots/parameters')
plt.show()




