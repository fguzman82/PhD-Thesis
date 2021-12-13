import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['axes.linewidth'] = 0.0  # set the value globally
plt.rcParams.update({'font.size': 5})
plt.rc("font", family="sans-serif")
plt.rc("axes.spines", top=True, right=True, left=True, bottom=True)
# Some example data to display
scale = 0.99

# fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(7.5, 3))
fig, axs = plt.subplots(nrows=1, ncols=2)
axs = np.reshape(axs, (1, 2))
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.03, hspace=0.03)
print(axs.shape)

axs[0,0].spines['top'].set_visible(False)
axs[0,0].spines['right'].set_visible(False)
axs[0,0].spines['bottom'].set_visible(False)
axs[0,0].spines['left'].set_visible(False)
axs[0,0].set_ylabel('MP',fontsize=15)
axs[0,0].set_xticks([])
axs[0,0].set_yticks([])
axs[0,0].imshow(np.random.rand(224, 224))
# axs[0, 0].set_title('Axis [0, 0]')
im = axs[0,1].imshow(np.random.rand(224, 224))

# l, b, w, h = axs[1].get_position().bounds
# w_cbar = 0.009
# h_cbar = h * 0.9  # scale
# b_cbar = b
# l_cbar = l + scale * w + 0.001
# cbaxes = fig.add_axes([l_cbar + 0.015, b_cbar + 0.015, w_cbar, h_cbar])


# cax = fig.add_axes([0.9, 0.19, 0.095, 0.5])
# cbar = fig.colorbar(im, cax=cax)
# cbar.outline.set_visible(True)
# cbar.ax.tick_params(labelsize=15, width=0.5, length=1.2, direction='inout', pad=0.5)


# cbar.set_ticks([-1, 0, 1])
# cbar.set_ticklabels([-1, 0, 1])
# axs[0, 1].set_title('Axis [0, 1]')
# axs[1, 0].imshow(np.zeros((224, 224)))
# axs[1, 0].set_title('Axis [1, 0]')
# axs[1, 1].imshow(np.zeros((224, 224)))
# axs[1, 1].set_title('Axis [1, 1]')


# Hide x labels and tick labels for top plots and y ticks for right plots.
# for ax in axs.flat:
#     ax.label_outer()

plt.show()

# plt.rcParams["figure.figsize"] = [7.00, 3.50]
# plt.rcParams["figure.autolayout"] = True
# data = np.random.rand(4, 4)
# im = plt.imshow(data, cmap="copper")
# cbar = plt.colorbar(im)
# cbar.set_ticks([0.2, 0.4, 0.6, 0.8])
# cbar.set_ticklabels(["A", "B", "C", "D"])
# plt.show()