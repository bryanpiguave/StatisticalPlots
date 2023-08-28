
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from scipy.stats import multivariate_normal
from matplotlib.lines import Line2D



"""
    Linear regression animation 

"""
def main():
    # Set the random seed for reproducibility
    np.random.seed(0)

    # Generate correlated random data
    mean = [0, 0]
    cov = [[1, 0.8], [0.8, 1]]  # Covariance matrix for linear correlation
    num_samples = 2000
    data = np.random.multivariate_normal(mean, cov, num_samples)

    # Separate the variables
    x = data[:, 0]
    y = data[:, 1]

    # Create a 3D scatter plot
    fig = plt.figure()
    fig.patch.set_facecolor('black')
    ax = fig.add_subplot(111, projection='3d')
    # Rotate the plot by 30 degrees to the left
    ax.view_init(elev=20., azim=-30)

    # Scatter plot of data
    ax.scatter(x, y, np.zeros_like(x), marker='o',alpha=0.5,color="skyblue")

    # Create mesh grid
    x_vals = np.linspace(min(x), max(x), 30)
    y_vals = np.linspace(min(y), max(y), 30)
    x_vals, y_vals = np.meshgrid(x_vals, y_vals)

    # Calculate joint PDF values for each point in the grid
    pos = np.empty(x_vals.shape + (2,))
    pos[:, :, 0] = x_vals
    pos[:, :, 1] = y_vals
    pdf_vals = multivariate_normal(mean, cov).pdf(pos)

    # Plot the mesh grid
    cmap = plt.get_cmap('coolwarm')  # Choose a different colormap

    mesh = ax.plot_surface(x_vals, y_vals, pdf_vals, cmap=cmap, alpha=0.5)

    # Best fit regression line
    x_regression = np.linspace(min(x), max(x), 100)
    y_regression = multivariate_normal(mean, cov).mean[1] + cov[1][0] / cov[0][0] * (x_regression - mean[0])
    z_regression = np.zeros_like(x_regression)

    regression_line=ax.plot(x_regression, y_regression, z_regression, color='blue', linewidth=3, label='Regression Line')


    legend_elements = [Line2D([0], [0], color='red', label=r'Conditional pdf $f(y|x)$'),
                    Line2D([0], [0], color='blue', label=r'Regression Line $y = \beta_0+\beta_1*x$'),
                    Line2D([0], [0], color='brown', label=r'Conditional expectation $E[Y|X=x]$',marker="_") ]

    # Add legend
    ax.legend(handles=legend_elements)


    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Joint pdf f(x,y)')
    ax.set_title('Linear regression',color="white")

    ax.set_facecolor('black')
    ax.w_xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.w_yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.w_zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.grid(True,alpha=0.5)
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.zaxis.label.set_color('white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.tick_params(axis='z', colors='white')


    # Identity line
    conditional_line, = ax.plot([], [], [], color='red', linewidth=2,alpha=0.8)


    # Conditional expectation
    conditional_expectation, = ax.plot([], [], [], color='brown', linewidth=2,marker="_")
    # Animation update function
    def animate(frame):
        x_line = (frame / 20.0) * (max(x) - min(x)) + min(x)  # Move along the x-axis
        mean_y_given_x = mean[1] + cov[1][0] / cov[0][0] * (x_line - mean[0])
        
        y_line = np.linspace(min(y), max(y), 100)
        z_line = multivariate_normal(mean, cov).pdf(np.column_stack((np.full(len(y_line), x_line), y_line)))
        
        conditional_line.set_data(np.array([x_line] * len(y_line)), y_line)
        conditional_line.set_3d_properties(z_line)
        conditional_expectation.set_data(np.array([x_line, x_line]), np.array([mean_y_given_x, mean_y_given_x]))
        conditional_expectation.set_3d_properties([0, multivariate_normal(mean, cov).pdf([x_line, mean_y_given_x])])
        
        return conditional_line, conditional_expectation


    # Create animation
    ani = FuncAnimation(fig, animate, frames=20, interval=120, blit=True)
    # Adjust grid line transparency
    for line in ax.xaxis.get_gridlines():
        line.set_alpha(0.2)
    for line in ax.yaxis.get_gridlines():
        line.set_alpha(0.2)
    for line in ax.zaxis.get_gridlines():
        line.set_alpha(0.2)

    watermark_text = 'Bryan Piguave'
    ax.annotate(watermark_text, xy=(0.37, 0.5), xycoords='axes fraction', fontsize=10, color='gray', ha='right')

    plt.tight_layout()
    ani.save('joint_distribution_animation.gif', writer='imagemagick',dpi=120)
    plt.show()

if __name__=="__main__":
    main()
