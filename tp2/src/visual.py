import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns


def generate_heatmap_animation(csv_file_path, population_size=10):
    df = pd.read_csv(csv_file_path)
    df['generation'] = df.index // population_size  # Adjust for actual population size

    fig, ax = plt.subplots(figsize=(10, 8))
    gen_data = df[df['generation'] == 0].select_dtypes(include=[float, int])

    # Create the initial heatmap without clearing the axis in each frame
    heatmap = sns.heatmap(gen_data, annot=False, cmap="viridis", ax=ax, cbar=True)

    ax.set_title("Heatmap of Generation 0")
    ax.set_xlabel("Attributes")
    ax.set_ylabel("Individuals")

    # Set ticks and labels once
    ax.set_xticks(range(len(gen_data.columns)))
    ax.set_xticklabels(gen_data.columns, rotation=90)

    tick_step = max(population_size // 10, 1)  # Adjust step size based on population
    ticks = list(range(population_size, 0, -tick_step))
    ax.set_yticks(ticks)
    ax.set_yticklabels([str(population_size - tick + 1) for tick in ticks])

    # Get the QuadMesh object from the heatmap
    quadmesh = heatmap.get_children()[0]

    def animate(generation):
        gen_data = df[df['generation'] == generation].select_dtypes(include=[float, int])
        heatmap_data = gen_data.values.flatten()  # Flatten the data array for set_array()
        quadmesh.set_array(heatmap_data)  # Update the heatmap with new data
        ax.set_title(f"Heatmap of Generation {generation}")

    num_generations = df['generation'].max() + 1
    ani = FuncAnimation(fig, animate, frames=num_generations, repeat=False)
    ani.save('../output/population_heatmaps.mp4', writer='ffmpeg', fps=100)
    plt.close(fig)




def generate_barplot_animation(csv_file_path, population_size=10):
    df = pd.read_csv(csv_file_path)
    df['generation'] = df.index // population_size  # Assuming each 10 rows is a new generation

    # store the character in a new column
    character = df['character'].iloc[0]
    # Normalize the height by scaling it for visualization purposes
    df['height'] = df['height'] * 100  # Scale height to match other genotype values

    # Select the genotype related columns
    genotype_columns = ['strength', 'dexterity', 'intelligence', 'vigor', 'constitution', 'height']

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [3, 1]})
    plt.subplots_adjust(wspace=0.3)
    # Create a secondary y-axis for the height scale
    ax3 = ax1.twinx()

    # Pre-calculate the minimum, maximum, and mean of fitness per generation
    fitness_stats = df.groupby('generation')['fitness'].agg(['mean', 'min', 'max'])

    def animate(generation):
        ax1.clear()
        ax2.clear()
        ax3.clear()  # Clear the secondary axis to ensure it's correctly setup each frame

        # Filter data for the current generation
        gen_data = df[df['generation'] == generation][genotype_columns]
        character = df['character'].iloc[0]
        # Calculate mean and standard deviation for the error bars
        means = gen_data.mean()
        errors = gen_data.std()

        # Create the bar plot with error bars in the first subplot
        bars = means.plot(kind='bar', yerr=errors, ax=ax1, capsize=4, color=['mediumaquamarine']*5 + ['skyblue'], ecolor='black')
        ax1.set_xticklabels(genotype_columns, rotation=0, horizontalalignment='center')

        # Setup the secondary y-axis again after clear
        ax3.set_ylim(0, 200)  # Since we multiplied height by 100, adjust accordingly
        ax3.set_ylabel('Altura (cm)')
        ax3.yaxis.set_label_position('right')
        #ax3.yaxis.label.set_color('skyblue')
        
        # Adding fitness plot in the second subplot
        x_values = fitness_stats.index[:generation + 1]
        mean_values = fitness_stats['mean'][:generation + 1]
        min_values = fitness_stats['min'][:generation + 1]
        max_values = fitness_stats['max'][:generation + 1]

        # Plot mean fitness and fill area between min and max
        ax2.plot(x_values, mean_values, color='forestgreen', linestyle='-', linewidth=2)
        ax2.fill_between(x_values, min_values, max_values, color='forestgreen', alpha=0.3)

        # Add grid to the second subplot
        ax2.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')

        ax2.set_xlim(0, df['generation'].max())
        ax2.set_ylim(0, 100)
        ax2.set_title("Fitness en el Tiempo")
        ax2.set_xlabel("Generación")
        ax2.set_ylabel("Fitness")

        # Set plot details for the first subplot
        ax1.set_title(f"Valores Promedio del Genotipo del {character} - Generación {generation}")
        ax1.set_xlabel("Atributos del Genotipo")
        ax1.set_ylabel("Puntos")
        ax1.set_ylim(0, 200)  # Adjust ylim to show error bars clearly

    num_generations = df['generation'].max() + 1
    ani = FuncAnimation(fig, animate, frames=num_generations, repeat=False)


    ani.save('../output/genotype_changes_with_fitness.mp4', writer='ffmpeg', fps=100)
    plt.close(fig) 



# cuando los haces tenes que poner el population size a mano sino vas a hacer cagadas,
# despues podemos configurarlo para que reciba el config y lo deduzca del population size directamente
csv_file_path = '../output/population_evolution.csv'  
#generate_barplot_animation(csv_file_path, 20)
generate_heatmap_animation(csv_file_path, 20)
