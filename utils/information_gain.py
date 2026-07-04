import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # type: ignore
from scipy.stats import entropy # type: ignore

class InformationGain: 
    def __init__(self, target_col, *feature_cols, data, n_bins=10):
        self.n_bins = n_bins
        self.target_col = target_col
        self.feature_cols = feature_cols
        self.data = data


    def entropy_parent(self): 
        # this return the entropy for the whole dataset (default entropy based on target column)
        # the purpose of supervised segmentation is to find features that lower entropy from the whole dataset
        counts = self.data[self.target_col].value_counts(normalize=True).values
        return entropy(counts, base=2)
    
    def visualize_entropy_parent(self):
        entropy_value = self.entropy_parent()
        proportion = 1  # full proportion range

        p = np.linspace(0, proportion, 500)
        H = np.full_like(p, entropy_value)  # flat line at your entropy value

        fig, ax = plt.subplots(figsize=(7, 4), facecolor='white')

        # Filled area with hatch
        ax.fill_between(p, H, alpha=0.4, color='steelblue', hatch='////', edgecolor='steelblue')
        ax.plot(p, H, color='steelblue', linewidth=1.5)

        # Dotted grid
        ax.set_xticks(np.arange(0, 1.1, 0.2))
        ax.set_yticks(np.arange(0, 1.1, 0.2))
        ax.grid(True, linestyle=':', color='black', linewidth=0.8)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Proportion', fontsize=12)
        ax.set_ylabel('Entropy', fontsize=12)
        ax.set_title('Default entropy for whole dataset (based on target column)', fontsize=13)

        # Thick border
        for spine in ax.spines.values():
            spine.set_linewidth(2)

        plt.tight_layout()
        plt.show()

    def information_gain(self):

        for col in self.feature_cols:
            print(f'Start calculating information gain on {col.upper()}')
            
            categories = self.data[col].unique()
            self.data[col].value_counts(normalize=True).values
            results = []

            for cat in categories:

                subset = self.data[self.data[col] == cat][self.target_col]
                prop = len(subset) / len(self.data)                              # width = prevalence
                ent = entropy(subset.value_counts(normalize=True).values, base=2)  # height = entropy
                results.append({'cat': cat, 'prop': prop, 'entropy': ent})
    
            # sort by proportion ascending (left to right)
            results = sorted(results, key=lambda x: x['prop'])

            fig, ax = plt.subplots(figsize=(7, 4), facecolor='white')

            x_cursor = 0
            for r in results:
                width = r['prop']
                height = r['entropy']

                # hatched bar
                ax.bar(
                    x=x_cursor + width / 2,
                    height=height,
                    width=width,
                    align='center',
                    color='steelblue',
                    alpha=0.4,
                    hatch='////',
                    edgecolor='steelblue',
                    linewidth=1.2
                )

                # circled label in the middle of the bar
                ax.text(
                    x_cursor + width / 2,
                    height / 2,
                    str(r['cat']),
                    ha='center', va='center', fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='circle,pad=0.3', fc='white', ec='black', lw=1.5)
                )

                x_cursor += width

            # grid and styling
            ax.set_xticks(np.arange(0, 1.1, 0.2))
            ax.set_yticks(np.arange(0, 1.1, 0.2))
            ax.grid(True, linestyle=':', color='black', linewidth=0.8)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel('Proportion', fontsize=12)
            ax.set_ylabel('Entropy', fontsize=12)
            ax.set_title(f'Entropy and prevalence of values for {col.upper()}', fontsize=13)

            for spine in ax.spines.values():
                spine.set_linewidth(2)

            print(f'Information Gain of {col} is: ', self.entropy_parent() - sum([r['prop'] * r['entropy'] for r in results]))

            plt.tight_layout()
            plt.show()
            print('\n')
            