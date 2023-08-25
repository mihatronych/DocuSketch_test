import matplotlib.pyplot as plt
import pandas as pd
import glob


class DrawingPlots:
    def draw_plots(self, json_path):
        plt.rcParams["figure.figsize"] = (10, 8)
        df = pd.read_json(json_path)

        dfa = df.gt_corners.value_counts()

        # plot
        ax = dfa.plot(kind='bar', x='gt_corners', y='%', rot=0, grid=True,
                      ylabel='Amount', xlabel='Number of corners',
                      title='Percentage of samples per number of corners')
        plt.savefig('plots/samples_amount_per_corners_num.png')

        plt.clf()

        dfa = df.gt_corners.value_counts()

        # plot
        ax = dfa.plot(kind='pie', x='gt_corners', y='%', rot=0, grid=True,
                      ylabel='Percentage', xlabel='Number of corners',
                      title='Percentage of samples per number of corners',
                      autopct='%.2f')
        plt.savefig('plots/samples_percentage_per_corners_num.png')
        plt.clf()

        df['name'] = pd.factorize(df['name'])[0]
        corr = df.corr()
        corr.style.background_gradient(cmap='coolwarm')
        fig, ax = plt.subplots()
        ax.matshow(corr)
        plt.xticks(range(len(corr.columns)), corr.columns, rotation='vertical')
        plt.yticks(range(len(corr.columns)), corr.columns)
        plt.savefig('plots/columns_correlation_heatmap.png')
        plt.clf()

        # determine where Predicted equals Actual
        df['Match'] = df.rb_corners.eq(df.gt_corners)

        # display(df.head())

        # groupby and get percent
        dfa = df.groupby('gt_corners').Match.mean().mul(100).round(2)

        # display(dfa.head())

        # plot
        ax = dfa.plot(kind='bar', x='gt_corners', y='%', rot=0, legend=False, grid=True,
                      ylabel='Percent %', xlabel='Number of corners', title='Accuracy Rate % per number of corners')
        plt.savefig('plots/accuracy_per_corners_num.png')
        plt.clf()
        paths = glob.glob("plots/*")
        plt.cla()
        return paths


