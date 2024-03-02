from LoadDataframe import *
from matplotlib import pyplot as plt

import seaborn as sns

def main():
   
    basic_rnn_df = load_dataframe("../logs/BasicRNN_Model")
    basic_rnn_df.loc[:, "Model"] = "Basic RNN"

    gru_df = load_dataframe("../logs/GRU_Model")
    gru_df.loc[:, "Model"] = "GRU"

    lstm_df = load_dataframe("../logs/LSTM_Model")
    lstm_df.loc[:, "Model"] = "LSTM"
    
    df = pd.concat([basic_rnn_df, gru_df, lstm_df])

   
    fig, axes = plt.subplots(2, 2, figsize=(10, 5))
    

    sns.lineplot(data=df, x="Epoch", y="train accuracy", ax=axes[0][0], markers=True, hue="Model")
    axes[0][0].set_title("Train accuracy")
    axes[0][0].set_ylabel("Accuracy")

    sns.lineplot(data=df, x="Epoch", y="test accuracy", ax=axes[1][0], markers=True, hue="Model")
    axes[1][0].set_title("Test accuracy")
    axes[1][0].set_ylabel("Accuracy")

    sns.lineplot(data=df, x="Epoch", y="train loss", ax=axes[0][1], markers=True, hue="Model")
    axes[0][1].set_title("Train loss")
    axes[0][1].set_ylabel("Loss")

    sns.lineplot(data=df, x="Epoch", y="test loss", ax=axes[1][1], markers=True, hue="Model")
    axes[1][1].set_title("Test loss")
    axes[1][1].set_ylabel("Loss")

    # grid
    for ax in axes.flatten():
        ax.grid()

    plt.tight_layout()
    plt.savefig("./Plots/AccuracyLoss.png", bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")